from __future__ import annotations

import json
import selectors
import subprocess
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from darjeeling.runtime.timing import elapsed_ms
from darjeeling.schemas import Frame, LayerResult

DEFAULT_BENCHMARK_UTTERANCES = (
    "set an alarm for seven tomorrow morning",
    "what is the weather",
    "play some jazz",
)


class RustL1BuildError(RuntimeError):
    pass


class RustL1WorkerError(RuntimeError):
    pass


class RustL1Response(BaseModel):
    request_id: str
    accepted: bool
    frame: Frame | None = None
    program_path: str = ""
    native_latency_us: int = 0
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


def binary_path_for(crate_dir: Path, *, release: bool = False) -> Path:
    profile = "release" if release else "debug"
    suffix = ".exe" if sys.platform == "win32" else ""
    return crate_dir / "target" / profile / f"darjeeling-l1-programbank{suffix}"


def build_l1_binary(crate_dir: Path, *, release: bool = False) -> Path:
    command = ["cargo", "build"]
    if release:
        command.append("--release")
    result = subprocess.run(command, cwd=crate_dir, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RustL1BuildError(result.stderr.strip() or result.stdout.strip())
    binary_path = binary_path_for(crate_dir, release=release)
    if not binary_path.exists():
        raise RustL1BuildError(f"cargo succeeded but binary is missing: {binary_path}")
    return binary_path


class RustL1Worker:
    def __init__(self, binary_path: Path, *, timeout_s: float = 5.0) -> None:
        self.binary_path = binary_path
        self.timeout_s = timeout_s
        self._process: subprocess.Popen[str] | None = None
        self._selector: selectors.BaseSelector | None = None

    def start(self) -> None:
        if self._process is not None:
            return
        if not self.binary_path.exists():
            raise RustL1WorkerError(f"L1 worker binary does not exist: {self.binary_path}")
        self._process = subprocess.Popen(
            [str(self.binary_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self._process.stdout is None:
            raise RustL1WorkerError("L1 worker stdout is unavailable")
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._process.stdout, selectors.EVENT_READ)

    def answer(self, utterance: str, *, request_id: str | None = None) -> RustL1Response:
        self.start()
        process = self._require_process()
        if process.stdin is None or process.stdout is None or self._selector is None:
            raise RustL1WorkerError("L1 worker pipes are unavailable")
        if process.poll() is not None:
            raise RustL1WorkerError(f"L1 worker exited with code {process.returncode}")

        request_id = request_id or f"l1-{uuid.uuid4().hex}"
        process.stdin.write(json.dumps({"request_id": request_id, "utterance": utterance}) + "\n")
        process.stdin.flush()

        events = self._selector.select(timeout=self.timeout_s)
        if not events:
            self.close(kill=True)
            raise RustL1WorkerError(f"L1 worker timed out after {self.timeout_s:.3f}s")
        line = process.stdout.readline()
        if not line:
            raise RustL1WorkerError("L1 worker closed stdout without a response")
        return RustL1Response.model_validate_json(line)

    def close(self, *, kill: bool = False) -> None:
        process = self._process
        selector = self._selector
        self._process = None
        self._selector = None
        if selector is not None:
            selector.close()
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if kill and process.poll() is None:
            process.kill()
        elif process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise RustL1WorkerError("L1 worker has not been started")
        return self._process

    def __enter__(self) -> RustL1Worker:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class RustProgramBankLayer:
    def __init__(self, worker: RustL1Worker) -> None:
        self.worker = worker

    def try_answer(self, utterance: str) -> LayerResult:
        with elapsed_ms() as ms:
            try:
                response = self.worker.answer(utterance)
            except RustL1WorkerError as exc:
                return LayerResult(
                    layer="L1",
                    accepted=False,
                    reason=f"rust worker error: {exc}",
                    latency_ms=ms(),
                )

            return LayerResult(
                layer="L1",
                accepted=response.accepted and response.frame is not None,
                frame=response.frame,
                confidence=1.0 if response.accepted else None,
                reason=response.reason,
                latency_ms=ms(),
                metadata={
                    "request_id": response.request_id,
                    "program_path": response.program_path,
                    "native_latency_us": response.native_latency_us,
                    **response.metadata,
                },
            )


def benchmark_worker(
    binary_path: Path,
    utterances: Iterable[str],
    *,
    timeout_s: float = 2.0,
) -> dict[str, Any]:
    integration_latencies_ms: list[float] = []
    native_latencies_us: list[int] = []
    program_path_counts: dict[str, int] = {}
    accepted = 0
    total = 0
    with RustL1Worker(binary_path, timeout_s=timeout_s) as worker:
        benchmark_started_at = perf_counter()
        for utterance in utterances:
            started_at = perf_counter()
            response = worker.answer(utterance)
            integration_latencies_ms.append((perf_counter() - started_at) * 1000.0)
            native_latencies_us.append(response.native_latency_us)
            accepted += int(response.accepted)
            total += 1
            program_path_counts[response.program_path] = (
                program_path_counts.get(response.program_path, 0) + 1
            )
        elapsed_s = perf_counter() - benchmark_started_at
    return {
        "requests": total,
        "accepted": accepted,
        "accepted_share": accepted / total if total else 0.0,
        "integration_avg_ms": _avg(integration_latencies_ms),
        "integration_p50_ms": _percentile(integration_latencies_ms, 50),
        "integration_p95_ms": _percentile(integration_latencies_ms, 95),
        "native_avg_us": _avg(native_latencies_us),
        "native_p50_us": _percentile(native_latencies_us, 50),
        "native_p95_us": _percentile(native_latencies_us, 95),
        "native_max_us": max(native_latencies_us) if native_latencies_us else 0,
        "throughput_qps": total / elapsed_s if elapsed_s > 0 else 0.0,
        "program_path_counts": dict(sorted(program_path_counts.items())),
    }


def _avg(values: Iterable[float | int]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: Iterable[float | int], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
