from __future__ import annotations

import json
import shutil
import sys
import textwrap
from pathlib import Path

from darjeeling.errors import ArtifactError

_PYTHON_SANDBOX_RUNNER = r"""
import json
import os
import runpy
import socket
import subprocess
import sys
import sysconfig
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
command = config["command"]
cwd = Path(config["cwd"]).resolve()


def _roots(values):
    roots = []
    for value in values:
        try:
            roots.append(Path(value).expanduser().resolve(strict=False))
        except OSError:
            continue
    return roots


allowed_read_roots = _roots(config["allowed_read_roots"])
allowed_write_roots = _roots(config["allowed_write_roots"])
denied_read_roots = _roots(config["denied_read_roots"])
denied_write_roots = _roots(config["denied_write_roots"])
allow_network = bool(config.get("allow_network", False))

system_roots = []
for value in set(sys.path + [sys.prefix, sys.base_prefix, sys.exec_prefix]):
    if not value:
        continue
    try:
        system_roots.append(Path(value).expanduser().resolve(strict=False))
    except OSError:
        pass
for key in ["stdlib", "platstdlib", "purelib", "platlib", "scripts"]:
    value = sysconfig.get_paths().get(key)
    if value:
        system_roots.append(Path(value).expanduser().resolve(strict=False))
for value in ["/dev/null", "/dev/urandom", "/dev/random"]:
    path = Path(value)
    if path.exists():
        system_roots.append(path)


def _resolve_path(value):
    if value is None or isinstance(value, int):
        return None
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    if not isinstance(value, str):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def _contains(root, path):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return path == root


def _inside(path, roots):
    return any(_contains(root, path) for root in roots)


def _is_write(mode, flags):
    if isinstance(mode, str) and any(part in mode for part in ["w", "a", "x", "+"]):
        return True
    if isinstance(flags, int):
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        return bool(flags & write_flags)
    return False


def _check_read(path):
    if path is None:
        return
    if _inside(path, allowed_read_roots) or _inside(path, system_roots):
        return
    if _inside(path, denied_read_roots):
        raise PermissionError(f"read denied by Darjeeling sandbox: {path}")
    raise PermissionError(f"read outside Darjeeling sandbox: {path}")


def _check_write(path):
    if path is None:
        return
    if _inside(path, denied_write_roots):
        raise PermissionError(f"write denied by Darjeeling sandbox: {path}")
    if _inside(path, allowed_write_roots):
        return
    raise PermissionError(f"write outside Darjeeling sandbox: {path}")


def _audit(event, args):
    if event == "open":
        path = _resolve_path(args[0] if args else None)
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else None
        if _is_write(mode, flags):
            _check_write(path)
        else:
            _check_read(path)
        return
    if event in {"os.listdir", "os.scandir", "os.stat", "os.lstat"}:
        _check_read(_resolve_path(args[0] if args else None))
        return
    if event in {
        "os.mkdir",
        "os.rmdir",
        "os.remove",
        "os.unlink",
        "os.rename",
        "os.replace",
        "os.symlink",
        "os.link",
        "os.chmod",
        "os.chown",
    }:
        for value in args[:2]:
            _check_write(_resolve_path(value))
        return
    if event.startswith("socket.") and not allow_network:
        raise PermissionError(f"{event} denied by Darjeeling sandbox")
    if event in {"subprocess.Popen", "os.system", "os.exec"}:
        raise PermissionError(f"{event} denied by Darjeeling sandbox")


sys.addaudithook(_audit)
os.chdir(cwd)

args = command[1:]
if not args:
    raise SystemExit("Python sandbox command requires script, -m, or -c")
if args[0] == "-c":
    code = args[1] if len(args) > 1 else ""
    sys.argv = ["-c", *args[2:]]
    exec(compile(code, "<darjeeling-sandbox-command>", "exec"), {"__name__": "__main__"})
elif args[0] == "-m":
    if len(args) < 2:
        raise SystemExit("Python -m command requires a module")
    sys.argv = [args[1], *args[2:]]
    runpy.run_module(args[1], run_name="__main__", alter_sys=True)
else:
    script = args[0]
    sys.argv = [script, *args[1:]]
    runpy.run_path(script, run_name="__main__")
"""


def is_python_command(command: list[str]) -> bool:
    if not command:
        return False
    executable = Path(command[0]).name.lower()
    if executable in {"python", "python3"}:
        return True
    return executable.startswith("python3.") or executable.startswith("python.")


def resolve_python_command(command: list[str]) -> list[str]:
    if not is_python_command(command):
        raise ArtifactError("portable sandbox only supports Python commands")
    resolved = list(command)
    executable = shutil.which(resolved[0]) if not Path(resolved[0]).is_absolute() else resolved[0]
    if executable is None:
        raise ArtifactError(f"Python executable not found: {resolved[0]}")
    resolved[0] = executable
    return resolved


def build_python_sandbox_command(
    command: list[str],
    *,
    cwd: Path,
    config_path: Path,
    allowed_read_roots: list[Path],
    allowed_write_roots: list[Path],
    denied_read_roots: list[Path],
    denied_write_roots: list[Path],
    allow_network: bool = False,
) -> list[str]:
    resolved_command = resolve_python_command(command)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "command": resolved_command,
        "cwd": str(cwd),
        "allowed_read_roots": [str(path) for path in allowed_read_roots],
        "allowed_write_roots": [str(path) for path in allowed_write_roots],
        "denied_read_roots": [str(path) for path in denied_read_roots],
        "denied_write_roots": [str(path) for path in denied_write_roots],
        "allow_network": allow_network,
    }
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    runner = textwrap.dedent(_PYTHON_SANDBOX_RUNNER).strip()
    return [sys.executable, "-c", runner, str(config_path)]
