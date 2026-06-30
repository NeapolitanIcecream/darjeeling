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
import shutil
import sys
import sysconfig
from pathlib import Path

_CONFIG_ENV_VAR = "DARJEELING_PORTABLE_SANDBOX_CONFIG"
if len(sys.argv) > 1 and sys.argv[1] == "--config-env":
    config = json.loads(os.environ[_CONFIG_ENV_VAR])
else:
    config_path = Path(sys.argv[1]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
command = config["command"]
cwd = Path(config["cwd"]).resolve()
runner_path = Path(config["runner_path"]).resolve()
allow_dependency_install = bool(config.get("allow_dependency_install", False))


def _roots(values):
    roots = []
    for value in values:
        try:
            roots.append(Path(value).expanduser().resolve(strict=False))
        except OSError:
            continue
    return roots


_allowed_read_root_values = tuple(config["allowed_read_roots"])
_allowed_write_root_values = tuple(config["allowed_write_roots"])
_denied_read_root_values = tuple(config["denied_read_roots"])
_denied_write_root_values = tuple(config["denied_write_roots"])
allowed_read_roots = tuple(_roots(_allowed_read_root_values))
allowed_write_roots = tuple(_roots(_allowed_write_root_values))
denied_read_roots = tuple(_roots(_denied_read_root_values))
denied_write_roots = tuple(_roots(_denied_write_root_values))
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


def _command_parts(value):
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return []
    parts = []
    for part in value:
        if isinstance(part, bytes):
            part = os.fsdecode(part)
        if not isinstance(part, str):
            return []
        parts.append(part)
    return parts


def _same_path(left, right):
    try:
        return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)
    except (OSError, TypeError):
        return False


def _child_config_matches_policy(child_config):
    if child_config.get("allowed_read_roots") != list(_allowed_read_root_values):
        return False
    if child_config.get("allowed_write_roots") != list(_allowed_write_root_values):
        return False
    if child_config.get("denied_read_roots") != list(_denied_read_root_values):
        return False
    if child_config.get("denied_write_roots") != list(_denied_write_root_values):
        return False
    if bool(child_config.get("allow_network", False)) != allow_network:
        return False
    if bool(child_config.get("allow_dependency_install", False)) != allow_dependency_install:
        return False
    if not _same_path(child_config.get("runner_path"), runner_path):
        return False
    try:
        child_cwd = Path(child_config["cwd"]).resolve(strict=False)
    except (KeyError, OSError, TypeError):
        return False
    try:
        _check_read(child_cwd)
    except PermissionError:
        return False
    try:
        child_command = _normalize_subprocess_command(child_config.get("command"))
    except PermissionError:
        return False
    return child_config.get("command") == child_command


def _is_sandboxed_child_popen(args):
    if not allow_dependency_install or len(args) < 4:
        return False
    event_command = _command_parts(args[1])
    expected = [sys.executable, "-I", str(runner_path), "--config-env"]
    if len(event_command) != len(expected):
        return False
    if not _same_path(event_command[0], expected[0]):
        return False
    if event_command[1:] != expected[1:]:
        return False
    env = args[3]
    if not isinstance(env, dict):
        return False
    config_json = env.get(_CONFIG_ENV_VAR)
    if not isinstance(config_json, str):
        return False
    try:
        child_config = json.loads(config_json)
    except json.JSONDecodeError:
        return False
    return _child_config_matches_policy(child_config)


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
    if event == "subprocess.Popen":
        if _is_sandboxed_child_popen(args):
            return
        raise PermissionError(f"{event} denied by Darjeeling sandbox")
    if event in {
        "os.system",
        "os.exec",
        "os.posix_spawn",
        "os.fork",
        "os.forkpty",
    }:
        raise PermissionError(f"{event} denied by Darjeeling sandbox")


sys.addaudithook(_audit)


def _is_python_command(value):
    try:
        executable = Path(os.fsdecode(value)).name.lower()
    except TypeError:
        return False
    if executable in {"python", "python3"}:
        return True
    return executable.startswith("python3.") or executable.startswith("python.")


def _resolve_executable(value):
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    if not isinstance(value, str) or not value:
        return None
    if Path(value).is_absolute():
        return value
    resolved = shutil.which(value)
    return resolved or value


def _normalize_subprocess_command(raw_command):
    if isinstance(raw_command, tuple):
        raw_command = list(raw_command)
    if not isinstance(raw_command, list) or not raw_command:
        raise PermissionError(
            "portable dependency installation requires a Python command list"
        )
    command_parts = [
        os.fsdecode(part) if isinstance(part, bytes) else part for part in raw_command
    ]
    if not all(isinstance(part, str) and part for part in command_parts):
        raise PermissionError(
            "portable dependency installation requires a Python command list"
        )
    executable = _resolve_executable(command_parts[0])
    if executable is None or not _is_python_command(executable):
        raise PermissionError(
            "portable dependency installation only allows sandboxed Python subprocesses"
        )
    command_parts[0] = executable
    return command_parts


def _subprocess_cwd(value):
    if value is None:
        return cwd
    path = _resolve_path(value)
    _check_read(path)
    return path


def _child_config(child_command, child_cwd):
    child_config = dict(config)
    child_config["command"] = child_command
    child_config["cwd"] = str(child_cwd)
    child_config["runner_path"] = str(runner_path)
    return json.dumps(child_config, sort_keys=True)


def _child_env(_existing_env, config_json):
    env = {"PATH": os.environ.get("PATH", "")}
    python_unbuffered = os.environ.get("PYTHONUNBUFFERED")
    if python_unbuffered is not None:
        env["PYTHONUNBUFFERED"] = python_unbuffered
    env[_CONFIG_ENV_VAR] = config_json
    return env


_original_popen = subprocess.Popen


def _sandboxed_popen(*popen_args, **popen_kwargs):
    if not allow_dependency_install:
        raise PermissionError("subprocess.Popen denied by Darjeeling sandbox")
    if len(popen_args) > 1:
        raise PermissionError(
            "portable dependency installation only allows command as a positional argument"
        )
    if popen_kwargs.get("shell"):
        raise PermissionError(
            "portable dependency installation does not allow shell subprocesses"
        )
    for key, message in {
        "executable": "executable overrides",
        "preexec_fn": "pre-exec hooks",
    }.items():
        if popen_kwargs.get(key) is not None:
            raise PermissionError(
                f"portable dependency installation does not allow {message}"
            )
    if popen_kwargs.get("pass_fds"):
        raise PermissionError(
            "portable dependency installation does not allow inherited file descriptors"
        )
    if popen_args:
        raw_command = popen_args[0]
        remaining_args = popen_args[1:]
        command_in_kwargs = False
    else:
        raw_command = popen_kwargs.get("args")
        remaining_args = ()
        command_in_kwargs = "args" in popen_kwargs
    child_command = _normalize_subprocess_command(raw_command)
    child_cwd = _subprocess_cwd(popen_kwargs.get("cwd"))
    config_json = _child_config(child_command, child_cwd)
    wrapped_command = [sys.executable, "-I", str(runner_path), "--config-env"]
    wrapped_kwargs = dict(popen_kwargs)
    wrapped_kwargs.pop("shell", None)
    wrapped_kwargs.pop("executable", None)
    wrapped_kwargs.pop("preexec_fn", None)
    wrapped_kwargs.pop("pass_fds", None)
    wrapped_kwargs["env"] = _child_env(wrapped_kwargs.get("env"), config_json)
    if "cwd" in wrapped_kwargs:
        wrapped_kwargs["cwd"] = str(child_cwd)
    if command_in_kwargs:
        wrapped_kwargs["args"] = wrapped_command
        return _original_popen(*remaining_args, **wrapped_kwargs)
    return _original_popen(wrapped_command, *remaining_args, **wrapped_kwargs)


subprocess.Popen = _sandboxed_popen
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


def _path_contains(root: Path, path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


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
    allow_dependency_install: bool = False,
) -> list[str]:
    resolved_command = resolve_python_command(command)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    runner = textwrap.dedent(_PYTHON_SANDBOX_RUNNER).strip()
    runner_path = config_path.with_name(f"{config_path.stem}_runner.py")
    if allow_dependency_install and any(
        _path_contains(root, runner_path) for root in allowed_write_roots
    ):
        raise ArtifactError(
            "portable dependency installation requires a runner outside writable roots"
        )
    runner_path.write_text(runner + "\n", encoding="utf-8")
    config = {
        "command": resolved_command,
        "cwd": str(cwd),
        "runner_path": str(runner_path),
        "allowed_read_roots": [str(path) for path in allowed_read_roots],
        "allowed_write_roots": [str(path) for path in allowed_write_roots],
        "denied_read_roots": [str(path) for path in denied_read_roots],
        "denied_write_roots": [str(path) for path in denied_write_roots],
        "allow_network": allow_network,
        "allow_dependency_install": allow_dependency_install,
    }
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return [sys.executable, "-I", str(runner_path), str(config_path)]
