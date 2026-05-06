import json
import os
import queue
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


_CODE_FENCE_RE = re.compile(r"^```(?:python)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
_TEXT_PREVIEW_EXTENSIONS = {".txt", ".csv", ".tsv", ".json", ".md", ".yaml", ".yml", ".log"}
_ENV_DENYLIST_PATTERNS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "OPENAI",
    "ANTHROPIC",
    "GOOGLE_API_KEY",
    "GEMINI",
    "QWEN",
)


def _strip_code_fences(code: str) -> str:
    code = (code or "").strip()
    if code.startswith("```") and code.endswith("```"):
        code = _CODE_FENCE_RE.sub("", code).strip()
    return code


def _safe_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    clipped = len(text) - limit
    return text[:limit] + f"\n\n...[truncated {clipped} chars]"


@dataclass
class ExecutionArtifact:
    relative_path: str
    absolute_path: str
    size_bytes: int
    kind: str


@dataclass
class ExecutionResult:
    status: str
    stdout: str
    stderr: str
    rich_output: str
    exec_time_sec: float
    artifacts: List[ExecutionArtifact] = field(default_factory=list)
    error_name: Optional[str] = None
    workspace_dir: Optional[str] = None
    input_files: List[str] = field(default_factory=list)

    def to_tool_message(self, max_text_chars: int = 12000, preview_text_files: int = 2) -> str:
        parts = [
            f"STATUS: {self.status}",
            f"EXEC_TIME_SEC: {self.exec_time_sec:.3f}",
        ]

        if self.workspace_dir:
            parts.append(f"WORKSPACE: {self.workspace_dir}")
        if self.input_files:
            parts.append("INPUT_FILES:\n" + "\n".join(f"- {x}" for x in self.input_files))

        if self.stdout.strip():
            parts.append(f"STDOUT:\n```text\n{_safe_text(self.stdout, max_text_chars)}\n```")
        if self.rich_output.strip():
            parts.append(f"RESULTS:\n```text\n{_safe_text(self.rich_output, max_text_chars)}\n```")
        if self.stderr.strip():
            parts.append(f"STDERR:\n```text\n{_safe_text(self.stderr, max_text_chars)}\n```")
        if self.error_name:
            parts.append(f"ERROR_NAME: {self.error_name}")

        if self.artifacts:
            lines = [
                f"- {artifact.relative_path} ({artifact.kind}, {artifact.size_bytes} bytes) -> {artifact.absolute_path}"
                for artifact in self.artifacts
            ]
            parts.append("ARTIFACTS:\n" + "\n".join(lines))

            previews = []
            preview_count = 0
            for artifact in self.artifacts:
                suffix = Path(artifact.relative_path).suffix.lower()
                if suffix not in _TEXT_PREVIEW_EXTENSIONS:
                    continue
                if preview_count >= preview_text_files:
                    break
                preview_count += 1
                try:
                    content = Path(artifact.absolute_path).read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                previews.append(
                    f"PREVIEW {artifact.relative_path}:\n```text\n{_safe_text(content, 2500)}\n```"
                )
            if previews:
                parts.append("\n\n".join(previews))

        return "\n\n".join(parts).strip()


class LocalKernelExecutor:
    """
    Persistent, per-sample Python executor built on a local Jupyter kernel.

    This is intentionally not a security sandbox on par with a VM or container,
    but it is a substantial upgrade over launching arbitrary temp scripts:
    - one fresh workspace per sample
    - one persistent Python state per sample
    - optional resource limits applied inside the kernel process
    - API-like structured results for the model
    """

    def __init__(
        self,
        workspace_root: str = "./workspace/unified_ci",
        startup_timeout_sec: int = 30,
        execution_timeout_sec: int = 45,
        max_output_chars: int = 12000,
        cleanup_workspace_on_success: bool = False,
        cpu_time_limit_sec: Optional[int] = 40,
        max_file_size_mb: Optional[int] = 100,
        max_open_files: Optional[int] = 128,
        max_processes: Optional[int] = 64,
        memory_limit_mb: Optional[int] = None,
    ) -> None:
        self.workspace_root = os.path.abspath(workspace_root)
        self.startup_timeout_sec = startup_timeout_sec
        self.execution_timeout_sec = execution_timeout_sec
        self.max_output_chars = max_output_chars
        self.cleanup_workspace_on_success = cleanup_workspace_on_success
        self.cpu_time_limit_sec = cpu_time_limit_sec
        self.max_file_size_mb = max_file_size_mb
        self.max_open_files = max_open_files
        self.max_processes = max_processes
        self.memory_limit_mb = memory_limit_mb

        self._km = None
        self._kc = None
        self._sample_id = None
        self._workspace_dir: Optional[str] = None
        self._image_alias: Optional[str] = None
        self._input_files: List[str] = []
        self._bootstrapped = False

    @property
    def workspace_dir(self) -> Optional[str]:
        return self._workspace_dir

    @property
    def input_files(self) -> List[str]:
        return list(self._input_files)

    def begin_sample(self, sample_id: str, image_path: Optional[str] = None) -> Dict[str, Optional[str]]:
        self.end_sample(status="reset")

        sample_slug = str(sample_id).replace(os.sep, "_")
        self._workspace_dir = os.path.abspath(os.path.join(self.workspace_root, f"sample_{sample_slug}"))
        if os.path.exists(self._workspace_dir):
            shutil.rmtree(self._workspace_dir)
        os.makedirs(self._workspace_dir, exist_ok=True)
        self._sample_id = sample_id
        self._image_alias = None
        self._input_files = []
        self._bootstrapped = False

        image_alias = None
        if image_path:
            src = Path(image_path)
            ext = src.suffix.lower() or ".img"
            image_alias = f"input_image{ext}"
            dst = Path(self._workspace_dir) / image_alias
            shutil.copy2(src, dst)
            self._input_files.append(image_alias)
            original_name = src.name
            if original_name != image_alias:
                shutil.copy2(src, Path(self._workspace_dir) / original_name)
                self._input_files.append(original_name)
            (Path(self._workspace_dir) / "ORIGINAL_IMAGE_PATH.txt").write_text(str(src) + "\n", encoding="utf-8")
            self._input_files.append("ORIGINAL_IMAGE_PATH.txt")
            self._image_alias = image_alias

        self._start_kernel()
        self._run_bootstrap()

        return {
            "workspace_dir": self._workspace_dir,
            "image_alias": image_alias,
            "input_files": list(self._input_files),
        }

    def end_sample(self, status: str = "done") -> None:
        if self._kc is not None:
            try:
                self._kc.stop_channels()
            except Exception:
                pass
        if self._km is not None:
            try:
                self._km.shutdown_kernel(now=True)
            except Exception:
                pass

        if (
            self.cleanup_workspace_on_success
            and status == "success"
            and self._workspace_dir
            and os.path.isdir(self._workspace_dir)
        ):
            try:
                shutil.rmtree(self._workspace_dir)
            except Exception:
                pass

        self._km = None
        self._kc = None
        self._sample_id = None
        self._workspace_dir = None
        self._image_alias = None
        self._input_files = []
        self._bootstrapped = False

    def execute_python(self, code: str) -> ExecutionResult:
        if self._workspace_dir is None or self._kc is None:
            raise RuntimeError("begin_sample() must be called before execute_python().")

        code = _strip_code_fences(code)
        if not code.strip():
            return ExecutionResult(
                status="error",
                stdout="",
                stderr="No code provided.",
                rich_output="",
                exec_time_sec=0.0,
                workspace_dir=self._workspace_dir,
                input_files=self.input_files,
            )

        workspace = Path(self._workspace_dir)
        before_files = {
            p.relative_to(workspace).as_posix()
            for p in workspace.rglob("*")
            if p.is_file()
        }

        msg_id = self._kc.execute(code, store_history=True, stop_on_error=True)
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        rich_parts: List[str] = []
        error_name = None
        status = "success"
        start_time = time.perf_counter()
        deadline = start_time + self.execution_timeout_sec

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                status = "timeout"
                stderr_parts.append(
                    f"Execution exceeded the time limit of {self.execution_timeout_sec} seconds."
                )
                try:
                    self._km.interrupt_kernel()
                except Exception:
                    pass
                break

            try:
                msg = self._kc.get_iopub_msg(timeout=remaining)
            except queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                if content.get("name") == "stderr":
                    stderr_parts.append(content.get("text", ""))
                else:
                    stdout_parts.append(content.get("text", ""))
            elif msg_type in ("display_data", "execute_result"):
                data = content.get("data", {})
                if "text/plain" in data:
                    rich_parts.append(str(data["text/plain"]))
                elif data:
                    rich_parts.append(json.dumps(data, ensure_ascii=False)[:4000])
            elif msg_type == "error":
                status = "error"
                error_name = content.get("ename")
                traceback_lines = content.get("traceback") or []
                if traceback_lines:
                    stderr_parts.append("\n".join(traceback_lines))
                else:
                    stderr_parts.append(
                        f"{content.get('ename', 'ExecutionError')}: {content.get('evalue', '')}"
                    )
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        exec_time_sec = time.perf_counter() - start_time

        after_files = {
            p.relative_to(workspace).as_posix()
            for p in workspace.rglob("*")
            if p.is_file()
        }
        reserved = set(self._input_files)
        generated = sorted(x for x in (after_files - before_files) if x not in reserved)

        artifacts: List[ExecutionArtifact] = []
        for rel_path in generated:
            abs_path = workspace / rel_path
            kind = "image" if abs_path.suffix.lower() in _IMAGE_EXTENSIONS else "file"
            try:
                size_bytes = abs_path.stat().st_size
            except OSError:
                size_bytes = -1
            artifacts.append(
                ExecutionArtifact(
                    relative_path=rel_path,
                    absolute_path=str(abs_path),
                    size_bytes=size_bytes,
                    kind=kind,
                )
            )

        return ExecutionResult(
            status=status,
            stdout=_safe_text("".join(stdout_parts), self.max_output_chars),
            stderr=_safe_text("".join(stderr_parts), self.max_output_chars),
            rich_output=_safe_text("\n".join(rich_parts), self.max_output_chars),
            exec_time_sec=exec_time_sec,
            artifacts=artifacts,
            error_name=error_name,
            workspace_dir=self._workspace_dir,
            input_files=self.input_files,
        )

    def _start_kernel(self) -> None:
        try:
            from jupyter_client import KernelManager
        except ImportError as exc:
            raise ImportError(
                "jupyter_client is required. Install with: pip install jupyter_client ipykernel"
            ) from exc

        env = self._scrub_env(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONNOUSERSITE", "1")
        env.setdefault("MPLBACKEND", "Agg")
        env["HOME"] = self._workspace_dir

        km = KernelManager(kernel_name="python3")
        km.start_kernel(cwd=self._workspace_dir, env=env)
        kc = km.blocking_client()
        kc.start_channels()
        kc.wait_for_ready(timeout=self.startup_timeout_sec)
        self._km = km
        self._kc = kc

    def _run_bootstrap(self) -> None:
        if self._bootstrapped:
            return
        limits = {
            "cpu_time_limit_sec": self.cpu_time_limit_sec,
            "max_file_size_mb": self.max_file_size_mb,
            "max_open_files": self.max_open_files,
            "max_processes": self.max_processes,
            "memory_limit_mb": self.memory_limit_mb,
        }
        bootstrap_code = f"""
import os
import sys
import json
from pathlib import Path
os.chdir({self._workspace_dir!r})
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', {str(Path(self._workspace_dir) / '.mplconfig')!r})
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
try:
    from PIL import Image
except Exception:
    Image = None
INPUT_IMAGE = {self._image_alias!r}
WORKSPACE_DIR = {self._workspace_dir!r}
def emit(obj):
    print(json.dumps(obj, ensure_ascii=False))
def inspect_input_image():
    if not INPUT_IMAGE:
        raise RuntimeError('No input image is available for this sample.')
    if Image is None:
        raise RuntimeError('Pillow is not available in the kernel.')
    img = Image.open(INPUT_IMAGE)
    return {{
        'filename': Path(getattr(img, 'filename', INPUT_IMAGE)).name,
        'size': list(img.size),
        'mode': img.mode,
    }}
try:
    import resource
    limits = {limits!r}
    if hasattr(resource, 'RLIMIT_CPU') and limits['cpu_time_limit_sec']:
        resource.setrlimit(resource.RLIMIT_CPU, (limits['cpu_time_limit_sec'], limits['cpu_time_limit_sec'] + 1))
    if hasattr(resource, 'RLIMIT_FSIZE') and limits['max_file_size_mb']:
        value = limits['max_file_size_mb'] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (value, value))
    if hasattr(resource, 'RLIMIT_NOFILE') and limits['max_open_files']:
        value = limits['max_open_files']
        resource.setrlimit(resource.RLIMIT_NOFILE, (value, value))
    if hasattr(resource, 'RLIMIT_NPROC') and limits['max_processes']:
        value = limits['max_processes']
        resource.setrlimit(resource.RLIMIT_NPROC, (value, value))
    if hasattr(resource, 'RLIMIT_AS') and limits['memory_limit_mb']:
        value = limits['memory_limit_mb'] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (value, value))
except Exception as e:
    print(f'BOOTSTRAP_WARNING: {{e}}', file=sys.stderr)
"""
        result = self.execute_python(bootstrap_code)
        # Ignore non-fatal bootstrap warnings, but fail on true bootstrap errors.
        if result.status not in ("success",):
            raise RuntimeError(result.to_tool_message())
        self._bootstrapped = True

    @staticmethod
    def _scrub_env(env: Dict[str, str]) -> Dict[str, str]:
        clean = {}
        for key, value in env.items():
            upper = key.upper()
            if any(pattern in upper for pattern in _ENV_DENYLIST_PATTERNS):
                continue
            clean[key] = value
        return clean
