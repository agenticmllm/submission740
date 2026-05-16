from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool

from local_kernel_executor import LocalKernelExecutor

from qwen_agent.llm.schema import ContentItem



@register_tool("local_code_interpreter")
class SampleScopedLocalKernelCodeInterpreter(BaseTool):

    description = """
    A Python code execution tool running in a persistent local kernel.
    Use this to perform calculations, analyze data, and process/visualize images using libraries like PIL or matplotlib.
    The current image is already available in the working directory under its original filename.

    [CRITICAL RULE FOR IMAGES]: DO NOT use `plt.show()` or `img.show()`. You MUST save the image to the current directory (e.g., `plt.savefig('output.png')` or `img.save('output.png')`). The system will automatically present the saved image to your vision module in the next turn.
    [CRITICAL RULE FOR LOOPING]: Do NOT repeat the exact same code. If your code produces the same image or result as the previous step, STOP using this tool and proceed to your final synthesized answer immediately.
    [NETWORK RULE]: Internet access is strictly forbidden. Do not use requests, BeautifulSoup, or make any external network connections.
    """

    parameters = [{
        "name": "code",
        "type": "string",
        "description": "The Python code to execute.",
        "required": True,
    }]

    workspace_root = "./workspace"
    startup_timeout_seconds = 30
    timeout_seconds = 45
    keep_workspace = True
    max_output_chars = 12000
    cpu_time_limit_sec: Optional[int] = 40
    max_file_size_mb: Optional[int] = 100
    max_open_files: Optional[int] = 128
    max_processes: Optional[int] = None
    memory_limit_mb: Optional[int] = None

    _lock = threading.RLock()
    _executor: Optional[LocalKernelExecutor] = None
    _current_sample_id: Optional[Union[str, int]] = None
    _current_image_path: Optional[Path] = None

    @classmethod
    def configure(
        cls,
        workspace_root: Union[str, Path],
        timeout_seconds: int = 45,
        startup_timeout_seconds: int = 30,
        keep_workspace: bool = True,
        max_output_chars: int = 12000,
        cpu_time_limit_sec: Optional[int] = 40,
        max_file_size_mb: Optional[int] = 100,
        max_open_files: Optional[int] = 128,
        max_processes: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
    ) -> None:
        cls.workspace_root = str(Path(workspace_root).expanduser().resolve())
        cls.timeout_seconds = int(timeout_seconds)
        cls.startup_timeout_seconds = int(startup_timeout_seconds)
        cls.keep_workspace = bool(keep_workspace)
        cls.max_output_chars = int(max_output_chars)
        cls.cpu_time_limit_sec = cpu_time_limit_sec
        cls.max_file_size_mb = max_file_size_mb
        cls.max_open_files = max_open_files
        cls.max_processes = max_processes
        cls.memory_limit_mb = memory_limit_mb

    @classmethod
    def bind_sample(cls, sample_id: Union[str, int], image_path: Union[str, Path]) -> str:
        with cls._lock:
            cls.release_sample(status="reset")
            cls._executor = LocalKernelExecutor(
                workspace_root=cls.workspace_root,
                startup_timeout_sec=cls.startup_timeout_seconds,
                execution_timeout_sec=cls.timeout_seconds,
                max_output_chars=cls.max_output_chars,
                cleanup_workspace_on_success=not cls.keep_workspace,
                cpu_time_limit_sec=cls.cpu_time_limit_sec,
                max_file_size_mb=cls.max_file_size_mb,
                max_open_files=cls.max_open_files,
                max_processes=cls.max_processes,
                memory_limit_mb=cls.memory_limit_mb,
            )
            cls._current_sample_id = sample_id
            cls._current_image_path = Path(image_path).expanduser().resolve()
            sample_ctx = cls._executor.begin_sample(sample_id=str(sample_id), image_path=str(cls._current_image_path))
            return sample_ctx["workspace_dir"] or cls.workspace_root

    @classmethod
    def release_sample(cls, status: str = "done") -> None:
        with cls._lock:
            if cls._executor is not None:
                try:
                    cls._executor.end_sample(status=status)
                finally:
                    cls._executor = None
            cls._current_sample_id = None
            cls._current_image_path = None

    @classmethod
    def current_session_dir(cls) -> Optional[str]:
        if cls._executor is None:
            return None
        return cls._executor.workspace_dir

    @staticmethod
    def _extract_code(params: Union[str, Dict]) -> str:
        if isinstance(params, str):
            stripped = params.strip()
            if stripped.startswith("{"):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    code = parsed.get("code", "")
                else:
                    code = params
            else:
                code = params
        else:
            code = params.get("code", "")
        return str(code or "")

    def call(self, params: Union[str, Dict], **kwargs) -> Union[str, list]:
        cls = self.__class__
        with cls._lock:
            if cls._executor is None:
                return (
                    "STATUS: error\n\n"
                    "No sample is currently bound to local_code_interpreter. "
                    "Bind the sample before agent.run()."
                )
            code = self._extract_code(params)
            if not code.strip():
                return "STATUS: error\n\nExecution Error: missing 'code' argument."
            
            result = cls._executor.execute_python(code)
    
            image_outputs = []
            for artifact in result.artifacts:
                if artifact.kind == "image":
                    image_outputs.append(ContentItem(image=artifact.absolute_path))
            
            if image_outputs:
                text_summary = result.to_tool_message(max_text_chars=cls.max_output_chars)
                return [ContentItem(text=text_summary)] + image_outputs
            
            return result.to_tool_message(max_text_chars=cls.max_output_chars)