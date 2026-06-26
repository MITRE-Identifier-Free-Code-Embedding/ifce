# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""vLLM-backed embedding models (offline LLM and served subprocess).

vllm and openai are imported lazily inside the methods that use them so this
module imports without those GPU-only / optional dependencies installed.
"""

import os
import signal
import subprocess
import tempfile
import time

import numpy as np
import requests

from ifce.paths import CHECKPOINTS_DIR

from .base import EmbeddingModel

_TASK_DESCRIPTION = (
    "Given a source code function or a decompiled code, find the respective "
    "opposite representation of source code or decompilation that is equivalent."
)


class Qwen3EmbeddingModel(EmbeddingModel):
    def __init__(self, name: str, device: str = "cpu", prompting=False, **kwargs):
        super().__init__(name=name)
        possible_models = ["Qwen3-Embedding-0.6B", "Qwen3-Embedding-8B"]
        if self.name not in possible_models:
            raise ValueError(f"Name not in possible models: {possible_models}")
        self.prompting = prompting
        self.task_description = _TASK_DESCRIPTION
        from vllm import LLM

        self.model = LLM(model=f"Qwen/{self.name}")
        self.truncate_length = self.model.model_config.max_model_len - 1

    def _potentially_add_query(self, str_or_strs: str | list[str]) -> str | list[str]:
        if isinstance(str_or_strs, str):
            str_or_strs = (
                self._get_detailed_instruct(str_or_strs)
                if self.prompting
                else str_or_strs
            )
        elif isinstance(str_or_strs, list):
            str_or_strs = [
                self._get_detailed_instruct(item) if self.prompting else item
                for item in str_or_strs
            ]
        return str_or_strs

    def embed(self, str_to_embed: str) -> list[float]:
        str_to_embed = self._potentially_add_query(str_to_embed)
        outputs = self.model.embed(
            [str_to_embed], truncate_prompt_tokens=self.truncate_length, use_tqdm=False
        )
        return [np.array(o.outputs.embedding) for o in outputs]

    def embed_batch(self, strs_to_embed: list[str]) -> list[list[float]]:
        strs_to_embed = self._potentially_add_query(strs_to_embed)
        outputs = self.model.embed(
            strs_to_embed,
            truncate_prompt_tokens=self.truncate_length,  # Without the -1, vllm stalls indef.
            use_tqdm=False,
        )
        return [np.array(o.outputs.embedding) for o in outputs]

    def _get_detailed_instruct(self, query: str) -> str:
        return f"Instruct: {self.task_description}\nQuery:{query}"


class OnlineQwen3EmbeddingModel(EmbeddingModel):
    """
    Manages a vllm serve subprocess and queries it via the OpenAI-compatible
    /v1/embeddings endpoint. The server is torn down when the instance is
    deleted or used as a context manager.
    """

    DEFAULT_PORT = 8000
    HEALTH_TIMEOUT = 300  # seconds to wait for the server to become ready
    HEALTH_INTERVAL = 2  # seconds between readiness polls

    def __init__(
        self,
        name: str,
        device: str = "cpu",
        prompting: bool = False,
        port: int = DEFAULT_PORT,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        **kwargs,
    ):
        super().__init__(name=name)

        self.prompting = prompting
        self.port = port
        self.task_description = _TASK_DESCRIPTION

        self._process: subprocess.Popen | None = None
        self._start_server(
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            **kwargs,
        )

        from openai import OpenAI

        self.client = OpenAI(
            api_key="EMPTY", base_url=f"http://localhost:{self.port}/v1", timeout=60.0
        )

        # Probe the server so construction fails fast if it is not reachable.
        self.client.models.list()
        # vLLM doesn't expose max_model_len via the OpenAI models API,
        # so fall back to the value passed in or a safe default
        self.truncate_length = (max_model_len - 1) if max_model_len else 32767

    def _start_server(self, device, gpu_memory_utilization, max_model_len, **kwargs):
        cmd = [
            "vllm",
            "serve",
            f"Qwen/{self.name}",
            "--port",
            str(self.port),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
        ]
        if max_model_len is not None:
            cmd += ["--max-model-len", str(max_model_len)]
        for key, value in kwargs.items():
            cmd += ["--" + key.replace("_", "-"), str(value)]

        # Write to a temp log file instead of a pipe — no buffer deadlock possible
        self._log_file_path = os.path.join(
            tempfile.gettempdir(), f"vllm_{self.name}_{self.port}.log"
        )
        self._log_file = open(self._log_file_path, "wb")

        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=self._log_file,
            start_new_session=True,
        )
        self._wait_for_ready()

    def _read_server_output(self, tail_lines: int = 50) -> str:
        """Read the last N lines from the server log file."""
        try:
            with open(self._log_file_path, "rb") as f:
                # Efficiently seek to the end and read the tail
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                # Read up to 32KB from the end
                read_size = min(32768, file_size)
                f.seek(-read_size, os.SEEK_END)
                content = f.read().decode("utf-8", errors="replace")
            lines = content.splitlines()
            return "\n".join(lines[-tail_lines:])
        except Exception as e:
            return f"<could not read log file {self._log_file_path}: {e}>"

    def _wait_for_ready(self) -> None:
        url = f"http://localhost:{self.port}/v1/models"
        deadline = time.monotonic() + self.HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                output = self._read_server_output()
                raise RuntimeError(
                    f"vllm serve exited unexpectedly with code "
                    f"{self._process.returncode}.\n\n"
                    f"--- vllm server output ---\n{output}\n"
                    f"--------------------------"
                )
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(self.HEALTH_INTERVAL)

        # Timed out — capture output before killing the process
        output = self._read_server_output()
        self._shutdown()
        raise TimeoutError(
            f"vllm serve did not become ready within {self.HEALTH_TIMEOUT}s.\n\n"
            f"--- vllm server output ---\n{output}\n"
            f"--------------------------"
        )

    def _shutdown(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                self._process.wait(timeout=15)
            except Exception:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                self._process.wait()
        if hasattr(self, "_log_file") and self._log_file:
            self._log_file.close()
            self._log_file = None
        self._process = None

    def memory_stats(self) -> str:
        """Return the tail of the vLLM server log (used for memory reporting)."""
        return self._read_server_output()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._shutdown()

    def __del__(self):
        self._shutdown()

    def _get_detailed_instruct(self, text: str) -> str:
        return f"Instruct: {self.task_description}\nQuery: {text}"

    def _potentially_add_query(self, str_or_strs: str | list[str]) -> str | list[str]:
        if isinstance(str_or_strs, str):
            return (
                self._get_detailed_instruct(str_or_strs)
                if self.prompting
                else str_or_strs
            )
        return [
            self._get_detailed_instruct(s) if self.prompting else s for s in str_or_strs
        ]

    def embed(self, str_to_embed: str) -> np.ndarray:
        str_to_embed = self._potentially_add_query(str_to_embed)
        response = self.client.embeddings.create(
            model=f"Qwen/{self.name}",
            input=str_to_embed,
            extra_body={"truncate_prompt_tokens": self.truncate_length},
        )
        return np.array(response.data[0].embedding)

    def embed_batch(self, strs_to_embed: list[str]) -> list[np.ndarray]:
        strs_to_embed = self._potentially_add_query(strs_to_embed)
        response = self.client.embeddings.create(
            model=f"Qwen/{self.name}",
            input=strs_to_embed,
            extra_body={"truncate_prompt_tokens": self.truncate_length},
        )
        # The API returns items sorted by index, so ordering is preserved
        return [np.array(item.embedding) for item in response.data]


class OnlineCustomEmbeddingModel(OnlineQwen3EmbeddingModel):
    DEFAULT_PORT = 8000

    def __init__(
        self,
        name: str,
        device: str = "cpu",
        prompting: bool = False,
        port: int = DEFAULT_PORT,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        **kwargs,
    ):
        self.quantization = kwargs["quantization"] if "quantization" in kwargs else None
        super().__init__(name=name)
        self.former_name = "Qwen/Qwen3-Embedding-0.6B"

    def _start_server(self, device, gpu_memory_utilization, max_model_len, **kwargs):
        self.model_path = CHECKPOINTS_DIR / self.name
        cmd = [
            "vllm",
            "serve",
            f"{self.model_path}",
            "--port",
            str(self.port),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
        ]
        if self.quantization:
            cmd.extend(["-q", self.quantization])
            print(f"Quantizing as {self.quantization}")
        if max_model_len is not None:
            cmd += ["--max-model-len", str(max_model_len)]
        for key, value in kwargs.items():
            cmd += ["--" + key.replace("_", "-"), str(value)]

        # Write to a temp log file instead of a pipe — no buffer deadlock possible
        self._log_file_path = os.path.join(
            tempfile.gettempdir(), f"vllm_{self.name}_{self.port}.log"
        )
        self._log_file = open(self._log_file_path, "wb")

        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=self._log_file,
            start_new_session=True,
        )
        self._wait_for_ready()

    def embed(self, str_to_embed: str) -> np.ndarray:
        str_to_embed = self._potentially_add_query(str_to_embed)
        response = self.client.embeddings.create(
            model=f"{self.model_path}",
            input=str_to_embed,
            extra_body={"truncate_prompt_tokens": self.truncate_length},
        )
        return np.array(response.data[0].embedding)

    def embed_batch(self, strs_to_embed: list[str]) -> list[np.ndarray]:
        strs_to_embed = self._potentially_add_query(strs_to_embed)
        response = self.client.embeddings.create(
            model=f"{self.model_path}",
            input=strs_to_embed,
            extra_body={"truncate_prompt_tokens": self.truncate_length},
        )
        # The API returns items sorted by index, so ordering is preserved
        return [np.array(item.embedding) for item in response.data]


class CustomEmbeddingModelVLLM(EmbeddingModel):
    def __init__(self, name: str, device: str = "cpu", prompting=False, **kwargs):
        super().__init__(name=name)

        self.prompting = prompting
        self.task_description = _TASK_DESCRIPTION
        self.model_path = CHECKPOINTS_DIR / self.name
        from vllm import LLM

        self.model = LLM(model=str(self.model_path))
        self.truncate_length = self.model.model_config.max_model_len - 1

    def _potentially_add_query(self, str_or_strs: str | list[str]) -> str | list[str]:
        if isinstance(str_or_strs, str):
            str_or_strs = (
                self._get_detailed_instruct(str_or_strs)
                if self.prompting
                else str_or_strs
            )
        elif isinstance(str_or_strs, list):
            str_or_strs = [
                self._get_detailed_instruct(item) if self.prompting else item
                for item in str_or_strs
            ]
        return str_or_strs

    def embed(self, str_to_embed: str) -> list[float]:
        str_to_embed = self._potentially_add_query(str_to_embed)
        outputs = self.model.embed(
            [str_to_embed], truncate_prompt_tokens=self.truncate_length, use_tqdm=False
        )
        return [np.array(o.outputs.embedding) for o in outputs]

    def embed_batch(self, strs_to_embed: list[str]) -> list[list[float]]:
        strs_to_embed = self._potentially_add_query(strs_to_embed)
        outputs = self.model.embed(
            strs_to_embed,
            truncate_prompt_tokens=self.truncate_length,  # Without the -1, vllm stalls indef.
            use_tqdm=False,
        )
        return [np.array(o.outputs.embedding) for o in outputs]

    def _get_detailed_instruct(self, query: str) -> str:
        return f"Instruct: {self.task_description}\nQuery:{query}"
