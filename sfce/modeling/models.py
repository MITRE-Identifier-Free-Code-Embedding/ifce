from typing import List
from transformers import AutoModel, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel
import torch
from vllm import LLM
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import tempfile
from pathlib import Path
import subprocess
import itertools
import time
import signal
import requests
from openai import OpenAI

THIS_FILES_PARENT_DIRECTORY = Path(os.path.dirname(os.path.abspath(__file__))).parent


class EmbeddingModel:
    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.device = device
        self.model = None

    def embed(self, str_to_embed: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, strs_to_embed: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def parameter_count(self) -> int:
        if self.model is None:
            raise ValueError("Model must be instantiated")
        return sum(p.numel() for p in self.model.model.parameters())


class HFEmbeddingModel(EmbeddingModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _infer(self, encoded_input):
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings, p=2, dim=1
            )
        return sentence_embeddings.to("cpu").tolist()

    def embed(self, str_to_embed: str) -> List[float]:
        encoded_input = self.tokenizer(
            [str_to_embed], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        return self._infer(encoded_input)

    def embed_batch(self, strs_to_embed: List[str]) -> List[List[float]]:
        encoded_input = self.tokenizer(
            strs_to_embed, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        return self._infer(encoded_input)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.model.model.parameters())


class HFBGEEmbeddingModel(HFEmbeddingModel):
    def __init__(
        self, model_tokenizer_hf_name: str | None = None, device: str = "cpu", **kwargs
    ):
        super().__init__(**kwargs)
        self.device = device
        self.model_hf_name = f"BAAI/{self.name}"
        self.model = AutoModel.from_pretrained(
            self.model_hf_name, device_map=device
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_tokenizer_hf_name
            if model_tokenizer_hf_name is not None
            else self.model_hf_name
        )


class Qwen3HFEmbeddingModel(HFEmbeddingModel):
    def __init__(
        self, model_tokenizer_hf_name: str | None = None, device: str = "cpu", **kwargs
    ):
        super().__init__(**kwargs)
        self.device = device
        self.model_hf_name = f"Qwen/{self.name}"
        self.model = AutoModel.from_pretrained(
            self.model_hf_name, device_map=device
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_tokenizer_hf_name
            if model_tokenizer_hf_name is not None
            else self.model_hf_name
        )


class CustomHFEmbeddingModel(HFEmbeddingModel):
    def __init__(self, name: str, device: str = "cpu"):
        super().__init__(name=name, device=device)
        models_path = Path(THIS_FILES_PARENT_DIRECTORY) / "models" / name
        self.model = AutoModel.from_pretrained(models_path, device_map=device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(models_path)


class HFBGE3EmbeddingModel(EmbeddingModel):
    def __init__(self, name: str = "bge-m3", device: str = "cpu", **kwargs):
        super().__init__(name, device, **kwargs)
        self.model = BGEM3FlagModel(f"BAAI/{name}", use_fp16=True)

    def embed(self, str_to_embed: str) -> List[float]:
        result = self.model.encode(
            [str_to_embed],
            batch_size=1,
            max_length=8192,
        )["dense_vecs"]
        return result.tolist()

    def embed_batch(self, strs_to_embed: List[str]) -> List[List[float]]:
        result = self.model.encode(
            strs_to_embed,
            batch_size=len(strs_to_embed),
            max_length=8192,
        )["dense_vecs"]
        return result.tolist()


class Qwen3EmbeddingModel(EmbeddingModel):
    def __init__(self, name: str, device: str = "cpu", prompting=False, **kwargs):
        super().__init__(name=name)
        possible_models = ["Qwen3-Embedding-0.6B", "Qwen3-Embedding-8B"]
        if self.name not in possible_models:
            raise ValueError(f"Name not in possible models: {possible_models}")
        self.prompting = prompting
        self.task_description = "Given a source code function or a decompiled code, find the respective opposite representation of source code or decompilation that is equivalent."
        self.model = LLM(model=f"Qwen/{self.name}")
        self.truncate_length = self.model.model_config.max_model_len - 1

    def _potentially_add_query(self, str_or_strs: str | List[str]) -> str | List[str]:
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

    def embed(self, str_to_embed: str) -> List[float]:
        str_to_embed = self._potentially_add_query(str_to_embed)
        outputs = self.model.embed(
            [str_to_embed], truncate_prompt_tokens=self.truncate_length, use_tqdm=False
        )
        return [np.array(o.outputs.embedding) for o in outputs]

    def embed_batch(self, strs_to_embed: List[str]) -> List[List[float]]:
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
        self.task_description = (
            "Given a source code function or a decompiled code, find the respective "
            "opposite representation of source code or decompilation that is equivalent."
        )

        self._process: subprocess.Popen | None = None
        self._start_server(
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            **kwargs,
        )

        self.client = OpenAI(
            api_key="EMPTY", base_url=f"http://localhost:{self.port}/v1", timeout=60.0
        )

        models_response = self.client.models.list()
        matched = next(
            (m for m in models_response.data if m.id == f"Qwen/{self.name}"), None
        )
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
            except:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                self._process.wait()
        if hasattr(self, "_log_file") and self._log_file:
            self._log_file.close()
            self._log_file = None
        self._process = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._shutdown()

    def __del__(self):
        self._shutdown()

    def _get_detailed_instruct(self, text: str) -> str:
        return f"Instruct: {self.task_description}\nQuery: {text}"

    def _potentially_add_query(self, str_or_strs: str | List[str]) -> str | List[str]:
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

    def embed_batch(self, strs_to_embed: List[str]) -> List[np.ndarray]:

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
        self.model_path = Path(THIS_FILES_PARENT_DIRECTORY) / "models" / self.name
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

    def embed_batch(self, strs_to_embed: List[str]) -> List[np.ndarray]:

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
        self.task_description = "Given a source code function or a decompiled code, find the respective opposite representation of source code or decompilation that is equivalent."
        self.model_path = Path(THIS_FILES_PARENT_DIRECTORY) / "models" / self.name
        self.model = LLM(model=str(self.model_path))
        self.truncate_length = self.model.model_config.max_model_len - 1

    def _potentially_add_query(self, str_or_strs: str | List[str]) -> str | List[str]:
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

    def embed(self, str_to_embed: str) -> List[float]:
        str_to_embed = self._potentially_add_query(str_to_embed)
        outputs = self.model.embed(
            [str_to_embed], truncate_prompt_tokens=self.truncate_length, use_tqdm=False
        )
        return [np.array(o.outputs.embedding) for o in outputs]

    def embed_batch(self, strs_to_embed: List[str]) -> List[List[float]]:
        strs_to_embed = self._potentially_add_query(strs_to_embed)
        outputs = self.model.embed(
            strs_to_embed,
            truncate_prompt_tokens=self.truncate_length,  # Without the -1, vllm stalls indef.
            use_tqdm=False,
        )
        return [np.array(o.outputs.embedding) for o in outputs]

    def _get_detailed_instruct(self, query: str) -> str:
        return f"Instruct: {self.task_description}\nQuery:{query}"


class GemmaEmbeddingModel(EmbeddingModel):
    def __init__(self, name: str, device: str = "cpu"):
        super().__init__(name, device=device)
        self.model = SentenceTransformer(
            model_name_or_path=f"google/{self.name}", device=self.device
        )

    def embed(self, str_to_embed: str):
        embedding = self.model.encode_document(
            [str_to_embed]
        )  # There are other functions, like encode_query. What to do about them?
        return embedding

    def embed_batch(self, strs_to_embed: List[str]):
        embeddings = self.model.encode_document(
            strs_to_embed
        )  # There are other functions, like encode_query. What to do about them?
        return embeddings


class BaseSentenceTransformerModel(EmbeddingModel):
    def __init__(self, name: str, base_name: str, device: str = "cpu", **kwargs):
        super().__init__(name=name, device=device)
        self.base_name = base_name
        self.model = SentenceTransformer(
            model_name_or_path=f"{base_name}/{self.name}", device=self.device, **kwargs
        )

    def embed(self, str_to_embed: str) -> np.ndarray:
        embedding = self.model.encode([str_to_embed])
        return embedding

    def embed_batch(self, strs_to_embed: List[str]) -> np.ndarray:
        embeddings = self.model.encode(strs_to_embed)
        return embeddings

    def embed_batch_as_smaller_pieces(self, strs_to_embed: List[str]) -> np.ndarray:
        mini_bs = 4
        if (
            len(strs_to_embed) > mini_bs
        ):  # it's so large, we manage batching smaller than others
            all_embeddings = []
            for batch in itertools.batched(strs_to_embed, mini_bs):
                all_embeddings.extend(self.model.encode(list(batch)))
            return np.array(all_embeddings)
        else:
            all_embeddings = self.model.encode(strs_to_embed)
            return all_embeddings


class SnowflakeEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(self, name: str, base_name: str = "Snowflake", device: str = "cpu"):
        super().__init__(name=name, base_name=base_name, device=device)


class SalesforceEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(self, name: str, base_name: str = "Salesforce", device: str = "cpu"):
        super().__init__(name=name, base_name=base_name, device=device)


class NomicAIEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(
        self,
        name: str,
        base_name: str = "nomic-ai",
        device: str = "cpu",
        model_kwargs={"dtype": "bfloat16"},
    ):
        super().__init__(
            name=name, base_name=base_name, device=device, model_kwargs=model_kwargs
        )

    def embed_batch(self, strs_to_embed: List[str]) -> np.ndarray:
        if isinstance(strs_to_embed, str):
            raise TypeError("embed_batch must take a list of strs")
        return self.embed_batch_as_smaller_pieces(strs_to_embed=strs_to_embed)


class JinaAIEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(self, name: str, base_name: str = "jinaai", device: str = "cpu"):
        super().__init__(name=name, base_name=base_name, device=device)


class CodeSageEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(self, name: str, base_name: str = "codesage", device: str = "cpu"):
        super().__init__(name=name, base_name=base_name, device=device)


class BinSeekEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(self, name: str, base_name: str = "XingTuLab", device: str = "cpu"):
        super().__init__(name=name, base_name=base_name, device=device)


MODEL_CLASS_MAP = {
    "bge-small-en-v1.5": HFBGEEmbeddingModel,
    "bge-base-en-v1.5": HFBGEEmbeddingModel,
    "bge-large-en-v1.5": HFBGEEmbeddingModel,
    "bge-m3": HFBGE3EmbeddingModel,
    "Qwen3-Embedding-0.6B": OnlineQwen3EmbeddingModel,
    "Qwen3-Embedding-8B": OnlineQwen3EmbeddingModel,
    "embeddinggemma-300m": GemmaEmbeddingModel,
    "snowflake-arctic-embed-l-v2.0": SnowflakeEmbeddingModel,
    "SFR-Embedding-Mistral": SalesforceEmbeddingModel,
    "nomic-embed-code": NomicAIEmbeddingModel,
    "jina-embeddings-v2-base-code": JinaAIEmbeddingModel,
    "codesage-large-v2": CodeSageEmbeddingModel,
    "BinSeek-Embedding": BinSeekEmbeddingModel,
}


MODEL_SIZE_MAP = {
    "bge-small-en-v1.5": 0.033,
    "bge-base-en-v1.5": 0.1,
    "bge-large-en-v1.5": 0.3,
    "bge-m3": 0.56,
    "Qwen3-Embedding-0.6B": 0.6,
    "Qwen3-Embedding-8B": 8,
    "embeddinggemma-300m": 0.3,
    "snowflake-arctic-embed-l-v2.0": 0.3,
    "SFR-Embedding-Mistral": 7,
    "nomic-embed-code": 7,
    "jina-embeddings-v2-base-code": 0.16,
    "codesage-large-v2": 1.3,
    "BinSeek-Embedding": 0.3,
}

for entry in Path(Path(THIS_FILES_PARENT_DIRECTORY) / "models").iterdir():
    if entry.is_dir():
        MODEL_CLASS_MAP.update({entry.name: OnlineCustomEmbeddingModel})
        MODEL_SIZE_MAP.update(
            {entry.name: 0.6}
        )  # Hardcoded for now, needs adjusting if new custom model is introduced.

assert set(MODEL_CLASS_MAP.keys()) == set(MODEL_SIZE_MAP.keys())
