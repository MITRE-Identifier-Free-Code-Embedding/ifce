from typing import List
from transformers import AutoModel, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel
import torch
from vllm import LLM
import numpy as np
from sentence_transformers import SentenceTransformer

# TODO change type annotations to reflect model return types

class EmbeddingModel:
    def __init__(self, name:str, device: str = "cpu"):
        self.name = name
        self.device = device
        self.model = None

    def embed(self, str_to_embed:str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, strs_to_embed:List[str]) -> List[List[float]]:
        raise NotImplementedError
    
    def parameter_count(self) -> int:
        if self.model == None:
            raise ValueError("Model must be instantiated")
        return sum(p.numel() for p in self.model.model.parameters())
    
class HFBGEEmbeddingModel(EmbeddingModel):
    def __init__(self,
                 model_tokenizer_hf_name: str | None = None, 
                 device: str = "cpu",
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.model_hf_name = f"BAAI/{self.name}"
        self.model = AutoModel.from_pretrained(self.model_hf_name,
                                               device_map=device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_hf_name if model_tokenizer_hf_name is not None else self.model_hf_name)

    def _infer(self,encoded_input):
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.to('cpu').tolist()

    def embed(self, str_to_embed: str) -> List[float]:
        encoded_input = self.tokenizer([str_to_embed], 
                                       padding=True, 
                                       truncation=True, 
                                       return_tensors='pt').to(self.device)
        return self._infer(encoded_input)

    def embed_batch(self, strs_to_embed:List[str]) -> List[List[float]]:
        encoded_input = self.tokenizer(strs_to_embed, 
                                       padding=True, 
                                       truncation=True, 
                                       return_tensors='pt').to(self.device)
        return self._infer(encoded_input)
    
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.model.model.parameters())


class HFBGE3EmbeddingModel(EmbeddingModel):
    def __init__(self, 
                 name:str = 'bge-m3', 
                 device:str = "cpu", **kwargs):
        super().__init__(name, device, **kwargs)
        self.model = BGEM3FlagModel(f"BAAI/{name}", use_fp16=True)

    def embed(self, str_to_embed: str) -> List[float]:
        result = self.model.encode([str_to_embed], 
                                       batch_size=1, 
                                       max_length=8192,)["dense_vecs"]
        return result.tolist()
    
    def embed_batch(self, strs_to_embed:List[str]) -> List[List[float]]:
        result = self.model.encode(strs_to_embed, 
                                       batch_size=len(strs_to_embed), 
                                       max_length=8192,)["dense_vecs"]
        return result.tolist()

# TODO add Qwen3 models w / wo "prompting"?
class Qwen3EmbeddingModel(EmbeddingModel):
    def __init__(self, name:str, device:str = "cpu", prompting=False, **kwargs):
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
            str_or_strs = self._get_detailed_instruct(str_or_strs) if self.prompting else str_or_strs
        elif isinstance(str_or_strs, list):
            str_or_strs = [self._get_detailed_instruct(item) if self.prompting else item for item in str_or_strs]
        return str_or_strs

    def embed(self, str_to_embed: str) -> List[float]:
        str_to_embed = self._potentially_add_query(str_to_embed)
        outputs = self.model.embed([str_to_embed], 
                                   truncate_prompt_tokens=self.truncate_length,
                                   use_tqdm=False)
        return [np.array(o.outputs.embedding) for o in outputs]

    def embed_batch(self, strs_to_embed:List[str]) -> List[List[float]]:
        strs_to_embed = self._potentially_add_query(strs_to_embed)
        outputs = self.model.embed(strs_to_embed, 
                                   truncate_prompt_tokens=self.truncate_length, # Without the -1, vllm stalls indef.
                                   use_tqdm=False)
        return [np.array(o.outputs.embedding) for o in outputs]

    def _get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.task_description}\nQuery:{query}'

class GemmaEmbeddingModel(EmbeddingModel):
    def __init__(self, name: str, device:str = "cpu"):
        super().__init__(name, device=device)
        self.model = SentenceTransformer(model_name_or_path=f"google/{self.name}", device=self.device)

    def embed(self, str_to_embed:str ):
        embedding = self.model.encode_document([str_to_embed]) # There are other functions, like encode_query. What to do about them?
        return embedding
    

    def embed_batch(self, strs_to_embed:List[str]):
        embeddings = self.model.encode_document(strs_to_embed) # There are other functions, like encode_query. What to do about them?
        return embeddings
    

class BaseSentenceTransformerModel(EmbeddingModel):
    def __init__(self, name: str, base_name: str, device:str = "cpu"):
        super().__init__(name=name, device=device)
        self.model = SentenceTransformer(model_name_or_path=f"{base_name}/{self.name}", device=self.device)

    def embed(self, str_to_embed:str ) -> np.ndarray:
        embedding = self.model.encode([str_to_embed])
        return embedding
    
    def embed_batch(self, strs_to_embed:List[str]) -> np.ndarray:
        embeddings = self.model.encode(strs_to_embed)
        return embeddings
    

class SnowflakeEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(self, name: str, base_name: str = "Snowflake", device:str = "cpu"):
        super().__init__(name=name, base_name=base_name, device=device)

class SalesforceEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(self, name: str, base_name: str = "Salesforce", device:str = "cpu"):
        super().__init__(name=name, base_name=base_name, device=device)


MODEL_CLASS_MAP = {"bge-small-en-v1.5": HFBGEEmbeddingModel,
                    "bge-base-en-v1.5": HFBGEEmbeddingModel,
                    "bge-large-en-v1.5": HFBGEEmbeddingModel,
                    "bge-m3": HFBGE3EmbeddingModel,
                    "Qwen3-Embedding-0.6B": Qwen3EmbeddingModel,
                    "Qwen3-Embedding-8": Qwen3EmbeddingModel,
                    "embeddinggemma-300m": GemmaEmbeddingModel,
                    "snowflake-arctic-embed-l-v2.0": SnowflakeEmbeddingModel,
                    "SFR-Embedding-Mistral": SalesforceEmbeddingModel}