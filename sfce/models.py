from typing import List
from transformers import AutoModel, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel
import torch

class EmbeddingModel:
    def __init__(self, name:str):
        self.name = name

    def embed(self, str_to_embed:str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, strs_to_embed:List[str]) -> List[List[float]]:
        raise NotImplementedError
    
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


class HFBGE3EmbeddingModel(EmbeddingModel):
    def __init__(self, device:str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

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

MODEL_CLASS_MAP = {"bge-small-en-v1.5": HFBGEEmbeddingModel,
                    "bge-base-en-v1.5": HFBGEEmbeddingModel,
                    "bge-large-en-v1.5": HFBGEEmbeddingModel,
                    "bge-m3": HFBGE3EmbeddingModel}