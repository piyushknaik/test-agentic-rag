from dataclasses import dataclass
from graphitic.client import EmbedderConfig

@dataclass
class OllamaEmbedderConfig(EmbedderConfig):
    model: str
    base_url: str = "http://localhost:11434"
    embedding_dim: int = 768