"""Optional provider adapters.

Each adapter lazily imports its heavy dependency, so importing this package is
cheap and only *using* an adapter without its extra installed raises a clear,
actionable error. Install what you need:

    pip install "attestq[openai]"   # OpenAIChat
    pip install "attestq[ollama]"   # OllamaEmbedder, OllamaChat
    pip install "attestq[chroma]"   # ChromaStore
    pip install "attestq[rerank]"   # CrossEncoderReranker
"""

from .chroma import ChromaStore
from .ollama import OllamaChat, OllamaEmbedder
from .openai_chat import OpenAIChat
from .openai_embed import OpenAIEmbedder
from .rerank import CrossEncoderReranker

__all__ = [
    "OpenAIChat",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "OllamaChat",
    "ChromaStore",
    "CrossEncoderReranker",
]
