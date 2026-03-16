"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # LLM Configuration
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    # Embedding and Reranking Models
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="CROSS_ENCODER_MODEL",
    )

    # Index Paths
    faiss_index_path: str = Field(
        default="indexes/faiss.index", alias="FAISS_INDEX_PATH"
    )
    bm25_index_path: str = Field(
        default="indexes/bm25.pkl", alias="BM25_INDEX_PATH"
    )

    # Chunking
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")

    # Retrieval Configuration
    top_k_retrieval: int = Field(default=100, alias="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=20, alias="TOP_K_RERANK")
    top_k_context: int = Field(default=5, alias="TOP_K_CONTEXT")
    max_context_tokens: int = Field(default=3000, alias="MAX_CONTEXT_TOKENS")

    # Hybrid Search Weights
    hybrid_bm25_weight: float = Field(default=0.5, alias="HYBRID_BM25_WEIGHT")
    hybrid_dense_weight: float = Field(
        default=0.5, alias="HYBRID_DENSE_WEIGHT"
    )

    # Infrastructure
    redis_url: str = Field(
        default="redis://localhost:6379", alias="REDIS_URL"
    )
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Evaluation
    llm_judge_model: str = Field(
        default="gpt-4o-mini", alias="LLM_JUDGE_MODEL"
    )
    hallucination_threshold: float = Field(
        default=0.5, alias="HALLUCINATION_THRESHOLD"
    )

    model_config = {"env_file": ".env", "populate_by_name": True}


settings = Settings()
