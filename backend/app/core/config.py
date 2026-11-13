from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Prefer repo-root .env, then backend/.env (local uvicorn cwd may be either)
_ROOT = Path(__file__).resolve().parents[3]
_BACKEND = Path(__file__).resolve().parents[2]
_ENV_FILES = (
    str(_ROOT / ".env"),
    str(_BACKEND / ".env"),
    ".env",
)
_CONTAINER_MODELS_ROOT = "/models"
_RUNTIME_MODELS_MARKER = "/data/runtime/models"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    postgres_user: str = "rag"
    postgres_password: str = "rag"
    postgres_db: str = "rag"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str | None = None

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_url: str | None = None

    # 旧字段。写稿与任务模型不再读取它，避免 .env 里残留的 LLM_PROVIDER=kimi 盖过千问。
    llm_provider: str = "kimi"  # 仅兼容旧环境；路由以 answer/task_llm_provider 为准
    # LangChain 可选适配：provider=langchain 时用 LANGCHAIN_MODEL（如 anthropic:claude-sonnet-4-5）
    langchain_model: str = ""
    # stub 仅离线/CI：需 ALLOW_LLM_STUB=true，且 APP_ENV 不能是 production
    allow_llm_stub: bool = False
    openai_api_key: str = ""
    gemini_api_key: str = ""
    # Moonshot / Kimi（国内默认 api.moonshot.cn；也可用 KIMI_API_KEY 别名）
    moonshot_api_key: str = ""
    kimi_api_key: str = ""
    kimi_base_url: str = "https://api.moonshot.cn/v1"
    llm_kimi_model: str = "kimi-k2.6"
    llm_gemini_model: str = "gemini-3.5-flash"
    llm_openai_model: str = "gpt-4o-mini"
    # Agent 分类 / query 改写用，不占 Kimi 写稿并发
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_deepseek_model: str = "deepseek-v4-flash"
    # 阿里云百炼千问（OpenAI 兼容模式）。密钥只来自环境，不要写进代码。
    dashscope_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    answer_llm_provider: str = "qwen"
    answer_llm_model: str = "qwen3.8-max"
    task_llm_provider: str = "qwen"
    task_llm_model: str = "qwen3.8-flash"
    llm_max_output_tokens: int = 1024
    llm_temperature: float = 0.2
    # Extra attempts after the first failure. Only 429 and timeouts are retried.
    llm_max_retries: int = 2
    llm_retry_base_seconds: float = 0.4
    llm_retry_max_seconds: float = 8.0

    qa_top_k: int = 8
    qa_max_context_chars: int = 6000
    qa_verify: bool = True
    qa_persist_traces: bool = True

    # Day 12 UI + CORS
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Multi-user login (content isolation by owner_id). Scripts can set false.
    auth_enabled: bool = True
    jwt_secret: str = "dev-only-change-me"
    jwt_expire_hours: int = 168
    # 回归闸门主路径是 users.role=admin。这两项只作回退。
    eval_gate_user_id: str = ""
    eval_gate_username: str = ""

    # Day 13 Agent (factoid budget; unchanged)
    agent_max_steps: int = 8
    agent_planner: str = "auto"  # auto | rules | llm
    agent_min_coverage_rate: float = 0.5
    agent_max_retrieves: int = 2

    # Synthesis (Slice 0+); factoid still uses the Day 13 vars above
    agent_synthesis_max_steps: int = 22
    agent_synthesis_max_subtopics: int = 6
    agent_synthesis_max_retrieves: int = 8
    agent_synthesis_max_catalog_sections: int = 80
    agent_synthesis_min_module_coverage: float = 0.5
    agent_synthesis_section_score_min: float = 0.1
    # 子问题向量挂章节的余弦下限。未用综合任务集校准前不要据此改排序。
    agent_section_hint_min: float = 0.45
    # Cross-Encoder 输出经 Sigmoid，落在 0-1。低于该分视为噪声，直接不进证据
    agent_retrieve_min_score: float = 0.02
    # 达到该分即认为语义相关，可越过子问题的字面覆盖闸门
    agent_hit_semantic_min: float = 0.1
    # 命中块前后各扩几块；0 关闭。超预算时丢邻居、不丢命中
    agent_neighbor_radius: int = 1
    # synthesis 写稿前的全局证据上限（每节点先留最好的一条）
    agent_synthesis_max_evidence: int = 16
    # 已命中但精排分低于该值的节点，覆盖率不足时再检一次
    agent_weak_node_score: float = 0.15
    # Digest lane: DeepSeek reads uploaded documents before compose
    agent_digest_max_docs: int = 8
    agent_digest_max_chars: int = 6000
    # Bounded local reading; no per-batch model calls. Independent writer budget.
    agent_reading_max_batches: int = 3
    agent_reading_batch_chars: int = 6000
    agent_reading_total_chars: int = 18000
    agent_reading_generation_chars: int = 12000
    agent_reading_timeout_seconds: float = 30.0
    agent_longform_max_model_calls: int = 8
    agent_longform_input_tokens: int = 64000
    agent_longform_output_tokens: int = 16000
    agent_longform_final_output_reserve: int = 6000
    agent_longform_analysis_output_tokens: int = 1800
    agent_longform_model_timeout_seconds: float = 240.0
    agent_longform_thinking_tokens: int = 1024
    agent_longform_analysis_thinking_tokens: int = 384

    # Day 14 search cache
    search_cache_enabled: bool = True
    search_trace_candidates: bool = False  # Debug only: IDs/scores, no text or snippets.
    sparse_index_cache_enabled: bool = False
    sparse_index_cache_max_entries: int = 4
    sparse_index_cache_max_bytes: int = 268435456
    sparse_index_cache_max_chunks: int = 250000
    sparse_index_cache_max_source_chars: int = 16000000
    sparse_index_cache_wait_seconds: float = 5.0
    search_cache_ttl: int = 3600

    # P0-2 rate limits (Redis fixed window; fail-open if Redis is down)
    rate_limit_enabled: bool = True
    rate_limit_window_seconds: int = 60
    rate_limit_auth_per_minute: int = 5
    rate_limit_query_per_minute: int = 20
    rate_limit_agent_per_minute: int = 3
    rate_limit_jobs_per_minute: int = 10
    rate_limit_agent_concurrency: int = 1
    rate_limit_agent_stale_seconds: int = 1800
    # Model I/O and liveness are separate. Queue wait is not a running heartbeat.
    llm_connect_timeout_seconds: float = 5
    llm_read_timeout_seconds: float = 120
    llm_write_timeout_seconds: float = 30
    agent_heartbeat_interval_seconds: int = 15
    agent_running_stale_seconds: int = 45
    # Shorter than the heartbeat stale window, so a dead process is visible
    # while its last database heartbeat can still look fresh.
    agent_worker_lease_seconds: int = 30
    # 0 disables the wall-clock cap. In-flight model calls still use the LLM read timeout.
    agent_run_timeout_seconds: float = 0
    agent_tool_timeout_seconds: float = 0
    # Stop after this many tool observations that carry ``error``. 0 disables the streak.
    agent_max_consecutive_tool_errors: int = 3

    # G0–G4 knowledge graph (bypass; does not change /query)
    graph_enabled: bool = True
    graph_extract_max_chunks: int = 10
    graph_extract_max_tokens: int = 4096
    graph_extract_min_chars: int = 800
    # Reasoning models burn the budget on reasoning_content; graph extract needs JSON in content.
    graph_extract_deepseek_model: str = "deepseek-chat"
    graph_extract_window_chars: int = 6000
    graph_expand_enabled: bool = False
    graph_expand_hops: int = 2

    verify_entail_threshold: float = 0.35
    verify_lexical_threshold: float = 0.45
    verify_lexical_gate: float = 0.15

    ingest_max_bytes: int = 20 * 1024 * 1024
    # multilingual-e5-small encodes 512 tokens. This counter is ~1 per CJK
    # character or Latin word, so 220 stays inside the window with the section prefix.
    chunk_min_tokens: int = 140
    chunk_max_tokens: int = 220
    chunk_overlap_tokens: int = 40
    # e5 requires these prefixes, including the trailing space.
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "

    embedding_model_name: str = "intfloat/multilingual-e5-small"
    embedding_model_path: str = ""
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    hf_home: str = ""

    dense_top_k: int = 10
    dense_max_top_k: int = 100
    vector_index_type: str = "hnsw"  # hnsw | ivfflat
    ivfflat_lists: int = 100

    sparse_top_k: int = 10
    sparse_max_top_k: int = 100

    hybrid_top_k: int = 50
    hybrid_dense_candidates: int = 100
    hybrid_sparse_candidates: int = 100
    rrf_k: int = 60

    rerank_model_name: str = "BAAI/bge-reranker-base"
    rerank_model_path: str = ""
    rerank_device: str = "cpu"
    rerank_batch_size: int = 16
    rerank_top_k: int = 8
    rerank_max_top_k: int = 50
    rerank_candidates: int = 50
    # 1 serializes Cross-Encoder calls in this process. 0 keeps them unbounded.
    rerank_max_inflight: int = 1

    @property
    def cors_origin_list(self) -> list[str]:
        raw = (self.cors_origins or "*").strip()
        if raw == "*":
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]

    @property
    def embedding_model_ref(self) -> str:
        if self.embedding_model_path:
            return self._containerize_host_model_path(self.embedding_model_path)
        return self.embedding_model_name

    @property
    def rerank_model_ref(self) -> str:
        if self.rerank_model_path:
            return self._containerize_host_model_path(self.rerank_model_path)
        return self.rerank_model_name

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.database_url:
            return self._normalize_db_url(self.database_url)
        host = self._resolve_service_host(self.postgres_host)
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_connection_url(self) -> str:
        if self.redis_url:
            return self._normalize_redis_url(self.redis_url)
        host = self._resolve_service_host(self.redis_host)
        return f"redis://{host}:{self.redis_port}/0"

    @classmethod
    def _containerize_host_model_path(cls, path: str) -> str:
        """Map host `.env` model paths onto the Compose `/models` mount in Docker."""
        raw = (path or "").strip()
        if not raw or not Path("/.dockerenv").exists():
            return raw
        if raw == _CONTAINER_MODELS_ROOT or raw.startswith(_CONTAINER_MODELS_ROOT + "/"):
            return raw
        normalized = raw.replace("\\", "/")
        idx = normalized.lower().find(_RUNTIME_MODELS_MARKER)
        if idx < 0:
            return raw
        suffix = normalized[idx + len(_RUNTIME_MODELS_MARKER) :]
        if suffix and not suffix.startswith("/"):
            suffix = "/" + suffix
        return _CONTAINER_MODELS_ROOT + suffix

    @classmethod
    def _resolve_service_host(cls, host: str) -> str:
        if host not in {"postgres", "redis"}:
            return host
        if Path("/.dockerenv").exists():
            return host
        return "localhost"

    @classmethod
    def _normalize_redis_url(cls, url: str) -> str:
        if "://redis:" in url and not Path("/.dockerenv").exists():
            return url.replace("://redis:", "://localhost:")
        return url

    @classmethod
    def _normalize_db_url(cls, url: str) -> str:
        if "@postgres:" in url and not Path("/.dockerenv").exists():
            return url.replace("@postgres:", "@localhost:")
        return url


def validate_runtime_timeouts(settings: Settings) -> None:
    """Reject a config that still treats every timeout as the same long budget."""
    connect = float(settings.llm_connect_timeout_seconds)
    read = float(settings.llm_read_timeout_seconds)
    write = float(settings.llm_write_timeout_seconds)
    interval = int(settings.agent_heartbeat_interval_seconds)
    stale = int(settings.agent_running_stale_seconds)
    if connect <= 0 or write <= 0 or read <= connect:
        raise ValueError("llm timeouts must satisfy 0 < connect < read and write > 0")
    if interval <= 0 or interval >= stale:
        raise ValueError("heartbeat interval must be shorter than the stale threshold")
    lease = int(settings.agent_worker_lease_seconds)
    if lease <= 0 or lease >= stale:
        raise ValueError("worker lease must be positive and shorter than the running stale threshold")
    if stale == int(settings.rate_limit_agent_stale_seconds) or read == stale:
        raise ValueError("running stale threshold must differ from the rate-limit window and the read timeout")


@lru_cache
def get_settings() -> Settings:
    return Settings()
