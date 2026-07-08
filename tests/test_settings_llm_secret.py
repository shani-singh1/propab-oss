from propab.config import Settings


def test_llm_api_secret_openai() -> None:
    s = Settings(llm_provider="openai", openai_api_key="sk-test", google_api_key="g")
    assert s.llm_api_secret == "sk-test"


def test_llm_api_secret_gemini() -> None:
    s = Settings(llm_provider="gemini", openai_api_key="sk-test", google_api_key="AIzaSyX")
    assert s.llm_api_secret == "AIzaSyX"


def test_llm_api_secret_ollama() -> None:
    s = Settings(llm_provider="ollama", openai_api_key="sk-test", google_api_key="g")
    assert s.llm_api_secret == ""


def test_embed_api_secret_openai() -> None:
    s = Settings(embed_provider="openai", openai_api_key="sk-emb", google_api_key="gemb")
    assert s.embed_api_secret == "sk-emb"


def test_embed_api_secret_google() -> None:
    s = Settings(embed_provider="google", openai_api_key="sk-emb", google_api_key="gemb")
    assert s.embed_api_secret == "gemb"


# --- Orchestrator-as-brain redesign: model split + concurrency/retune bounds ---


def test_orchestrator_worker_model_defaults_none() -> None:
    s = Settings()
    assert s.orchestrator_model is None
    assert s.worker_model is None


def test_effective_models_fall_back_to_llm_model_when_unset() -> None:
    # Pass llm_model explicitly so this is independent of any ambient .env.
    s = Settings(llm_model="gpt-4o", orchestrator_model=None, worker_model=None)
    assert s.effective_orchestrator_model == "gpt-4o"
    assert s.effective_worker_model == "gpt-4o"


def test_effective_models_use_override_when_set() -> None:
    s = Settings(
        llm_model="gpt-4o",
        orchestrator_model="o3",
        worker_model="gpt-4o-mini",
    )
    assert s.effective_orchestrator_model == "o3"
    assert s.effective_worker_model == "gpt-4o-mini"


def test_max_parallel_workers_default_matches_campaign_concurrency() -> None:
    # Canonical name mirrors the existing campaign concurrency default so behavior
    # is unchanged (0 = derive from batch_size * inflight_multiplier).
    s = Settings()
    assert s.max_parallel_workers == 0
    assert s.max_parallel_workers == s.campaign_max_concurrent_sub_agents


def test_max_retune_rounds_per_hypothesis_default() -> None:
    s = Settings()
    assert s.max_retune_rounds_per_hypothesis == 3


def test_new_fields_load_from_env(monkeypatch) -> None:
    # Env-var loading follows the class convention: bare (case-insensitive) field names.
    monkeypatch.setenv("ORCHESTRATOR_MODEL", "claude-frontier")
    monkeypatch.setenv("WORKER_MODEL", "claude-cheap")
    monkeypatch.setenv("MAX_PARALLEL_WORKERS", "8")
    monkeypatch.setenv("MAX_RETUNE_ROUNDS_PER_HYPOTHESIS", "5")
    s = Settings()
    assert s.orchestrator_model == "claude-frontier"
    assert s.worker_model == "claude-cheap"
    assert s.max_parallel_workers == 8
    assert s.max_retune_rounds_per_hypothesis == 5
    assert s.effective_orchestrator_model == "claude-frontier"
    assert s.effective_worker_model == "claude-cheap"
