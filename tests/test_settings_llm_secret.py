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
