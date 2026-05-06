"""
config.py
=========
Loads .env and exposes all configuration as typed constants.
Import this at the top of any module that needs settings.

Usage:
    from config import cfg
    llm = ChatGroq(api_key=cfg.groq_api_key, model=cfg.llm_model)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root (works regardless of cwd) ────────────────────
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)


@dataclass(frozen=True)
class _Config:
    # ── Groq ──────────────────────────────────────────────────────────────────
    groq_api_key:    str  = ""
    llm_model:       str  = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens:  int  = 1024

    # ── Ollama fallback ───────────────────────────────────────────────────────
    ollama_base_url: str  = "http://localhost:11434"
    ollama_model:    str  = "qwen2.5:7b"

    # ── Feature flags ─────────────────────────────────────────────────────────
    use_ollama:      bool = False   # set USE_OLLAMA=true in .env to enable

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key) and self.groq_api_key != "your_groq_api_key_here"

    @property
    def llm_provider(self) -> str:
        if self.use_ollama:
            return "ollama"
        if self.has_groq_key:
            return "groq"
        return "none"


def _load() -> _Config:
    return _Config(
        groq_api_key    = os.getenv("GROQ_API_KEY", ""),
        llm_model       = os.getenv("LLM_MODEL",       "llama-3.3-70b-versatile"),
        llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1")),
        llm_max_tokens  = int(os.getenv("LLM_MAX_TOKENS",    "1024")),
        ollama_base_url = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434"),
        ollama_model    = os.getenv("OLLAMA_MODEL",     "qwen2.5:7b"),
        use_ollama      = os.getenv("USE_OLLAMA",       "false").lower() == "true",
    )


# ── Singleton ─────────────────────────────────────────────────────────────────
cfg = _load()