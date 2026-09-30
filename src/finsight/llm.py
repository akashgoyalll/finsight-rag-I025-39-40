"""Chat model factory (provider-agnostic, via LangChain's init_chat_model).

LLM_MODEL uses init_chat_model's "<provider>:<model>" format. Default: Google Gemini
("google_genai:gemini-3.5-flash", free tier in Google AI Studio; key in GOOGLE_API_KEY).

Sampling: Google's Gemini 3 developer guide says temperature/top_p/top_k are "no longer recommended"
for Gemini 3.x models, so no temperature is sent to them. Other models get temperature=0 so
answers are as repeatable as the provider allows.
"""
import os
from langchain.chat_models import init_chat_model

DEFAULT_MODEL = "google_genai:gemini-3.5-flash"


def model_name() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_MODEL


def get_llm(model: str | None = None):
    model = model or model_name()
    kwargs = {} if "gemini-3" in model else {"temperature": 0}
    return init_chat_model(model, **kwargs)
