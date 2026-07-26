"""Provider-neutral model layer for cognitive agents."""

from joker.models.exceptions import (
    ModelBudgetExceeded,
    ModelConfigurationError,
    ModelError,
    ModelProviderUnavailable,
    ModelRefusal,
    ModelResponseEmpty,
    ModelTimeout,
    StructuredOutputFailure,
)
from joker.models.fake_provider import FakeCallRecord, FakeModelProvider
from joker.models.ollama_provider import OllamaModelProvider
from joker.models.openai_provider import OpenAIModelProvider
from joker.models.provider import ModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import (
    ModelProfileConfig,
    ModelRequest,
    ModelResult,
    ModelsConfig,
    OllamaProviderConfig,
    OpenAIProviderConfig,
    ProviderHealth,
    default_model_profiles,
)
from joker.models.telemetry import (
    build_model_call_completed,
    build_model_call_failed,
    build_model_call_started,
    build_routing_decision,
)

__all__ = [
    "FakeCallRecord",
    "FakeModelProvider",
    "ModelBudgetExceeded",
    "ModelConfigurationError",
    "ModelError",
    "ModelProfileConfig",
    "ModelProvider",
    "ModelProviderUnavailable",
    "ModelRefusal",
    "ModelRegistry",
    "ModelRequest",
    "ModelResponseEmpty",
    "ModelResult",
    "ModelRouter",
    "ModelTimeout",
    "ModelsConfig",
    "OllamaModelProvider",
    "OllamaProviderConfig",
    "OpenAIProviderConfig",
    "OpenAIModelProvider",
    "ProviderHealth",
    "StructuredOutputFailure",
    "build_model_call_completed",
    "build_model_call_failed",
    "build_model_call_started",
    "build_routing_decision",
    "default_model_profiles",
]
