__all__ = [
    "LLMClient",
    "OpenAIClient",
    "AsyncOpenAIClient",
    "AzureOpenAIClient",
    "AsyncAzureOpenAIClient",
    "calculate_api_cost",
]


def __getattr__(name):
    if name == "LLMClient":
        from .base_client import LLMClient

        return LLMClient
    if name in {"OpenAIClient", "AsyncOpenAIClient"}:
        from .openai_client import AsyncOpenAIClient, OpenAIClient

        return {"OpenAIClient": OpenAIClient, "AsyncOpenAIClient": AsyncOpenAIClient}[name]
    if name in {"AzureOpenAIClient", "AsyncAzureOpenAIClient"}:
        from .azure_openai_client import AsyncAzureOpenAIClient, AzureOpenAIClient

        return {
            "AzureOpenAIClient": AzureOpenAIClient,
            "AsyncAzureOpenAIClient": AsyncAzureOpenAIClient,
        }[name]
    if name == "calculate_api_cost":
        from .api_cost import calculate_api_cost

        return calculate_api_cost
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
