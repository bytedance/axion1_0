# Modified by the Axion authors: resolve providers relative to this package.
class LLMClient():
    def __init__(self, provider, is_async=False):
        self.provider = provider
        self.is_async = is_async
        if provider == 'azure_openai':
            if is_async:
                from .azure_openai_client import AsyncAzureOpenAIClient
                self.client = AsyncAzureOpenAIClient()
            else:
                from .azure_openai_client import AzureOpenAIClient
                self.client = AzureOpenAIClient()
        elif provider == 'openai':
            if is_async:
                from .openai_client import AsyncOpenAIClient
                self.client = AsyncOpenAIClient()
            else:
                from .openai_client import OpenAIClient
                self.client = OpenAIClient()
        elif provider == 'local_openai':
            if is_async:
                from .local_openai_client import AsyncLocalOpenAIClient
                self.client = AsyncLocalOpenAIClient()
            else:
                from .local_openai_client import LocalOpenAIClient
                self.client = LocalOpenAIClient()

        elif provider == 'bedrock_anthropic':
            if is_async:
                from .bedrock_anthropic_client import AsyncBedrockAntrhopicClient
                self.client = AsyncBedrockAntrhopicClient()
            else:
                from .bedrock_anthropic_client import BedrockAntrhopicClient
                self.client = BedrockAntrhopicClient()
        else:
            raise ValueError(f'Provider {provider} not supported')
    
    def response(self, **kwargs):
        # Reject asynchronous providers on the synchronous API.
        if self.is_async:
            raise ValueError(f'Provider {self.provider} is async and does not support synchronous response')
        return self.client.response(**kwargs)

    async def async_response(self, **kwargs):
        # Reject synchronous providers on the asynchronous API.
        if not self.is_async:
            raise ValueError(f'Provider {self.provider} is not async and does not support asynchronous response')
        return await self.client.response(**kwargs)
