"""Exercise installed-package client dispatch through the real OpenAI SDK."""
import asyncio
import json

import httpx
import openai
import pytest


@pytest.mark.parametrize('is_async', [False, True])
def test_packaged_client_dispatches_sync_and_async_requests_without_top_level_alias(monkeypatch, is_async):
    from axion.training.rl.obj_task_eval.llm_client import base_client, openai_client
    def respond(request):
        assert request.url.path == '/v1/chat/completions'
        data = json.loads(request.content)
        assert data['model'] == 'fixture-model'
        assert data['messages'] == [{'role': 'user', 'content': '2 + 3?'}]
        return httpx.Response(200, json={
            'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': 'fixture-model',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '5'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4},
        })
    transport = httpx.MockTransport(respond)
    if is_async:
        sdk = openai.AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
                                 http_client=httpx.AsyncClient(transport=transport))
        monkeypatch.setattr(openai_client, 'AsyncOpenAI', lambda **kwargs: sdk)
    else:
        sdk = openai.OpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
                           http_client=httpx.Client(transport=transport))
        monkeypatch.setattr(openai_client, 'OpenAI', lambda **kwargs: sdk)
    monkeypatch.delenv('AZURE_OPENAI_ENDPOINT', raising=False)
    client = base_client.LLMClient('openai', is_async=is_async)
    kwargs = dict(model='fixture-model', messages=[{'role': 'user', 'content': '2 + 3?'}], count_token=True)
    if is_async:
        async def request():
            try:
                return await client.async_response(**kwargs)
            finally:
                await sdk.close()
        result = asyncio.run(request())
    else:
        try:
            result = client.response(**kwargs)
        finally:
            sdk.close()
    assert result == ('5', {'input_tokens': 3, 'output_tokens': 1})
