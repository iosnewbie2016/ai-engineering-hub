import ollama

response_generator = ollama.chat(model="mistral", messages=[{"role": "user", "content": "Hello"}], stream=True)
print(type(response_generator))

from typing import AsyncGenerator

async def async_generator_sync_wrap(sync_gen):
    for item in sync_gen:
        yield item
        await asyncio.sleep(0)  # Allows async tasks to run

async def stream_ollama_response(model: str, messages: list) -> AsyncGenerator[str, None]:
    """
    Streams the Ollama chat response as an async generator.
    """
    response_generator = ollama.chat(model=model, messages=messages, stream=True)

    async for response in async_generator_sync_wrap(response_generator):
        yield response['message']['content']
