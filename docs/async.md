# Async client

`AsyncSynapsAI` mirrors every sync resource with `async` methods.

```python
import asyncio
from synapsai import AsyncSynapsAI

async def main():
    async with AsyncSynapsAI() as client:
        models = await client.models.list()
        print([m.id for m in models.data])

        stream = await client.chat.completions.create(
            model="your-model-id",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                print(delta.content, end="", flush=True)

asyncio.run(main())
```

## Notes

- Use `async with AsyncSynapsAI() as client:` (or call `await client.close()`).
- Streaming methods return async iterators.
- Agent runs: `async for event in client.agents.run(...)`.
- Model artifact / vector store uploads use the same method names with `await`.
