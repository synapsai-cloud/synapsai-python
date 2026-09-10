# Audio

`client.audio` — speech synthesis, transcription, and translation.

## Speech (TTS)

Streaming response (sync):

```python
from synapsai import SynapsAI

client = SynapsAI()

with client.audio.speech.with_streaming_response.create(
    model="your-tts-model",
    input="Hello from SynapsAI",
    response_format="mp3",
) as response:
    response.stream_to_file("out.mp3")
```

| Parameter | Default |
| --- | --- |
| `model` | required |
| `input` | required |
| `response_format` | `"mp3"` |
| `speed` | `1.0` |

## Transcriptions

```python
result = client.audio.transcriptions.create(
    model="your-asr-model",
    file="./meeting.wav",
    language="en",
    response_format="json",
)
print(result.text)
```

Set `stream=True` for streamed transcription chunks when the model supports it.

## Translations

Same shape as transcriptions (speech → English text by default on compatible models):

```python
result = client.audio.translations.create(
    model="your-asr-model",
    file="./clip.mp3",
)
print(result.text)
```
