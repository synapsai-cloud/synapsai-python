# Classifications

`client.classifications` and `client.classifications.zero_shot`

## Supervised / pipeline classifications

```python
from synapsai import SynapsAI

client = SynapsAI()

# Text
text = client.classifications.text(
    model="your-classifier",
    inputs="I love this product!",
    top_k=3,
)

# Image
image = client.classifications.image(
    model="your-image-classifier",
    inputs="./photo.jpg",
)

# Audio / video / token
audio = client.classifications.audio(model="...", inputs="./clip.wav")
video = client.classifications.video(model="...", inputs="./clip.mp4", num_frames=8)
tokens = client.classifications.token(model="...", inputs="Paris is the capital of France")
```

| Method | Typical inputs |
| --- | --- |
| `text` | string or list |
| `image` | path / bytes / PIL |
| `audio` | audio path / bytes |
| `video` | video path / bytes |
| `token` | string (NER-style) |

## Zero-shot

```python
zs = client.classifications.zero_shot

text = zs.text(
    model="your-zs-model",
    sequences="A new smartphone was announced today",
    candidate_labels=["technology", "sports", "politics"],
)

image = zs.image(
    model="your-zs-vision-model",
    image="./photo.jpg",
    candidate_labels=["cat", "dog", "bird"],
)

audio = zs.audio(
    model="your-zs-audio-model",
    audios="./clip.wav",
    candidate_labels=["speech", "music"],
)
```
