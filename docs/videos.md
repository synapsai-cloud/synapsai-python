# Videos

`client.videos`

Generate videos, poll for completion, and download assets.

## Create and poll

```python
from synapsai import SynapsAI

client = SynapsAI()

video = client.videos.create_and_poll(
    model="your-video-model",
    prompt="A slow pan across a mountain lake at sunrise",
    seconds=4,
    size="1280x720",
    poll_interval=2.0,
    timeout=600.0,
)
print(video.id, video.status)
```

Or create + poll manually:

```python
job = client.videos.create(
    model="your-video-model",
    prompt="...",
)
job = client.videos.retrieve(job.id)
```

## Download content

```python
with client.videos.download_content(video.id, variant="video") as content:
    content.write_to_file("output.mp4")

# variant: "video" | "thumbnail" | "spritesheet"
```

## Other methods

| Method | Description |
| --- | --- |
| `create(...)` | Start a generation job |
| `retrieve(video_id)` | Job status |
| `delete(video_id)` | Delete a job / asset |
| `download_content(video_id, variant?)` | Streaming binary download |
| `create_and_poll(...)` | Create until terminal status |

Raises `APIError` if `create_and_poll` exceeds `timeout`.
