# Images

`client.images`

Image generation, editing, and vision-style pipelines. File inputs accept paths, bytes, or PIL images (processed by the SDK).

## Generate

```python
response = client.images.generate(
    model="your-image-model",
    prompt="A red lighthouse at dusk",
    size="1024x1024",
    n=1,
)
print(response.data[0].url or response.data[0].b64_json)
```

## Edit

```python
response = client.images.edit(
    model="your-image-model",
    image="./photo.png",
    prompt="Replace the sky with aurora",
    mask="./mask.png",  # optional
)
```

## Vision / analysis helpers

| Method | Endpoint | Typical use |
| --- | --- | --- |
| `to_text` | `images/to-text` | Caption / describe |
| `feature_extraction` | `images/feature-extraction` | Image embeddings |
| `segmentation` | `images/segmentation` | Panoptic / instance masks |
| `depth_estimation` | `images/depth-estimation` | Depth maps |
| `object_detection` | `images/object-detection` | Boxes + labels |
| `mask_generation` | `images/mask-generation` | SAM-style masks |

```python
caption = client.images.to_text(
    model="your-vlm",
    inputs="./photo.png",
    prompt="Describe this image.",
    max_new_tokens=128,
)
```
