# Question answering

`client.question_answering`

## Text QA

```python
from synapsai import SynapsAI

client = SynapsAI()

answer = client.question_answering.text(
    model="your-qa-model",
    question="Where is the company based?",
    context="SynapsAI is based in Canada.",
)
print(answer)
```

## Document / visual / table

```python
# Document (image + OCR / layout QA)
doc = client.question_answering.document(
    model="your-doc-qa-model",
    image="./invoice.png",
    question="What is the total?",
)

# Visual QA
vqa = client.question_answering.visual(
    model="your-vqa-model",
    image="./scene.jpg",
    question="How many people are visible?",
)

# Table QA
table = client.question_answering.table(
    model="your-table-qa-model",
    table={
        "header": ["City", "Population"],
        "rows": [["Toronto", "2.9M"], ["Montreal", "1.8M"]],
    },
    query="Which city has more people?",
)
```

| Method | Inputs |
| --- | --- |
| `text` | `question`, `context` |
| `document` | `image`, `question`, optional `word_boxes` |
| `visual` | `image`, `question` |
| `table` | `table`, `query` |
