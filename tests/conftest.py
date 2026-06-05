import json

import pytest
from sentence_transformers import SentenceTransformer


MODEL_PATH = "models/paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = "eval_data/spam_paraphrases.jsonl"


@pytest.fixture(scope="session")
def model():
    model = SentenceTransformer(MODEL_PATH, local_files_only=True)

    embedding = model.encode(["Test sentence."])

    print()
    print(f"Embedding shape: {embedding.shape}")

    return model


@pytest.fixture(scope="session")
def dataset():
    with open(DATA_PATH, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


@pytest.fixture(scope="session")
def embedded_dataset(model, dataset):
    en_embeddings = model.encode(
        [row["en"] for row in dataset],
        convert_to_numpy=True,
    )
    ru_embeddings = model.encode(
        [row["ru"] for row in dataset],
        convert_to_numpy=True,
    )

    rows = []
    for row, en_embedding, ru_embedding in zip(dataset, en_embeddings, ru_embeddings):
        rows.append({
            **row,
            "en_embedding": en_embedding,
            "ru_embedding": ru_embedding,
        })

    return rows


@pytest.fixture(scope="session")
def dataset_grouped_by_type(dataset):
    grouped = dict()

    for row in dataset:
        spam_type = row["spam_type"]
        grouped.setdefault(spam_type, [])
        grouped[spam_type].append({"en": row["en"], "ru": row["ru"]})

    return grouped


@pytest.fixture(scope="function")
def ru_texts(dataset):
    return [row["ru"] for row in dataset]


@pytest.fixture(scope="function")
def en_texts(dataset):
    return [row["en"] for row in dataset]
