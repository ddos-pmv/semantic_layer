import json
from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer


MODEL_PATH = "models/paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = "eval_data/spam_paraphrases.jsonl"
REQUIRED_MODEL_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "model.onnx",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def missing_model_files():
    model_path = Path(MODEL_PATH)
    return [
        file
        for file in REQUIRED_MODEL_FILES
        if not (model_path / file).is_file()
    ]


@pytest.fixture(scope="session")
def model():
    missing_files = missing_model_files()
    assert not missing_files, (
        "Local model is incomplete. Run core/scripts/download_model.sh. "
        f"Missing files: {missing_files}"
    )

    model = SentenceTransformer(MODEL_PATH, local_files_only=True)

    embedding = model.encode(["Some something someone somehow sometimes hate you"])

    print()
    print(f"Embedding shape: {embedding.shape}")
    print(*embedding.tolist(), sep="\n")

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
