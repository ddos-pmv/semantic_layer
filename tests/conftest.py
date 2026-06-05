import pytest
import json
from sentence_transformers import SentenceTransformer

MODEL_PATH = "models/paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = "eval_data/spam_paraphrases.jsonl"

@pytest.fixture(scope="session")
def model():
    model = SentenceTransformer(MODEL_PATH, local_files_only=True)

    embedding = model.encode(["Test sentence here. Happy bd mother fucker"])

    print()
    print(f'Embedding shape: {embedding.shape}')

    return model


@pytest.fixture(scope="session")
def dataset(model):
    with open(DATA_PATH, encoding="utf-8") as file:
        dataset = [json.loads(line) for line in file]
        for row in dataset:
            row["ru_embedding"] = model.encode(row["ru"], )
            row["en_embedding"] = model.encode(row["en"])
        return dataset
@pytest.fixture(scope="session")
def dataset_grouped_by_type():
    with open(DATA_PATH, encoding="utf-8") as file:
        dataset = dict()

        for line in file:
            row = json.loads(line)
            spam_type = row["spam_type"]
            dataset.setdefault(spam_type, [])
            dataset[spam_type].append({"en": row["en"], "ru": row["ru"]})
    return dataset




@pytest.fixture(scope="function")
def ru_texts(dataset):
    return [row["ru"] for row in dataset]


@pytest.fixture(scope="function")
def en_texts(dataset):
    return [row["en"] for row in dataset]







