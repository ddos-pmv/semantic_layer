import copy
import pytest
import random
import numpy as np

from conftest import *

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def print_nearest_report(anchor, top_rows):
    def short_text(text, max_len=70):
        if len(text) <= max_len:
            return text

        return text[:max_len - 3] + "..."

    print()
    print("ANCHOR")
    print(f" spam_type: {anchor['spam_type']}")
    print(f" text: {anchor['text']}")
    print()
    print(f"TOP {len(top_rows)} NEAREST")
    print(" rank | score | same_type | type | text")
    print("  -----+--------+-----------+-------------------------+------------------------------------------")
    same_type_count = 0

    for rank, row in enumerate(top_rows, start=1):
        same_type = row["spam_type"] == anchor["spam_type"]

        if same_type:
            same_type_count += 1

        same_type_text = "yes" if same_type else "no"

        print(
            f"  {rank:>4} | "
            f"{row['similarity']:>6.3f} | "
            f"{same_type_text:<9} | "
            f"{row['spam_type']:<23} | "
            f"{short_text(row['text'])}"
        )

    print()
    print("SUMMARY")
    print(f"  same type in top {len(top_rows)}: {same_type_count}/{len(top_rows)}")


@pytest.fixture(scope="function")
def en_ru_pairs(en_rows, ru_rows):
    return list(zip(en_rows, ru_rows))

def test_en_embedding(model, en_texts):
    cntTexts = len(en_texts)
    embeddings = model.encode(en_texts, convert_to_numpy=True)

    assert cntTexts == len(embeddings)

    assert not np.isnan(embeddings).any()
    assert embeddings[0].dtype == np.float32
    assert embeddings[-1].dtype == np.float32


def test_ru_embedding(model, ru_texts):
    cntTexts = len(ru_texts)
    embeddings = model.encode(ru_texts)

    assert cntTexts == len(embeddings)

    assert not np.isnan(embeddings).any()
    assert not np.isnan(embeddings).any()
    assert embeddings.dtype == np.float32
    assert embeddings.dtype == np.float32




def test_top_7_nearest_en(model, dataset):
    # data = copy.deepcopy(dataset)

    row_idx = random.randint(0, len(dataset) - 1)

    anchor = dataset[row_idx]
    print(anchor["en"], anchor["spam_type"])


    result =  []

    for item in dataset:
        if item is anchor:
            continue
        score =  cosine_similarity(anchor["en_embedding"], item["en_embedding"])

        result.append({
            "text": item["en"],
            "spam_type": item["spam_type"],
            "similarity": score
        })

    result.sort(key=lambda x: x["similarity"], reverse=True)

    top_7 = result[:7]

    anchor = {
        "text" : anchor["en"],
        "spam_type" : anchor["spam_type"],
    }

    print_nearest_report(anchor, top_7)



def test_top_7_nearest_ru(model, dataset):
    row_idx = random.randint(0, len(dataset) - 1)

    anchor = dataset[row_idx]
    print(anchor["ru"], anchor["spam_type"])


    result =  []

    for item in dataset:
        if item is anchor:
            continue
        score =  cosine_similarity(anchor["ru_embedding"], item["ru_embedding"])

        result.append({
            "text": item["ru"],
            "spam_type": item["spam_type"],
            "similarity": score
        })

    result.sort(key=lambda x: x["similarity"], reverse=True)

    top_7 = result[:7]

    anchor = {
        "text" : anchor["ru"],
        "spam_type" : anchor["spam_type"],
    }

    print_nearest_report(anchor, top_7)











