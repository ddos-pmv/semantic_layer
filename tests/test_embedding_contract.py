import numpy as np

from conftest import MODEL_PATH, REQUIRED_MODEL_FILES, missing_model_files


EMBEDDING_DIMENSION = 384


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def nearest_rows(anchor, rows, text_key, embedding_key, top_k):
    result = []

    for item in rows:
        if item is anchor:
            continue

        result.append({
            "text": item[text_key],
            "spam_type": item["spam_type"],
            "similarity": cosine_similarity(
                anchor[embedding_key],
                item[embedding_key],
            ),
        })

    result.sort(key=lambda row: row["similarity"], reverse=True)
    return result[:top_k]


def same_type_count(anchor, top_rows):
    return sum(row["spam_type"] == anchor["spam_type"] for row in top_rows)


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

    count = 0
    for rank, row in enumerate(top_rows, start=1):
        same_type = row["spam_type"] == anchor["spam_type"]
        if same_type:
            count += 1

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
    print(f"  same type in top {len(top_rows)}: {count}/{len(top_rows)}")


def assert_top_k_quality(rows, text_key, embedding_key, top_k, min_same_type):
    worst_anchor = None
    worst_top_rows = None
    worst_same_type_count = top_k + 1

    for anchor in rows:
        top_rows = nearest_rows(anchor, rows, text_key, embedding_key, top_k)
        current_same_type_count = same_type_count(anchor, top_rows)

        if current_same_type_count < worst_same_type_count:
            worst_anchor = anchor
            worst_top_rows = top_rows
            worst_same_type_count = current_same_type_count

    printable_anchor = {
        "text": worst_anchor[text_key],
        "spam_type": worst_anchor["spam_type"],
    }
    print_nearest_report(printable_anchor, worst_top_rows)

    assert worst_same_type_count >= min_same_type


def test_dataset_schema(dataset, dataset_grouped_by_type):
    assert dataset

    for row in dataset:
        assert set(row) == {"spam_type", "en", "ru"}
        assert row["spam_type"]
        assert row["en"]
        assert row["ru"]

    for rows in dataset_grouped_by_type.values():
        assert len(rows) >= 2


def test_local_model_files_are_complete():
    assert not missing_model_files(), (
        f"{MODEL_PATH} is incomplete. "
        f"Required files: {list(REQUIRED_MODEL_FILES)}"
    )


def test_en_embedding(model, en_texts):
    embeddings = model.encode(en_texts, convert_to_numpy=True)

    assert len(en_texts) == len(embeddings)
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == EMBEDDING_DIMENSION
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()


def test_ru_embedding(model, ru_texts):
    embeddings = model.encode(ru_texts, convert_to_numpy=True)

    assert len(ru_texts) == len(embeddings)
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == EMBEDDING_DIMENSION
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()


def test_top_7_nearest_en(embedded_dataset):
    assert_top_k_quality(
        rows=embedded_dataset,
        text_key="en",
        embedding_key="en_embedding",
        top_k=7,
        min_same_type=3,
    )


def test_top_7_nearest_ru(embedded_dataset):
    assert_top_k_quality(
        rows=embedded_dataset,
        text_key="ru",
        embedding_key="ru_embedding",
        top_k=7,
        min_same_type=2,
    )
