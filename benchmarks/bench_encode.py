from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer(model_name_or_path='models/all-MiniLM-L6-v2', local)
embeddings = model.encode(sentences)
print(embeddings)