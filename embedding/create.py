import json
from sentence_transformers import SentenceTransformer
import os

# Get current working directory
cwd = os.getcwd()
filename = cwd + "/hupd_extracted/2018/16062262.json"


# from transformers import AutoTokenizer, AutoModel
# import torch

# Load SPECTER2 model and tokenizer from Hugging Face
# MODEL_NAME = "allenai/specter2"
# MODEL_NAME = "allenai/specter2_base"
model = SentenceTransformer("allenai/specter2_base")

# Load documents
with open(filename, "r", encoding="utf-8") as f:
    doc = json.load(f)

embeddings = []
# for doc in documents:
title = doc.get("title", "")
abstract = doc.get("abstract", "")
text = f"{title} [SEP] {abstract}"
emb = model.encode(text)
embeddings.append({
    "title": title,
    "abstract": abstract,
    "embedding": emb.tolist()
})

# Save to output JSON
with open("specter2_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=2)





# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME)

# # Function to create document embeddings
# def get_embedding(title, abstract):
#     text = f"{title} {tokenizer.sep_token} {abstract}"
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         # Use [CLS] token representation (first token)
#         cls_embedding = outputs.last_hidden_state[:, 0, :]
#     return cls_embedding.squeeze().numpy()

# # Load JSON file (list of documents with 'title' and 'abstract')
# with open("../hupd_extracted/2018/16062262.json", "r", encoding="utf-8") as f:
#     documents = json.load(f)

# # Generate and store embeddings
# embeddings = []
# for doc in documents:
#     title = doc.get("title", "")
#     abstract = doc.get("abstract", "")
#     embedding = get_embedding(title, abstract)
#     embeddings.append({
#         "title": title,
#         "abstract": abstract,
#         "embedding": embedding.tolist()
#     })

# # Save embeddings to new JSON
# with open("specter2_embeddings.json", "w", encoding="utf-8") as f:
#     json.dump(embeddings, f, indent=2)
