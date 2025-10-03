### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 02/10/2025
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:
```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?",
}

# --- Build TF-IDF ---
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents.values()).toarray()
terms = vectorizer.get_feature_names_out()
idf_values = vectorizer.idf_

# --- Term Frequency (TF) ---
tf_matrix = np.zeros_like(tfidf_matrix, dtype=int)
for i, doc in enumerate(documents.values()):
    words = doc.lower().split()
    for j, term in enumerate(terms):
        tf_matrix[i, j] = words.count(term)

# --- Document Frequency (DF) ---
df = np.sum(tf_matrix > 0, axis=0)

# --- Magnitude of each document vector ---
magnitudes = np.linalg.norm(tfidf_matrix, axis=1)

# --- Cosine similarity and dot product ---
def cosine_and_dot(query_vec, doc_vec):
    dot = np.dot(query_vec, doc_vec)
    norm_query = np.linalg.norm(query_vec)
    norm_doc = np.linalg.norm(doc_vec)
    if norm_query == 0 or norm_doc == 0:
        cosine = 0.0
    else:
        cosine = dot / (norm_query * norm_doc)
    return cosine, dot

# --- Get query from user ---
query = input("Enter your query: ")
query_vec = vectorizer.transform([query]).toarray()[0]

# --- Build term-level table ---
rows = []
for i, doc_id in enumerate(documents.keys()):
    cosine, dot = cosine_and_dot(query_vec, tfidf_matrix[i])
    for j, term in enumerate(terms):
        rows.append({
            "Document": doc_id,
            "Term": term,
            "TF": tf_matrix[i, j],
            "DF": int(df[j]),
            "IDF": round(idf_values[j], 4),
            "Weight(TF*IDF)": round(tfidf_matrix[i, j], 4),
            "Magnitude": round(magnitudes[i], 4),
            "Cosine with Query": round(cosine, 4),
            "Dot with Query": round(dot, 4)
        })

final_table = pd.DataFrame(rows)

# --- Ranking Table (only Document + Rank) ---
rank_df = final_table.groupby("Document").first().reset_index()
rank_df = rank_df[["Document", "Cosine with Query"]]
rank_df["Rank"] = rank_df["Cosine with Query"].rank(method="dense", ascending=False).astype(int)
rank_df = rank_df.sort_values(by="Rank").reset_index(drop=True)
rank_df = rank_df[["Document", "Rank"]]  # Keep only Document + Rank

# --- Display neatly ---
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print("\n=== Full TF-IDF & Similarity Table ===")
print(final_table)

print("\n=== Document Ranking ===")
print(rank_df)
```
### Output:
<img width="821" height="338" alt="Screenshot 2025-10-03 163632" src="https://github.com/user-attachments/assets/92ede55c-5ef6-4325-860c-4dc98ee3ce04" />

### Result:
Thus, We had successfully implemented the "Information Retrieval Using Vector Space Model in Python."
