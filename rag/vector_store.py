import os
import glob
import pickle
import hashlib

import faiss
import numpy as np

from rag.embedder import create_embeddings
from config import ( KNOWLEDGE_DIR, FAISS_INDEX_DIR,HASH_FILE,)


def calculate_folder_hash(folder_path=KNOWLEDGE_DIR):

    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")) +
        glob.glob(os.path.join(folder_path, "*.md")))

    if not files:
        raise FileNotFoundError(f"No CSV or Markdown files found inside {folder_path}")

    hasher = hashlib.md5()

    for file in files:

        with open(file, "rb") as f:

            while chunk := f.read(8192):
                hasher.update(chunk)

    return hasher.hexdigest()



def save_hash(folder_hash):

    with open(HASH_FILE, "w") as f:
        f.write(folder_hash)



def load_hash():

    if os.path.exists(HASH_FILE):

        with open(HASH_FILE, "r") as f:
            return f.read().strip()

    return None


def build_vector_store():

    embedded_docs = create_embeddings()
    dimension = len(embedded_docs[0]["embedding"])


    print(f"Embedding Dimension : {dimension}")

    index = faiss.IndexFlatL2(dimension)
    vectors = np.array(  [doc["embedding"] for doc in embedded_docs],  dtype=np.float32 )

    index.add(vectors)
    documents = [doc["document"]for doc in embedded_docs]

    return index, documents


def save_vector_store(index, documents):

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    faiss.write_index( index, str(FAISS_INDEX_DIR / "index.faiss") )
    
    with open(FAISS_INDEX_DIR / "documents.pkl", "wb" ) as f:
        pickle.dump(documents, f)


def load_vector_store():


    index = faiss.read_index( str(FAISS_INDEX_DIR / "index.faiss") )

    with open( FAISS_INDEX_DIR / "documents.pkl", "rb") as f:
        documents = pickle.load(f)

    return index, documents



if __name__ == "__main__":

    current_hash = calculate_folder_hash()
    previous_hash = load_hash()

    if current_hash == previous_hash:

        print("Knowledge Base is already up-to-date.")

    else:

        print("Creating Vector Store...")

        index, documents = build_vector_store()

        save_vector_store(index, documents)
        save_hash(current_hash)

        print("\nVector Store Created Successfully!")