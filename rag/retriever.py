import numpy as np
from rag.vector_store import load_vector_store
from rag.embedder import embed_query

def retrieve(question: str, top_k: int = 5):

    index, documents = load_vector_store()
    query_embedding = embed_query(question)

    query_embedding = np.array(  [query_embedding],dtype=np.float32)
    distances, indices = index.search( query_embedding,top_k)

    retrieved_docs = []

    for idx in indices[0]:

        if idx != -1:

            retrieved_docs.append(  documents[idx])

    return retrieved_docs


if __name__ == "__main__":

    question = input("Ask Question: ")

    results = retrieve(question)

    print("\nRetrieved Chunks\n")

    for i, doc in enumerate(results, start=1):

        print("=" * 70)
        print(f"Chunk {i}")
        print(doc.metadata)
        print()
        print(doc.page_content[:400])
        print()

