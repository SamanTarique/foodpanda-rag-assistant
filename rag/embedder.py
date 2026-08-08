from sentence_transformers import SentenceTransformer
from rag.chunker import create_chunks
from config import EMBEDDING_MODEL

BATCH_SIZE = 25

model = SentenceTransformer(EMBEDDING_MODEL)
def create_embeddings():

    chunks = create_chunks()

    embedded_documents = []

    for i in range(0, len(chunks), BATCH_SIZE):

        batch = chunks[i:i + BATCH_SIZE]

        texts = [chunk.page_content for chunk in batch]

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        for chunk, embedding in zip(batch, embeddings):

            embedded_documents.append({  "document": chunk,  "embedding": embedding })

    return embedded_documents


model = SentenceTransformer(EMBEDDING_MODEL)


def embed_query(query: str):

    embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    return embedding


if __name__ == "__main__":

    embedded_docs = create_embeddings()

    print(f"\nTotal Embedded Documents : {len(embedded_docs)}\n")

    print("=" * 70)

    print("Source:")
    print(embedded_docs[0]["document"].metadata)

    print("\nChunk:")
    print(embedded_docs[0]["document"].page_content[:300])

    print("\nEmbedding Dimension:")
    print(len(embedded_docs[0]["embedding"]))

    print("\nFirst 10 Values:")
    print(embedded_docs[0]["embedding"][:10])