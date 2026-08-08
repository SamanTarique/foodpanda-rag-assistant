import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import MAX_CHUNK_SIZE, CHUNK_OVERLAP
from rag.loader import load_knowledge




def split_markdown(text: str) -> list[str]:
    parts = re.split(r"\n(?=## )", text)
    return [part.strip() for part in parts if part.strip()]




def split_faq(text: str) -> list[str]:
    parts = re.split(r"\n(?=Q:|\*\*Q:|Question:)", text)
    return [part.strip() for part in parts if part.strip()]



def recursive_split(text: str) -> list[str]:

    if len(text) <= MAX_CHUNK_SIZE:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_text(text)



def create_chunks() -> list[Document]:

    docs = load_knowledge()
    print(type(docs))
    print(type(docs[0]))
    print(docs[0])
   

    all_chunks = []

    for doc in docs:

        source = doc["source"]
        text = doc["text"]

        if source.endswith(".csv"):

            all_chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source,
                        "type": "csv"
                    }
                )
            )

            continue
        
        if source == "faq.md":
            sections = split_faq(text)

        else:
            sections = split_markdown(text)

        for section in sections:

            small_chunks = recursive_split(section)

            for chunk in small_chunks:

                all_chunks.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": source,
                            "type": "markdown"
                        }
                    )
                )

    return all_chunks


if __name__ == "__main__":

    chunks = create_chunks()
    

    print(f"\nTotal Chunks : {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:5], start=1):

        print("=" * 70)
        print(f"Chunk {i}")
        print(chunk.metadata)
        print(chunk.page_content[:300])
        print()   
        
    
    
