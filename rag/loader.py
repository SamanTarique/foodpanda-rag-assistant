from pathlib import Path
import pandas as pd

from config import KNOWLEDGE_DIR


def load_knowledge():
    docs = []

    csv_files = sorted(KNOWLEDGE_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found inside: {KNOWLEDGE_DIR}"
        )

    for file in csv_files:

        df = pd.read_csv(file)

        for _, row in df.iterrows():

            docs.append({
                "source": file.name,
                "text": "\n".join(f"{col}: {row[col]}" for col in df.columns)
            })

    print(f"Loaded {len(csv_files)} CSV file(s).")



    md_files = sorted(KNOWLEDGE_DIR.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(
            f"No Markdown files found inside: {KNOWLEDGE_DIR}"
        )

    for file in md_files:

        with open(file, "r", encoding="utf-8") as f:

            docs.append({
                "source": file.name,
                "text": f.read()
            })

    print(f"Loaded {len(md_files)} Markdown file(s).")

    return docs


if __name__ == "__main__":

    docs = load_knowledge()

    print("\nKnowledge Files Loaded Successfully!\n")
    print(f"Total Documents: {len(docs)}\n")

    print("Sample Documents:\n")

    for i, doc in enumerate(docs[:5], start=1):

        print("=" * 60)
        print(f"Document {i}")
        print(f"Source : {doc['source']}")
        print(f"Text   :\n{doc['text'][:200]}")
        print()

