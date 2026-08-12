
# Foodpanda RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers Foodpanda-related questions using a controlled knowledge base of policies, FAQs, payment information, customer support information, and other public resources. The system retrieves the most relevant information from the knowledge base using FAISS and generates grounded responses with Google Gemini.

[![Python](https://img.shields.io/badge/language-Python-lightgrey)](#)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://your-live-demo-url.example.com)
[![Watch Demo](https://img.shields.io/badge/Watch-Demo-red)](https://youtu.be/YOUR_VIDEO_ID)

Short one-line project description here.

## Demo:


https://github.com/user-attachments/assets/d1960777-464f-4b45-b7b3-57dc5dd92cd8



**Live Demo Link** : https://foodpanda-rag-assistant-m7vjgaijsittwxutdj435d.streamlit.app/


---

## Features

- Retrieval-Augmented Generation (RAG) architecture
- FAISS vector search for semantic retrieval
- Sentence Transformers for document embeddings
- Google Gemini for answer generation
- Strict prompt grounding to reduce hallucinations
- Automatic knowledge-base hash checking
- Automatic FAISS index rebuilding when knowledge files change
- Markdown and CSV knowledge-base support
- Streamlit chat interface
- Evaluation workflow for testing chatbot responses
- Conversation history through Streamlit session state

## Project Structure

```text
Foodpanda-RAG-Assistant/
│
├── app.py
├── config.py
├── requirements.txt
├── check.py
│
├── knowledge/
│   ├── company_overview.md
│   ├── faq.md
│   ├── refund_policy.md
│   ├── delivery_policy.md
│   ├── privacy_policy.md
│   ├── terms_conditions.md
│   ├── customer_support.md
│   ├── pandamart.md
│   ├── pandapay_wallet.md
│   ├── payment_methods.md
│   ├── payment_methods.csv
│   ├── restaurants.csv
│   └── support_contacts.csv
│
├── rag/
│   ├── __init__.py
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompt.py
│   └── pipeline.py
│
├── faiss_index/
├── logs/
│   └── evaluation.csv
│
└── tests/
    └── test.py
```


## RAG Pipeline

```text
User Question
      ↓
Streamlit Interface
      ↓
Retriever
      ↓
FAISS Vector Search
      ↓
Relevant Documents
      ↓
Prompt Template
      ↓
Google Gemini
      ↓
Grounded Answer
```

## Technologies

* Python
* Streamlit
* Google Gemini API
* Sentence Transformers
* FAISS
* NumPy
* Pandas
* LangChain Core
* LangChain Text Splitters

## Knowledge Base

The chatbot uses a local knowledge base containing information related to:

* Company overview
* Frequently asked questions
* Refund policy
* Delivery policy
* Privacy policy
* Terms and conditions
* Customer support
* pandamart
* pandapay wallet
* Payment methods
* Restaurants
* Support contacts

The chatbot is instructed to answer only from retrieved knowledge. If sufficient information cannot be found, it returns:

> I couldn't find that information in the knowledge base.

## Installation

Clone the repository:

```bash
git clone https://github.com/SamanTarique/foodpanda-rag-assistant.git
cd foodpanda-rag-assistant
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-flash-latest
```

Do not commit `.env` or expose your API key publicly.

## Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser.

## Testing

The project includes an evaluation workflow with sample questions.

Run:

```bash
python -m tests.test
```

The test workflow sends the evaluation questions through the RAG pipeline and records the generated answers in:

```text
logs/evaluation.csv
```

This file can then be reviewed to evaluate answer quality and hallucination behavior.

## FAISS Index

The vector store automatically checks whether the knowledge base has changed using a folder hash.

If the knowledge files are unchanged, the existing index can be reused.

If the knowledge files are changed, the FAISS vector store is rebuilt so that the retrieved information remains synchronized with the knowledge base.

## Hallucination Control

The prompt layer applies strict grounding rules:

* Answers must use retrieved information only.
* External knowledge must not be introduced.
* Missing information must not be guessed.
* Multiple relevant retrieved documents can be combined.
* The response should remain concise and customer-friendly.

## Security

API keys and environment files are excluded from version control.

Never commit:

```text
.env
.venv/
__pycache__/
faiss_index/
```

## Limitations

* Response quality depends on the quality and coverage of the knowledge base.
* The chatbot cannot reliably answer questions that are outside the indexed information.
* Gemini API availability and usage limits may affect generation.
* FAISS is used as a local vector store and is intended for this project's lightweight RAG setup.

## Future Improvements

* Add source citations to chatbot responses
* Add retrieval-score display for debugging
* Improve evaluation metrics
* Add automated hallucination detection
* Add more Foodpanda knowledge documents
* Deploy the application for public access

## Author

**Saman Tarique**

BS Artificial Intelligence Student

GitHub: [https://github.com/SamanTarique](https://github.com/SamanTarique)
