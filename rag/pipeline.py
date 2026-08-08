from google import genai

from config import GEMINI_API_KEY, GEMINI_MODEL
from rag.prompt import format_prompt
from rag.retriever import retrieve


client = genai.Client(api_key=GEMINI_API_KEY)


def generate_answer(question: str):

    documents = retrieve(question)

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = format_prompt(context=context,question=question)

    response = client.models.generate_content( model=GEMINI_MODEL,contents=prompt)

    return response.text


if __name__ == "__main__":

    question = "How can I get a refund?"

    answer = generate_answer(question)

    print(answer)
    
 