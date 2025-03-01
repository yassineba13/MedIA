"""API"""
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

from ingest import create_cloud_sql_database_connection, get_embeddings, get_vector_store
from retrieve import get_format_relevant_documents
load_dotenv()

app = FastAPI()

# Initialize once and reuse
ENGINE = create_cloud_sql_database_connection()
EMBEDDING = get_embeddings()
vector_store = get_vector_store(ENGINE, EMBEDDING)

class UserInput(BaseModel):
    """
    UserInput is a data model representing user input.

    Attributes:
        question (str): The question of the user.
        history (List[str]): The history of the chat.

    """
    question: str
    history: List[str] = []


@app.post("/answer")
def answer(user_input: UserInput):

        # Add the question to the chat history
        historique_du_chat = user_input.history
        historique_du_chat.append(user_input.question)

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                (
                    "system",
                    """You are MEDIA, a medical assistant. You are here to help people with their medical questions.
                    answer ploitely and provide the best information possible.
                    here is the history of the chat: {history}
                    here are the relevant documents: {reference}
                    use the following information to answer the question: {question}
                    instructions:
                    0.use only the given refrence to answer the user question.
                    1.don't provide any medical advice.
                    2. don't provide any personal information.
                    3. don't provide any information that is not in the reference.
                    4. don't provide any information that is not related to the question.
                    5.if the question is not clear, ask the user to provide more information.
                    6.if the user ask for medical advice, tell them to consult a doctor.
                    7.if the user ask the question in a different language,answer the question in that language.
                    8.always tell the user to consult a doctor if they have any medical concerns.
                    9.always add to the answer that the information is from the reference and the focus area of the reference as provided in the reference.
                    

                    """,
                ),
                ("human", "The question is: {question}"),
            ]
        )

        chain = prompt | llm
        answer=chain.invoke({
                "question": user_input.question,
                "history": historique_du_chat,
                "reference": get_format_relevant_documents(user_input.question, vector_store)
            }).content
        return {"message": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8501)