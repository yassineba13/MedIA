"""API"""
from typing import List
from langchain_google_vertexai import VertexAIEmbeddings
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_google_cloud_sql_pg import PostgresEngine
from langchain_google_cloud_sql_pg import PostgresEngine

from ingest import create_cloud_sql_database_connection, get_embeddings, get_vector_store
from retrieve import get_relevant_documents, format_relevant_documents
load_dotenv()

app = FastAPI()

# Initialize once and reuse
ENGINE = create_cloud_sql_database_connection()
EMBEDDING = get_embeddings()


class UserInput(BaseModel):
    """
    UserInput is a data model representing user input.

    Attributes:
        question (str): The question of the user.

    """
    question: str


class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict


@app.post("/get_sources", response_model=List[DocumentResponse])
def get_sources(user_input: UserInput) -> List[DocumentResponse]:
    vector_store = get_vector_store(ENGINE, EMBEDDING)
    relevants_docs = get_relevant_documents(user_input.question, vector_store)
    frelevdoc= format_relevant_documents(relevants_docs)
    return [frelevdoc]


@app.post("/answer")
def answer(user_input: UserInput):
    """
    Generates a greeting message based on the user's input.
    Args:
        user_input (UserInput): An object containing user details such as name, genre, and language.
    Returns:
        dict: A dictionary containing a greeting message.
    """

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
                "You are a question answering chatbot. .",
            ),
            ("human", "The question is: {question}"),
        ]
    )

    chain = prompt | llm
    answer=chain.invoke({
            "question": user_input.question,
        }).content
    return {"message": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8501)