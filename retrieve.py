from langchain_google_cloud_sql_pg import PostgresVectorStore
from langchain_core.documents.base import Document
from ingest import create_cloud_sql_database_connection, get_embeddings, get_vector_store

def get_relevant_documents(query: str, vector_store: PostgresVectorStore) -> list[Document]:
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 2, 'lambda_mult': 0.3}
    )
    return retriever.invoke(query)

def format_relevant_documents(documents: list[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(documents):
        formatted_doc = (
            f"Question {i+1}: {doc.page_content}\n"
            f"Answer: {doc.metadata['answer']}\n"
            f"Source: {doc.metadata['source']}\n"
            f"Focus Area: {doc.metadata['focus_area']}\n"
            "-----"
        )
        formatted_docs.append(formatted_doc)
    return "\n".join(formatted_docs)

if __name__ == '__main__':
    engine = create_cloud_sql_database_connection()
    embedding = get_embeddings()
    vector_store = get_vector_store(engine, embedding)
    
    test_query = "What is fever?"
    documents = get_relevant_documents(test_query, vector_store)
    
    # Ajoutons un print du score
    results = vector_store.similarity_search_with_score(test_query, k=1)
    if results:
        score = results[0][1]
        print(f"Score de similarité : {score}\n")
    
    doc_str = format_relevant_documents(documents)
    print("Document le plus pertinent :")
    print(doc_str)
    print(type(doc_str))
    print("\nTest passed successfully.")