import os
from dotenv import load_dotenv
from langchain_google_cloud_sql_pg import PostgresEngine, PostgresVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from config import PROJECT_ID, INSTANCE, REGION, DATABASE, DB_USER, TABLE_NAME


load_dotenv()
DB_PASSWORD = os.environ["DB_PASSWORD"]


def create_cloud_sql_database_connection() -> PostgresEngine:
    engine = PostgresEngine.from_instance(
        project_id=PROJECT_ID,
        instance=INSTANCE,
        region=REGION,
        database=DATABASE,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    return engine


def get_embeddings() -> VertexAIEmbeddings:
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=PROJECT_ID
    )
    return embeddings


def get_vector_store(engine: PostgresEngine,embedding: VertexAIEmbeddings) -> PostgresVectorStore:
    vector_store = PostgresVectorStore.create_sync(
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=embedding,
    )
    return vector_store

if __name__ == '__main__':
        try:
            print("Testing database connection...")
            engine = create_cloud_sql_database_connection()
            print("✓ Database connection successful")

            print("\nTesting embeddings configuration...")
            embeddings = get_embeddings()
            print("✓ Embeddings configuration successful")

            print("\nTesting vector store access...")
            vector_store = get_vector_store(engine, embeddings)
            
            # Test simple query to verify everything works
            test_query = "What is glaucoma?"
            results = vector_store.similarity_search_with_score(test_query, k=1)
            if len(results) > 0:
                print("✓ Vector store access successful")
                print(f"✓ Successfully retrieved {len(results)} result(s)")
                doc, score = results[0]
                print("\nSample result:")
                print(f"Question: {doc.page_content}")
                print(f"Score: {score}")
            else:
                print("! No results found in vector store")

            print("\nAll tests completed successfully!")

        except Exception as e:
            print(f"\n❌ Error during testing: {str(e)}")