"""Module providing an example of Firestore Vector Similarity Search"""

import argparse
import logging
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from settings import get_settings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

settings = get_settings()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Search documents in a Firestore vector store."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="The search query prompt.",
    )
    return parser.parse_args()


def search(query: str):
    """Executes Firestore Vector Similarity Search"""
    embedding = VertexAIEmbeddings(
        model_name=settings.embedding_model_name,
        project=settings.project_id,
        location=settings.location,
    )

    client = MongoClient(settings.mongodb_atlas_cluster_uri)

    mongodb_collection = client[settings.db_name][settings.collection_name]

    vector_store = MongoDBAtlasVectorSearch(
        collection=mongodb_collection,
        embedding=embedding,
        index_name=settings.atlas_vector_search_index_name,
        relevance_score_fn="cosine",
    )
    
    logging.info(f"Now executing query: {query}")
    results: list[Document] = vector_store.similarity_search(
        query=query, k=int(settings.top_k), include_metadata=True
    )
    for result in results:
        print(result.page_content)


# Example: Suggest some exercises to relieve back pain
# python search-data.py --prompt "Suggest some exercises to relieve back pain"
if __name__ == "__main__":
    args = parse_arguments()
    prompt = args.prompt
    search(prompt)
