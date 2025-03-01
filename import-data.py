import logging
import json
from datasets import load_dataset
from google.cloud import firestore
from langchain_core.documents import Document
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from settings import get_settings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from uuid import uuid4

settings = get_settings()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_yoga_poses_data_from_hugging_face():
    """
    Loads the Yoga poses dataset.
    Hugging Face Dataset: https://huggingface.co/datasets/omergoshen/yoga_poses
    """
    try:
        # Load the yoga json array into memory from a file
        dataset = load_dataset("omergoshen/yoga_poses")
        poses = dataset["train"].to_list()
        logging.info(f"Loaded {len(poses)} poses.")
        return poses
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None


def load_yoga_poses_data_from_local_file(filename):
    """
    Loads the Yoga poses dataset from a local file
    """
    try:
        # Load the yoga json array into memory from a file
        poses = json.loads(open(filename).read())
        logging.info(f"Loaded {len(poses)} poses.")
        return poses
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None


def create_langchain_documents(poses):
    """Creates a list of Langchain Documents from a list of poses."""
    documents = []
    for pose in poses:
        # Convert the pose to a string representation for page_content
        page_content = (
            f"name: {pose.get('name', '')}\n"
            f"description: {pose.get('description', '')}\n"
            f"sanskrit_name: {pose.get('sanskrit_name', '')}\n"
            f"expertise_level: {pose.get('expertise_level', 'N/A')}\n"
            f"pose_type: {pose.get('pose_type', 'N/A')}\n"
        ).strip()
        # The metadata will be the whole pose
        metadata = pose

        document = Document(page_content=page_content, metadata=metadata)
        documents.append(document)
    logging.info(f"Created {len(documents)} Langchain documents.")
    return documents


def create_mongodb_collection_and_search_index(embeddings):
    # initialize MongoDB python client
    client = MongoClient(settings.mongodb_atlas_cluster_uri)

    mongodb_collection = client[settings.db_name][settings.collection_name]

    vector_store = MongoDBAtlasVectorSearch(
        collection=mongodb_collection,
        embedding=embeddings,
        index_name=settings.atlas_vector_search_index_name,
        relevance_score_fn="cosine",
    )

    # Create vector search index on the collection
    vector_store.create_vector_search_index(dimensions=int(settings.embedding_size))
    
    return vector_store


def main():
    all_poses = load_yoga_poses_data_from_local_file(
        "./data/yoga_poses_with_descriptions.json"
    )
    documents = create_langchain_documents(all_poses)
    logging.info(
        f"Successfully created langchain documents. Total documents: {len(documents)}"
    )

    embedding = VertexAIEmbeddings(
        model_name=settings.embedding_model_name,
        project=settings.project_id,
        location=settings.location,
    )

    vector_store = create_mongodb_collection_and_search_index(embeddings=embedding)

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)

    logging.info("Added documents to the vector store.")


if __name__ == "__main__":
    main()
