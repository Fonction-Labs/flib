from qdrant_client.http.models import Distance, VectorParams, ScoredPoint

from qdrant_client import QdrantClient


class SingleCollectionQdrantClient(QdrantClient):
    """
    host (str): either ':memory:', or 'http://<myadress>:<myport>' (default Qdrant port is 6333)
    """

    def __init__(self, host: str, collection_name: str):
        super().__init__(host)
        self.collection_name = collection_name


def create_qdrant_client(
    host: str, collection_name: str, embedding_vector_size: int
) -> SingleCollectionQdrantClient:
    qdrant_client = SingleCollectionQdrantClient(host, collection_name)
    qdrant_client.recreate_collection(
        collection_name=qdrant_client.collection_name,
        vectors_config=VectorParams(
            size=embedding_vector_size, distance=Distance.COSINE
        ),
    )
    return qdrant_client


def get_all_points_from_collection(
    qdrant_client: SingleCollectionQdrantClient, vector_size: int, limit: int = 100
) -> list[ScoredPoint]:
    """
    Retrieves all elements from a given (single-collection) Qdrant database.
    There is hard-limit to the numbers of retrievable elements, set to 100 by default.
    """
    return qdrant_client.search(
        qdrant_client.collection_name,
        query_vector=[
            0.0 for i in range(vector_size)
        ],  # Dummy vector, seems necessary, unfortunately... (to get rid of if we can)
        limit=limit,  # Hard-limit
    )
