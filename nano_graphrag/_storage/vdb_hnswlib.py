"""HNSW-based vector database storage implementation.

This module provides a vector storage implementation using the HNSW (Hierarchical
Navigable Small World) algorithm via the hnswlib library. HNSW is an efficient
approximate nearest neighbor search algorithm that provides fast similarity search
in high-dimensional vector spaces.

The storage supports:
- Efficient vector insertion and batch processing
- Fast k-nearest neighbor queries using cosine similarity
- Persistence of both the vector index and associated metadata
- Configurable HNSW parameters for tuning performance vs accuracy

Typical usage:
    storage = HNSWVectorStorage(
        namespace="my_vectors",
        global_config=config,
        embedding_func=embedding_function,
        meta_fields=["content", "source"]
    )
    await storage.upsert(data)
    results = await storage.query("search query", top_k=10)
"""
import asyncio
import os
from dataclasses import dataclass, field
from typing import Any
import pickle
import hnswlib
import numpy as np
import xxhash

from .._utils import logger
from ..base import BaseVectorStorage


@dataclass
class HNSWVectorStorage(BaseVectorStorage):
    """Vector storage implementation using HNSW (Hierarchical Navigable Small World) algorithm.

    This class provides efficient vector storage and similarity search capabilities using
    the hnswlib library. It stores vectors in an HNSW index for fast approximate nearest
    neighbor search and maintains associated metadata separately.

    The HNSW algorithm builds a multi-layer graph structure that enables logarithmic
    search complexity while maintaining high recall. This implementation uses cosine
    similarity as the distance metric.

    Attributes:
        ef_construction: Size of the dynamic candidate list during index construction.
            Higher values improve index quality but slow down construction. Default: 100.
        M: Maximum number of bi-directional links per element. Higher values improve
            recall but increase memory usage. Typical range: 12-48. Default: 16.
        max_elements: Maximum number of vectors the index can hold. Must be set before
            index creation. Default: 1000000.
        ef_search: Size of the dynamic candidate list during search. Higher values
            improve recall but slow down queries. Default: 50.
        num_threads: Number of threads to use for operations. -1 means use all available
            CPU cores. Default: -1.
        _index: The underlying hnswlib Index object. Initialized during __post_init__.
        _metadata: Dictionary mapping vector IDs to their associated metadata fields.
        _current_elements: Current number of elements stored in the index.

    Note:
        The index and metadata are persisted to disk automatically via the
        index_done_callback method. On initialization, existing indexes are loaded
        if available.
    """
    ef_construction: int = 100
    M: int = 16
    max_elements: int = 1000000
    ef_search: int = 50
    num_threads: int = -1
    _index: Any = field(init=False)
    _metadata: dict[str, dict] = field(default_factory=dict)
    _current_elements: int = 0

    def __post_init__(self):
        """Initialize the HNSW index and load existing data if available.

        This method is called automatically after dataclass initialization. It performs
        the following operations:
        1. Sets up file paths for the index and metadata storage
        2. Loads HNSW parameters from global config, overriding defaults if specified
        3. Creates a new hnswlib Index with cosine similarity metric
        4. Either loads an existing index from disk or initializes a new empty index

        The method looks for existing index files in the working directory specified
        in global_config. If both the index file and metadata file exist, they are
        loaded. Otherwise, a fresh index is created.

        Note:
            - All HNSW parameters (ef_construction, M, max_elements, ef_search, num_threads)
              can be overridden via global_config["vector_db_storage_cls_kwargs"]
            - The embedding dimension is automatically determined from the embedding function
            - File paths follow the pattern: {namespace}_hnsw.index and {namespace}_hnsw_metadata.pkl
        """
        self._index_file_name = os.path.join(
            self.global_config["working_dir"], f"{self.namespace}_hnsw.index"
        )
        self._metadata_file_name = os.path.join(
            self.global_config["working_dir"], f"{self.namespace}_hnsw_metadata.pkl"
        )
        self._embedding_batch_num = self.global_config.get("embedding_batch_num", 100)

        hnsw_params = self.global_config.get("vector_db_storage_cls_kwargs", {})
        self.ef_construction = hnsw_params.get("ef_construction", self.ef_construction)
        self.M = hnsw_params.get("M", self.M)
        self.max_elements = hnsw_params.get("max_elements", self.max_elements)
        self.ef_search = hnsw_params.get("ef_search", self.ef_search)
        self.num_threads = hnsw_params.get("num_threads", self.num_threads)
        self._index = hnswlib.Index(
            space="cosine", dim=self.embedding_func.embedding_dim
        )

        if os.path.exists(self._index_file_name) and os.path.exists(
            self._metadata_file_name
        ):
            self._index.load_index(
                self._index_file_name, max_elements=self.max_elements
            )
            with open(self._metadata_file_name, "rb") as f:
                self._metadata, self._current_elements = pickle.load(f)
            logger.info(
                f"Loaded existing index for {self.namespace} with {self._current_elements} elements"
            )
        else:
            self._index.init_index(
                max_elements=self.max_elements,
                ef_construction=self.ef_construction,
                M=self.M,
            )
            self._index.set_ef(self.ef_search)
            self._metadata = {}
            self._current_elements = 0
            logger.info(f"Created new index for {self.namespace}")

    async def upsert(self, data: dict[str, dict]) -> np.ndarray:
        """Insert or update vectors and their metadata in the index.

        This method takes a dictionary of documents, generates embeddings for their content,
        and adds them to the HNSW index. It also stores associated metadata for later retrieval.
        The embeddings are generated in batches to optimize performance.

        The method performs the following steps:
        1. Validates the input data and checks capacity constraints
        2. Extracts content and metadata fields from the input
        3. Generates embeddings in batches using the embedding function
        4. Creates unique integer IDs using xxhash for efficient indexing
        5. Stores metadata and adds vectors to the HNSW index

        Args:
            data: Dictionary mapping document IDs to document data. Each document should
                have a "content" field and may include additional metadata fields specified
                in self.meta_fields. Format: {doc_id: {"content": str, "field1": value1, ...}}

        Returns:
            NumPy array of uint32 integer IDs assigned to the inserted vectors. These IDs
            are generated using xxhash and are used internally by the HNSW index.

        Raises:
            ValueError: If inserting the data would exceed max_elements capacity.

        Note:
            - If data is empty, a warning is logged and an empty list is returned
            - Existing vectors with the same ID will be updated (upsert behavior)
            - Only metadata fields listed in self.meta_fields are stored, plus the "id" field
            - Embeddings are generated in batches controlled by self._embedding_batch_num
        """
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not data:
            logger.warning("You insert an empty data to vector DB")
            return []

        if self._current_elements + len(data) > self.max_elements:
            raise ValueError(
                f"Cannot insert {len(data)} elements. Current: {self._current_elements}, Max: {self.max_elements}"
            )

        list_data = [
            {
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batch_size = min(self._embedding_batch_num, len(contents))
        embeddings = np.concatenate(
            await asyncio.gather(
                *[
                    self.embedding_func(contents[i : i + batch_size])
                    for i in range(0, len(contents), batch_size)
                ]
            )
        )

        ids = np.fromiter(
            (xxhash.xxh32_intdigest(d["id"].encode()) for d in list_data),
            dtype=np.uint32,
            count=len(list_data),
        )
        self._metadata.update(
            {
                id_int: {
                    k: v for k, v in d.items() if k in self.meta_fields or k == "id"
                }
                for id_int, d in zip(ids, list_data)
            }
        )
        self._index.add_items(data=embeddings, ids=ids, num_threads=self.num_threads)
        self._current_elements = self._index.get_current_count()
        return ids

    async def query(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for the most similar vectors to the given query string.

        This method performs approximate k-nearest neighbor search using the HNSW index.
        It converts the query string to an embedding vector and finds the top-k most
        similar vectors using cosine similarity.

        The search process:
        1. Returns empty list if the index is empty
        2. Adjusts top_k if it exceeds the number of stored elements
        3. Dynamically adjusts ef_search parameter if needed for better recall
        4. Generates embedding for the query string
        5. Performs k-NN search in the HNSW index
        6. Combines results with stored metadata

        Args:
            query: The search query string to find similar vectors for.
            top_k: Maximum number of similar results to return. Default: 5.
                Will be capped at the current number of elements in the index.

        Returns:
            List of dictionaries containing the search results, ordered by similarity
            (most similar first). Each dictionary contains:
            - All metadata fields stored for the vector (including "id")
            - "distance": Cosine distance in range [0, 2] (lower is more similar)
            - "similarity": Cosine similarity in range [0, 1] (higher is more similar)
            Returns empty list if the index is empty.

        Note:
            - Cosine distance and similarity are related: similarity = 1 - distance
            - If top_k exceeds ef_search, ef_search is automatically increased to match
            - The method uses self.num_threads for parallel search operations
        """
        if self._current_elements == 0:
            return []

        top_k = min(top_k, self._current_elements)

        if top_k > self.ef_search:
            logger.warning(
                f"Setting ef_search to {top_k} because top_k is larger than ef_search"
            )
            self._index.set_ef(top_k)

        embedding = await self.embedding_func([query])
        labels, distances = self._index.knn_query(
            data=embedding[0], k=top_k, num_threads=self.num_threads
        )

        return [
            {
                **self._metadata.get(label, {}),
                "distance": distance,
                "similarity": 1 - distance,
            }
            for label, distance in zip(labels[0], distances[0])
        ]

    async def index_done_callback(self):
        """Persist the HNSW index and metadata to disk.

        This callback method is called to save the current state of the index and metadata
        to disk files. It ensures data persistence across sessions by writing both the
        HNSW index structure and the associated metadata dictionary.

        The method performs two save operations:
        1. Saves the HNSW index to a binary file using hnswlib's native format
        2. Saves the metadata dictionary and element count to a pickle file

        The saved files can be loaded later during __post_init__ to restore the exact
        state of the vector storage.

        Note:
            - This is typically called after completing a batch of insertions or updates
            - Both files must exist together for successful restoration
            - The metadata file stores both the metadata dict and the current element count
            - File paths are determined during __post_init__ based on namespace and working_dir
        """
        self._index.save_index(self._index_file_name)
        with open(self._metadata_file_name, "wb") as f:
            pickle.dump((self._metadata, self._current_elements), f)
