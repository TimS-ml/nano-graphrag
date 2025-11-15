"""
NanoVectorDB Storage Implementation

This module provides a lightweight vector database storage implementation using
NanoVectorDB for nano-graphrag. NanoVectorDB is a simple, file-based vector database
suitable for small to medium-sized datasets.
"""

import asyncio
import os
from dataclasses import dataclass
import numpy as np
from nano_vectordb import NanoVectorDB

from .._utils import logger
from ..base import BaseVectorStorage


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    """
    Vector storage implementation using NanoVectorDB.

    NanoVectorDB is a lightweight, file-based vector database that stores vectors
    in JSON format. This implementation provides async methods for upserting and
    querying vector embeddings with cosine similarity search.

    Attributes:
        cosine_better_than_threshold: Minimum cosine similarity threshold for query results (default: 0.2)
            Higher values return only more similar results
    """
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        """
        Initialize the NanoVectorDB client and configuration.

        Sets up the storage file path, batch size, and similarity threshold from global config.
        """

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        """
        Insert or update vector embeddings in the database.

        Batches the embedding generation for efficiency and upserts all vectors
        along with their metadata.

        Args:
            data: Dictionary mapping IDs to data dictionaries containing 'content' and metadata fields

        Returns:
            list: Results from the upsert operation
        """
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        """
        Query the vector database for similar vectors.

        Embeds the query string and performs cosine similarity search to find
        the most similar vectors in the database.

        Args:
            query: Query string to search for
            top_k: Number of top results to return (default: 5)

        Returns:
            list: List of result dictionaries with 'id', 'distance', and metadata fields
                Results are filtered by cosine_better_than_threshold
        """
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        """
        Callback invoked when indexing is complete.

        Saves the vector database to disk to persist all changes.
        """
        self._client.save()
