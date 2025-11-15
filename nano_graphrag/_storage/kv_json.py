"""
JSON-based Key-Value Storage Implementation

This module provides a simple file-based key-value storage using JSON format.
All data is stored in a single JSON file and loaded into memory for fast access.
"""

import os
from dataclasses import dataclass

from .._utils import load_json, logger, write_json
from ..base import (
    BaseKVStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    """
    Simple JSON file-based key-value storage.

    All data is loaded into memory on initialization and persisted to a JSON file
    when indexing is complete. This implementation is suitable for small to medium
    datasets that fit comfortably in memory.

    The storage file is named 'kv_store_{namespace}.json' in the working directory.
    """
    def __post_init__(self):
        """
        Initialize the JSON storage by loading existing data from file.

        Creates or loads the JSON file for this namespace and loads all data into memory.
        """
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        """Get all keys in the storage."""
        return list(self._data.keys())

    async def index_done_callback(self):
        """
        Callback invoked when indexing is complete.

        Persists all in-memory data to the JSON file.
        """
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        """
        Retrieve a value by its ID.

        Args:
            id: The key to look up

        Returns:
            The value associated with the ID, or None if not found
        """
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        """
        Retrieve multiple values by their IDs.

        Args:
            ids: List of keys to look up
            fields: Optional list of field names to include in results (default: all fields)

        Returns:
            List of values (or None for missing IDs), optionally filtered by fields
        """
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        """
        Filter to return only keys that don't exist in storage.

        Args:
            data: List of keys to check

        Returns:
            Set of keys from data that are not in storage
        """
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        """
        Insert or update key-value pairs.

        Args:
            data: Dictionary of key-value pairs to upsert
        """
        self._data.update(data)

    async def drop(self):
        """
        Delete all data from storage.

        Clears the in-memory dictionary. Changes are persisted on next index_done_callback.
        """
        self._data = {}
