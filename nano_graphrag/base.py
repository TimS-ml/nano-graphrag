"""Base classes and schemas for nano-graphrag storage and retrieval.

This module defines the core abstract base classes and data schemas used throughout
the nano-graphrag library. It includes:

- Query parameter configurations for different search modes (local, global, naive)
- Type definitions for text chunks and community schemas
- Abstract base classes for storage backends (vector, key-value, graph)
- Namespace and callback management for storage operations

The base classes provide interfaces that must be implemented by concrete storage
backends, enabling flexibility in choosing different storage solutions while
maintaining a consistent API.
"""

from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar, List

import numpy as np

from ._utils import EmbeddingFunc


@dataclass
class QueryParam:
    """Configuration parameters for graph-based queries.

    This class encapsulates all configuration options for querying the knowledge graph
    using different search modes (local, global, or naive). It controls token limits,
    search behavior, and response formatting across different query strategies.

    Attributes:
        mode: The search mode to use. Options are:
            - "global": Search across the entire graph using community reports
            - "local": Search within local neighborhoods of relevant entities
            - "naive": Simple text-based search without graph structure
        only_need_context: If True, only return the context without generating a response.
        response_type: The desired format of the response (e.g., "Multiple Paragraphs").
        level: The hierarchical level of communities to use in the search (default: 2).
        top_k: The number of top results to retrieve (default: 20).

        naive_max_token_for_text_unit: Maximum tokens for text units in naive search mode.

        local_max_token_for_text_unit: Maximum tokens for text units in local search (33% of 12000).
        local_max_token_for_local_context: Maximum tokens for local context in local search (40% of 12000).
        local_max_token_for_community_report: Maximum tokens for community reports in local search (27% of 12000).
        local_community_single_one: If True, use only a single community in local search.

        global_min_community_rating: Minimum rating threshold for communities in global search.
        global_max_consider_community: Maximum number of communities to consider in global search.
        global_max_token_for_community_report: Maximum tokens for community reports in global search.
        global_special_community_map_llm_kwargs: Special LLM kwargs for community mapping in global search,
            defaults to JSON object response format.
    """
    mode: Literal["local", "global", "naive"] = "global"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20
    # naive search
    naive_max_token_for_text_unit = 12000
    # local search
    local_max_token_for_text_unit: int = 4000  # 12000 * 0.33
    local_max_token_for_local_context: int = 4800  # 12000 * 0.4
    local_max_token_for_community_report: int = 3200  # 12000 * 0.27
    local_community_single_one: bool = False
    # global search
    global_min_community_rating: float = 0
    global_max_consider_community: float = 512
    global_max_token_for_community_report: int = 16384
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )


TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)
"""Type definition for text chunk data.

A TypedDict representing a single chunk of text from a document.

Fields:
    tokens: The number of tokens in this chunk.
    content: The actual text content of the chunk.
    full_doc_id: The identifier of the full document this chunk belongs to.
    chunk_order_index: The sequential index of this chunk within the document.
"""

SingleCommunitySchema = TypedDict(
    "SingleCommunitySchema",
    {
        "level": int,
        "title": str,
        "edges": list[list[str, str]],
        "nodes": list[str],
        "chunk_ids": list[str],
        "occurrence": float,
        "sub_communities": list[str],
    },
)
"""Type definition for a single community in the hierarchical graph structure.

A TypedDict representing a community (cluster) of related entities and their relationships.

Fields:
    level: The hierarchical level of this community in the graph (higher = more abstract).
    title: A descriptive title for this community.
    edges: List of edges (relationships) in this community, each edge is [source, target].
    nodes: List of node IDs (entities) that belong to this community.
    chunk_ids: List of text chunk IDs associated with this community.
    occurrence: A metric indicating the frequency or importance of this community.
    sub_communities: List of sub-community IDs that are children of this community.
"""


class CommunitySchema(SingleCommunitySchema):
    """Extended community schema with generated reports.

    Extends SingleCommunitySchema by adding generated report content in both
    string and structured JSON formats. These reports provide summaries and
    analyses of the community's content.

    Attributes:
        report_string: A human-readable text summary/report of the community.
        report_json: A structured JSON representation of the community report.
    """
    report_string: str
    report_json: dict


T = TypeVar("T")
"""Generic type variable for type-safe storage operations."""


@dataclass
class StorageNameSpace:
    """Base class for storage namespaces with lifecycle callbacks.

    Provides a namespace-based storage abstraction with callback hooks for
    indexing and querying operations. Subclasses can override callbacks to
    implement custom logic like committing transactions, flushing caches,
    or logging operations.

    Attributes:
        namespace: A unique identifier for this storage namespace, used to
            isolate different storage instances or collections.
        global_config: A dictionary containing global configuration parameters
            that apply to all operations in this namespace.
    """
    namespace: str
    global_config: dict

    async def index_start_callback(self):
        """Callback invoked when indexing operation starts.

        This hook is called at the beginning of an indexing operation and can
        be overridden to perform initialization tasks such as preparing transactions,
        setting up batch operations, or logging the start of indexing.
        """
        pass

    async def index_done_callback(self):
        """Callback invoked when indexing operation completes.

        This hook is called after an indexing operation finishes and can be
        overridden to perform cleanup or commit tasks such as flushing buffers,
        committing transactions, or logging completion status.
        """
        pass

    async def query_done_callback(self):
        """Callback invoked when query operation completes.

        This hook is called after a query operation finishes and can be
        overridden to perform post-query tasks such as logging queries,
        updating statistics, or cleaning up temporary resources.
        """
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    """Abstract base class for vector storage with semantic search capabilities.

    Provides an interface for storing and querying vector embeddings with semantic
    similarity search. Implementations must provide methods for upserting vectors
    and querying for nearest neighbors.

    Attributes:
        embedding_func: A function that converts text content into vector embeddings.
            If None, embeddings must be provided directly in the data.
        meta_fields: A set of metadata field names to store alongside vectors.
            These fields are preserved and returned with query results.
    """
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        """Query for the most semantically similar vectors.

        Converts the query string to an embedding and retrieves the top-k most
        similar vectors from storage using similarity metrics (e.g., cosine similarity).

        Args:
            query: The text query to search for.
            top_k: The number of most similar results to return.

        Returns:
            A list of dictionaries containing the matching documents and their
            metadata. Each dictionary typically includes the content, similarity
            score, and any meta_fields specified in the storage configuration.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Insert or update vectors in the storage.

        Takes a dictionary of data where keys are document IDs and values contain
        the content and/or embeddings. If embedding_func is provided, the 'content'
        field will be automatically embedded. Otherwise, an 'embedding' field must
        be present in the value dictionary.

        Args:
            data: A dictionary mapping document IDs to their data. Each value dict
                should contain either:
                - 'content': Text to be embedded (if embedding_func is set), or
                - 'embedding': Pre-computed vector embedding (if embedding_func is None)
                Additional metadata fields specified in meta_fields will also be stored.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    """Abstract base class for key-value storage operations.

    Provides a generic interface for storing and retrieving typed data using
    key-value semantics. The type parameter T specifies the type of values
    stored, enabling type-safe operations.

    Type Parameters:
        T: The type of values stored in this key-value store.
    """

    async def all_keys(self) -> list[str]:
        """Retrieve all keys present in the storage.

        Returns:
            A list of all key strings currently stored in the namespace.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        """Retrieve a single value by its key.

        Args:
            id: The key identifier to look up.

        Returns:
            The value associated with the key, or None if the key doesn't exist.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        """Retrieve multiple values by their keys.

        Args:
            ids: A list of key identifiers to retrieve.
            fields: Optional set of field names to retrieve. If None, retrieves
                all fields. If provided, only the specified fields are returned
                in each value.

        Returns:
            A list of values corresponding to the input keys. Returns None for
            keys that don't exist, maintaining the same order as the input ids.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """Filter out keys that don't exist in storage.

        Checks which keys from the input list are not present in the storage.

        Args:
            data: A list of key identifiers to check.

        Returns:
            A set of keys from the input that do NOT exist in the storage.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        """Insert or update multiple key-value pairs.

        If a key already exists, its value is updated. If a key is new,
        it is inserted into the storage.

        Args:
            data: A dictionary mapping keys to their values to be stored.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def drop(self):
        """Delete all data in this storage namespace.

        Removes all key-value pairs from the storage namespace. This operation
        is irreversible and should be used with caution.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    """Abstract base class for graph storage operations.

    Provides an interface for storing and querying graph-structured data consisting
    of nodes (entities) and edges (relationships). Supports both individual and
    batch operations for efficient graph manipulation and analysis.
    """

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            node_id: The unique identifier of the node to check.

        Returns:
            True if the node exists, False otherwise.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            source_node_id: The identifier of the source (start) node.
            target_node_id: The identifier of the target (end) node.

        Returns:
            True if an edge exists from source to target, False otherwise.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connections) of a node.

        Args:
            node_id: The unique identifier of the node.

        Returns:
            The number of edges connected to this node.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def node_degrees_batch(self, node_ids: List[str]) -> List[str]:
        """Get the degrees of multiple nodes in batch.

        Args:
            node_ids: A list of node identifiers.

        Returns:
            A list of degrees corresponding to each node in the input list.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the degree (weight or count) of a specific edge.

        Args:
            src_id: The identifier of the source node.
            tgt_id: The identifier of the target node.

        Returns:
            The degree or weight of the edge between the two nodes.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        """Get the degrees of multiple edges in batch.

        Args:
            edge_pairs: A list of tuples, where each tuple contains
                (source_node_id, target_node_id).

        Returns:
            A list of edge degrees corresponding to each edge pair in the input.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        """Retrieve a node's data by its identifier.

        Args:
            node_id: The unique identifier of the node to retrieve.

        Returns:
            A dictionary containing the node's data and attributes, or None if
            the node doesn't exist.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        """Retrieve multiple nodes' data in batch.

        Args:
            node_ids: A list of node identifiers to retrieve.

        Returns:
            A dictionary mapping node IDs to their data. Returns None for nodes
            that don't exist.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """Retrieve an edge's data between two nodes.

        Args:
            source_node_id: The identifier of the source node.
            target_node_id: The identifier of the target node.

        Returns:
            A dictionary containing the edge's data and attributes, or None if
            the edge doesn't exist.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_edges_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> list[Union[dict, None]]:
        """Retrieve multiple edges' data in batch.

        Args:
            edge_pairs: A list of tuples, where each tuple contains
                (source_node_id, target_node_id).

        Returns:
            A list of dictionaries containing edge data, with None for edges
            that don't exist. Order matches the input edge_pairs.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        """Retrieve all edges connected to a specific node.

        Args:
            source_node_id: The identifier of the node.

        Returns:
            A list of tuples representing edges, where each tuple contains
            (source_node_id, target_node_id). Returns None if the node doesn't exist.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> list[list[tuple[str, str]]]:
        """Retrieve all edges for multiple nodes in batch.

        Args:
            node_ids: A list of node identifiers.

        Returns:
            A list of edge lists, where each inner list contains tuples of
            (source_node_id, target_node_id) for the corresponding node.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        """Insert or update a node in the graph.

        If the node already exists, its data is updated. If it's new, the node
        is created.

        Args:
            node_id: The unique identifier for the node.
            node_data: A dictionary containing the node's attributes and data.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, str]]]):
        """Insert or update multiple nodes in batch.

        Args:
            nodes_data: A list of tuples, where each tuple contains
                (node_id, node_data_dict).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        """Insert or update an edge between two nodes.

        If the edge already exists, its data is updated. If it's new, the edge
        is created.

        Args:
            source_node_id: The identifier of the source node.
            target_node_id: The identifier of the target node.
            edge_data: A dictionary containing the edge's attributes and data.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def upsert_edges_batch(
        self, edges_data: list[tuple[str, str, dict[str, str]]]
    ):
        """Insert or update multiple edges in batch.

        Args:
            edges_data: A list of tuples, where each tuple contains
                (source_node_id, target_node_id, edge_data_dict).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        """Perform community detection/clustering on the graph.

        Applies a clustering algorithm to identify communities (densely connected
        groups of nodes) within the graph structure.

        Args:
            algorithm: The name of the clustering algorithm to use (e.g., "leiden").

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Retrieve the hierarchical community structure of the graph.

        Returns the complete community representation including communities at
        different hierarchical levels, their member nodes, edges, and relationships.

        Returns:
            A dictionary mapping community IDs to their SingleCommunitySchema data,
            which includes level, title, edges, nodes, chunk_ids, occurrence,
            and sub_communities.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        """Generate embeddings for nodes in the graph.

        Note:
            This feature is not currently used in nano-graphrag and will raise
            NotImplementedError.

        Args:
            algorithm: The name of the embedding algorithm to use.

        Returns:
            A tuple containing:
                - A numpy array of node embeddings
                - A list of node IDs corresponding to the embeddings

        Raises:
            NotImplementedError: Node embedding is not used in nano-graphrag.
        """
        raise NotImplementedError("Node embedding is not used in nano-graphrag.")
