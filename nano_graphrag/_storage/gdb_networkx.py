"""NetworkX-based graph storage implementation for nano-graphrag.

This module provides a graph storage backend using NetworkX, a Python library for
the creation, manipulation, and study of complex networks. The implementation supports
graph persistence via GraphML format, community detection using hierarchical Leiden
clustering, and node embeddings using Node2Vec.

The NetworkXStorage class extends BaseGraphStorage and provides asynchronous methods
for graph operations including node/edge management, clustering, and embedding generation.
"""

import html
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, cast, List
import networkx as nx
import numpy as np
import asyncio

from .._utils import logger
from ..base import (
    BaseGraphStorage,
    SingleCommunitySchema,
)
from ..prompt import GRAPH_FIELD_SEP


@dataclass
class NetworkXStorage(BaseGraphStorage):
    """NetworkX-based implementation of graph storage for nano-graphrag.

    This class provides a complete graph storage solution using NetworkX as the
    underlying graph data structure. It supports persistence via GraphML format,
    community detection algorithms, and node embedding techniques.

    Attributes:
        _graph: The underlying NetworkX graph instance.
        _graphml_xml_file: Path to the GraphML file for persistence.
        _clustering_algorithms: Dictionary mapping algorithm names to clustering methods.
        _node_embed_algorithms: Dictionary mapping algorithm names to embedding methods.
    """
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        """Load a NetworkX graph from a GraphML file.

        Args:
            file_name: Path to the GraphML file to load.

        Returns:
            The loaded NetworkX graph if the file exists, None otherwise.
        """
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        """Write a NetworkX graph to a GraphML file.

        Args:
            graph: The NetworkX graph to save.
            file_name: Path to the GraphML file where the graph will be written.

        Note:
            Logs the number of nodes and edges before writing.
        """
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Extract the largest connected component with stable node and edge ordering.

        This method extracts the largest connected component from the graph and ensures
        that nodes and edges are sorted in a deterministic way. Node names are normalized
        by converting to uppercase, stripping whitespace, and unescaping HTML entities.

        Args:
            graph: The input NetworkX graph.

        Returns:
            The largest connected component as a stabilized NetworkX graph with
            normalized and sorted nodes and edges.

        Note:
            Based on Microsoft's GraphRAG implementation:
            https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Stabilize a graph to ensure deterministic node and edge ordering.

        This method creates a new graph with nodes and edges sorted in a consistent
        manner, ensuring that graphs with the same structure will always have the
        same representation. For undirected graphs, edges are normalized so that
        the source node is always lexicographically less than the target node.

        Args:
            graph: The input NetworkX graph (directed or undirected).

        Returns:
            A new graph with stabilized (sorted) nodes and edges, maintaining the
            same type (directed/undirected) as the input graph.

        Note:
            Based on Microsoft's GraphRAG implementation:
            https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        """Initialize the NetworkX storage after dataclass initialization.

        This method sets up the graph storage by:
        - Determining the GraphML file path based on the working directory and namespace
        - Loading an existing graph from disk if available
        - Initializing an empty graph if no existing graph is found
        - Registering available clustering and embedding algorithms

        Note:
            This is called automatically after the dataclass __init__ method.
        """
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        """Callback invoked when indexing is complete to persist the graph.

        Writes the current graph state to the GraphML file for persistence.
        """
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            node_id: The identifier of the node to check.

        Returns:
            True if the node exists, False otherwise.
        """
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            source_node_id: The identifier of the source node.
            target_node_id: The identifier of the target node.

        Returns:
            True if the edge exists, False otherwise.
        """
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        """Retrieve node data from the graph.

        Args:
            node_id: The identifier of the node to retrieve.

        Returns:
            A dictionary containing the node's attributes if the node exists,
            None otherwise.
        """
        return self._graph.nodes.get(node_id)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        """Retrieve multiple nodes in a single batch operation.

        Args:
            node_ids: List of node identifiers to retrieve.

        Returns:
            Dictionary mapping node IDs to their attribute dictionaries, or None
            for nodes that don't exist.
        """
        return await asyncio.gather(*[self.get_node(node_id) for node_id in node_ids])

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connections) of a node.

        Args:
            node_id: The identifier of the node.

        Returns:
            The degree of the node, or 0 if the node doesn't exist.

        Note:
            Returns 0 for non-existent nodes instead of raising an error.
        """
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def node_degrees_batch(self, node_ids: List[str]) -> List[str]:
        """Get degrees for multiple nodes in a single batch operation.

        Args:
            node_ids: List of node identifiers.

        Returns:
            List of node degrees in the same order as the input node IDs.
        """
        return await asyncio.gather(*[self.node_degree(node_id) for node_id in node_ids])

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Calculate the combined degree of both endpoints of an edge.

        Args:
            src_id: The identifier of the source node.
            tgt_id: The identifier of the target node.

        Returns:
            Sum of the degrees of both nodes, treating non-existent nodes as
            having degree 0.
        """
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        """Get combined degrees for multiple edges in a single batch operation.

        Args:
            edge_pairs: List of tuples, each containing (source_id, target_id).

        Returns:
            List of combined edge degrees in the same order as the input pairs.
        """
        return await asyncio.gather(*[self.edge_degree(src_id, tgt_id) for src_id, tgt_id in edge_pairs])

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """Retrieve edge data from the graph.

        Args:
            source_node_id: The identifier of the source node.
            target_node_id: The identifier of the target node.

        Returns:
            A dictionary containing the edge's attributes if the edge exists,
            None otherwise.
        """
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_edges_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> list[Union[dict, None]]:
        """Retrieve multiple edges in a single batch operation.

        Args:
            edge_pairs: List of tuples, each containing (source_id, target_id).

        Returns:
            List of edge attribute dictionaries, or None for edges that don't exist,
            in the same order as the input pairs.
        """
        return await asyncio.gather(*[self.get_edge(source_node_id, target_node_id) for source_node_id, target_node_id in edge_pairs])

    async def get_node_edges(self, source_node_id: str):
        """Get all edges connected to a specific node.

        Args:
            source_node_id: The identifier of the node.

        Returns:
            List of tuples representing edges connected to the node, where each
            tuple is (source_id, target_id). Returns None if the node doesn't exist.
        """
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> list[list[tuple[str, str]]]:
        """Get edges for multiple nodes in a single batch operation.

        Args:
            node_ids: List of node identifiers.

        Returns:
            List of edge lists, where each edge list contains tuples of
            (source_id, target_id) for the corresponding node. Returns None
            for nodes that don't exist.
        """
        return await asyncio.gather(*[self.get_node_edges(node_id) for node_id
        in node_ids])

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        """Insert or update a node in the graph.

        If the node already exists, its attributes will be updated. If it doesn't
        exist, a new node will be created.

        Args:
            node_id: The identifier of the node.
            node_data: Dictionary of attributes to set on the node.
        """
        self._graph.add_node(node_id, **node_data)

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, str]]]):
        """Insert or update multiple nodes in a single batch operation.

        Args:
            nodes_data: List of tuples, each containing (node_id, node_attributes).
        """
        await asyncio.gather(*[self.upsert_node(node_id, node_data) for node_id, node_data in nodes_data])

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        """Insert or update an edge in the graph.

        If the edge already exists, its attributes will be updated. If it doesn't
        exist, a new edge will be created.

        Args:
            source_node_id: The identifier of the source node.
            target_node_id: The identifier of the target node.
            edge_data: Dictionary of attributes to set on the edge.
        """
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def upsert_edges_batch(
        self, edges_data: list[tuple[str, str, dict[str, str]]]
    ):
        """Insert or update multiple edges in a single batch operation.

        Args:
            edges_data: List of tuples, each containing
                (source_id, target_id, edge_attributes).
        """
        await asyncio.gather(*[self.upsert_edge(source_node_id, target_node_id, edge_data)
                for source_node_id, target_node_id, edge_data in edges_data])
        
    async def clustering(self, algorithm: str):
        """Perform community detection clustering on the graph.

        Args:
            algorithm: Name of the clustering algorithm to use. Currently
                supported algorithms: "leiden".

        Raises:
            ValueError: If the specified algorithm is not supported.

        Note:
            The clustering results are stored as node attributes in the graph.
        """
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Extract community structure and metadata from the clustered graph.

        This method processes the clustering information stored in node attributes
        to build a hierarchical community schema. It computes community membership,
        edges within communities, and relationships between different levels of
        the hierarchy.

        Returns:
            Dictionary mapping cluster IDs to community metadata, where each
            community includes:
            - level: The hierarchical level of the community
            - title: Human-readable title for the community
            - nodes: List of node IDs belonging to this community
            - edges: List of edges within this community
            - chunk_ids: List of source chunk IDs associated with this community
            - occurrence: Normalized occurrence score (0.0 to 1.0)
            - sub_communities: List of child communities at the next level

        Note:
            This method requires that clustering has been performed first, as it
            relies on the "clusters" attribute being present in node data.
        """
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )
        max_num_ids = 0
        levels = defaultdict(set)
        for node_id, node_data in self._graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                results[cluster_key]["level"] = level
                results[cluster_key]["title"] = f"Cluster {cluster_key}"
                results[cluster_key]["nodes"].add(node_id)
                results[cluster_key]["edges"].update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                results[cluster_key]["chunk_ids"].update(
                    node_data["source_id"].split(GRAPH_FIELD_SEP)
                )
                max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                results[comm]["sub_communities"] = [
                    c
                    for c in next_level_comms
                    if results[c]["nodes"].issubset(results[comm]["nodes"])
                ]

        for k, v in results.items():
            v["edges"] = list(v["edges"])
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["chunk_ids"] = list(v["chunk_ids"])
            v["occurrence"] = len(v["chunk_ids"]) / max_num_ids
        return dict(results)

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        """Store clustering results as node attributes in the graph.

        Args:
            cluster_data: Dictionary mapping node IDs to lists of cluster assignments,
                where each assignment contains level and cluster information.

        Note:
            Cluster data is serialized as JSON and stored in the "clusters" node attribute.
        """
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    async def _leiden_clustering(self):
        """Perform hierarchical Leiden community detection on the graph.

        This method applies the hierarchical Leiden algorithm to detect communities
        in the largest connected component of the graph. The algorithm produces a
        multi-level hierarchy of communities, where each level represents a different
        granularity of community structure.

        The clustering uses configuration parameters:
        - max_graph_cluster_size: Maximum size for any single cluster
        - graph_cluster_seed: Random seed for reproducible clustering

        Note:
            The clustering results are stored as node attributes using the
            _cluster_data_to_subgraphs method. Only the largest connected component
            is clustered; isolated nodes and smaller components are not included.
        """
        from graspologic.partition import hierarchical_leiden

        graph = NetworkXStorage.stable_largest_connected_component(self._graph)
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=self.global_config["max_graph_cluster_size"],
            random_seed=self.global_config["graph_cluster_seed"],
        )

        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        self._cluster_data_to_subgraphs(node_communities)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        """Generate vector embeddings for graph nodes.

        This method creates low-dimensional vector representations of nodes based
        on the graph structure, which can be used for similarity search, clustering,
        or other machine learning tasks.

        Args:
            algorithm: Name of the embedding algorithm to use. Currently
                supported algorithms: "node2vec".

        Returns:
            A tuple containing:
            - embeddings: NumPy array of shape (n_nodes, embedding_dim) with node embeddings
            - node_ids: List of node identifiers corresponding to each embedding row

        Raises:
            ValueError: If the specified algorithm is not supported.
        """
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        """Generate node embeddings using the Node2Vec algorithm.

        Node2Vec learns node embeddings by performing biased random walks on the
        graph and then applying skip-gram models (similar to Word2Vec) to learn
        representations that preserve network neighborhoods.

        Returns:
            A tuple containing:
            - embeddings: NumPy array of shape (n_nodes, embedding_dim) with node embeddings
            - node_ids: List of node identifiers from the "id" attribute of each node

        Note:
            Uses parameters from global_config["node2vec_params"] which typically
            includes dimensions, walk length, number of walks, p, and q parameters.
        """
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
