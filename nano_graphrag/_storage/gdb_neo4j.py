"""Neo4j Graph Database Storage Implementation.

This module provides a Neo4j-based implementation of the graph storage backend
for nano-graphrag. It handles storing and retrieving graph nodes, edges, and
community structures using Neo4j as the underlying database.

The implementation supports:
    - Asynchronous operations for better performance
    - Batch operations for efficient bulk processing
    - Graph clustering using the Leiden algorithm via GDS library
    - Community schema extraction and management
    - Full CRUD operations on nodes and edges
"""

import json
import asyncio
from collections import defaultdict
from typing import List
from neo4j import AsyncGraphDatabase
from dataclasses import dataclass
from typing import Union
from ..base import BaseGraphStorage, SingleCommunitySchema
from .._utils import logger
from ..prompt import GRAPH_FIELD_SEP

neo4j_lock = asyncio.Lock()


def make_path_idable(path):
    """Convert a file path into a valid identifier string.

    Replaces special characters in a path with underscores to create a valid
    identifier that can be used as a Neo4j label or namespace.

    Args:
        path: The file path string to convert.

    Returns:
        A sanitized string with special characters replaced:
            - "." -> "_"
            - "/" -> "__"
            - "-" -> "_"
            - ":" -> "_"
            - "\\" -> "__"
    """
    return path.replace(".", "_").replace("/", "__").replace("-", "_").replace(":", "_").replace("\\", "__")


@dataclass
class Neo4jStorage(BaseGraphStorage):
    """Neo4j-based graph storage implementation.

    This class provides a complete graph storage backend using Neo4j database.
    It extends BaseGraphStorage and implements all required methods for storing
    and retrieving nodes, edges, and performing graph operations.

    The storage uses a namespace-based approach where each graph is identified
    by a unique label derived from the working directory and namespace configuration.
    This allows multiple graphs to coexist in the same Neo4j database instance.

    Attributes:
        neo4j_url: The connection URL for the Neo4j database.
        neo4j_auth: Authentication credentials tuple (username, password).
        namespace: The unique namespace/label for this graph instance.
        async_driver: The Neo4j async driver instance for database operations.

    Note:
        Requires neo4j_url and neo4j_auth to be provided in global_config['addon_params'].
        The Neo4j Graph Data Science (GDS) library must be installed for clustering operations.
    """

    def __post_init__(self):
        """Initialize Neo4j storage after dataclass initialization.

        Sets up the Neo4j connection parameters, creates a unique namespace identifier,
        and initializes the async database driver.

        Raises:
            ValueError: If neo4j_url or neo4j_auth are not provided in addon_params.

        Note:
            The namespace is created by combining the sanitized working directory path
            with the configured namespace, ensuring unique identification across different
            graph instances.
        """
        self.neo4j_url = self.global_config["addon_params"].get("neo4j_url", None)
        self.neo4j_auth = self.global_config["addon_params"].get("neo4j_auth", None)
        self.namespace = (
            f"{make_path_idable(self.global_config['working_dir'])}__{self.namespace}"
        )
        logger.info(f"Using the label {self.namespace} for Neo4j as identifier")
        if self.neo4j_url is None or self.neo4j_auth is None:
            raise ValueError("Missing neo4j_url or neo4j_auth in addon_params")
        self.async_driver = AsyncGraphDatabase.driver(
            self.neo4j_url, auth=self.neo4j_auth, max_connection_pool_size=50,
        )

    # async def create_database(self):
    #     async with self.async_driver.session() as session:
    #         try:
    #             constraints = await session.run("SHOW CONSTRAINTS")
    #             # TODO I don't know why CREATE CONSTRAINT IF NOT EXISTS still trigger error
    #             # so have to check if the constrain exists
    #             constrain_exists = False

    #             async for record in constraints:
    #                 if (
    #                     self.namespace in record["labelsOrTypes"]
    #                     and "id" in record["properties"]
    #                     and record["type"] == "UNIQUENESS"
    #                 ):
    #                     constrain_exists = True
    #                     break
    #             if not constrain_exists:
    #                 await session.run(
    #                     f"CREATE CONSTRAINT FOR (n:{self.namespace}) REQUIRE n.id IS UNIQUE"
    #                 )
    #                 logger.info(f"Add constraint for namespace: {self.namespace}")

    #         except Exception as e:
    #             logger.error(f"Error accessing or setting up the database: {str(e)}")
    #             raise

    async def _init_workspace(self):
        """Initialize and verify the Neo4j workspace.

        Verifies that the database connection is properly authenticated and
        that connectivity to the Neo4j server can be established.

        Raises:
            Exception: If authentication or connectivity verification fails.

        Note:
            Database creation is currently disabled due to async compatibility issues.
        """
        await self.async_driver.verify_authentication()
        await self.async_driver.verify_connectivity()
        # TODOLater: create database if not exists always cause an error when async
        # await self.create_database()

    async def index_start_callback(self):
        """Callback executed when indexing starts.

        Initializes the Neo4j workspace and creates necessary database indexes
        for optimized query performance. Indexes are created on:
            - id: Node identifier
            - entity_type: Type of entity
            - communityIds: Community membership information
            - source_id: Source document identifier

        Raises:
            Exception: If workspace initialization or index creation fails.

        Note:
            This method is called automatically by the framework at the start
            of the indexing process.
        """
        logger.info("Init Neo4j workspace")
        await self._init_workspace()

        # create index for faster searching
        try:
            async with self.async_driver.session() as session:
                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{self.namespace}`) ON (n.id)"
                )

                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{self.namespace}`) ON (n.entity_type)"
                )

                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{self.namespace}`) ON (n.communityIds)"
                )

                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{self.namespace}`) ON (n.source_id)"
                )
                logger.info("Neo4j indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise e

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            node_id: The unique identifier of the node to check.

        Returns:
            True if the node exists, False otherwise.
        """
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:`{self.namespace}`) WHERE n.id = $node_id RETURN COUNT(n) > 0 AS exists",
                node_id=node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            source_node_id: The unique identifier of the source node.
            target_node_id: The unique identifier of the target node.

        Returns:
            True if an edge exists from source to target, False otherwise.
        """
        async with self.async_driver.session() as session:
            result = await session.run(
                f"""
                MATCH (s:`{self.namespace}`)
                WHERE s.id = $source_id
                MATCH (t:`{self.namespace}`)
                WHERE t.id = $target_id
                RETURN EXISTS((s)-[]->(t)) AS exists
                """,
                source_id=source_node_id,
                target_id=target_node_id,
            )

            record = await result.single()
            return record["exists"] if record else False

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connections) of a node.

        Args:
            node_id: The unique identifier of the node.

        Returns:
            The number of edges connected to the node (both incoming and outgoing).
        """
        results = await self.node_degrees_batch([node_id])
        return results[0] if results else 0

    async def node_degrees_batch(self, node_ids: List[str]) -> List[str]:
        """Get the degrees of multiple nodes in a single batch operation.

        Args:
            node_ids: List of node identifiers to query.

        Returns:
            List of integers representing the degree of each node, in the same
            order as the input node_ids. Returns empty dict if node_ids is empty.

        Note:
            This batch operation is more efficient than calling node_degree()
            multiple times when querying many nodes.
        """
        if not node_ids:
            return {}

        result_dict = {node_id: 0 for node_id in node_ids}
        async with self.async_driver.session() as session:
            result = await session.run(
                f"""
                UNWIND $node_ids AS node_id
                MATCH (n:`{self.namespace}`)
                WHERE n.id = node_id
                OPTIONAL MATCH (n)-[]-(m:`{self.namespace}`)
                RETURN node_id, COUNT(m) AS degree
                """,
                node_ids=node_ids
            )

            async for record in result:
                result_dict[record["node_id"]] = record["degree"]

        return [result_dict[node_id] for node_id in node_ids]

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the combined degree of both nodes in an edge.

        Args:
            src_id: The unique identifier of the source node.
            tgt_id: The unique identifier of the target node.

        Returns:
            The sum of degrees of both the source and target nodes.
        """
        results = await self.edge_degrees_batch([(src_id, tgt_id)])
        return results[0] if results else 0

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        """Get the combined degrees of multiple edges in a single batch operation.

        For each edge pair, calculates the sum of the degrees of both the source
        and target nodes.

        Args:
            edge_pairs: List of tuples, where each tuple contains (source_id, target_id).

        Returns:
            List of integers representing the combined degree for each edge pair,
            in the same order as the input. Returns a list of zeros if an error occurs.

        Note:
            This batch operation is more efficient than calling edge_degree()
            multiple times when querying many edges.
        """
        if not edge_pairs:
            return []

        result_dict = {tuple(edge_pair): 0 for edge_pair in edge_pairs}

        edges_params = [{"src_id": src, "tgt_id": tgt} for src, tgt in edge_pairs]

        try:
            async with self.async_driver.session() as session:
                result = await session.run(
                    f"""
                    UNWIND $edges AS edge

                    MATCH (s:`{self.namespace}`)
                    WHERE s.id = edge.src_id
                    WITH edge, s
                    OPTIONAL MATCH (s)-[]-(n1:`{self.namespace}`)
                    WITH edge, COUNT(n1) AS src_degree

                    MATCH (t:`{self.namespace}`)
                    WHERE t.id = edge.tgt_id
                    WITH edge, src_degree, t
                    OPTIONAL MATCH (t)-[]-(n2:`{self.namespace}`)
                    WITH edge.src_id AS src_id, edge.tgt_id AS tgt_id, src_degree, COUNT(n2) AS tgt_degree

                    RETURN src_id, tgt_id, src_degree + tgt_degree AS degree
                    """,
                    edges=edges_params
                )

                async for record in result:
                    src_id = record["src_id"]
                    tgt_id = record["tgt_id"]
                    degree = record["degree"]

                    # Update result dictionary
                    edge_pair = (src_id, tgt_id)
                    result_dict[edge_pair] = degree

            return [result_dict[tuple(edge_pair)] for edge_pair in edge_pairs]
        except Exception as e:
            logger.error(f"Error in batch edge degree calculation: {e}")
            return [0] * len(edge_pairs)



    async def get_node(self, node_id: str) -> Union[dict, None]:
        """Retrieve a node's data by its identifier.

        Args:
            node_id: The unique identifier of the node to retrieve.

        Returns:
            A dictionary containing the node's properties, or None if the node
            doesn't exist.
        """
        result = await self.get_nodes_batch([node_id])
        return result[0] if result else None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        """Retrieve multiple nodes' data in a single batch operation.

        Args:
            node_ids: List of node identifiers to retrieve.

        Returns:
            List of dictionaries containing node properties, in the same order as
            input node_ids. Returns empty dict if node_ids is empty. Non-existent
            nodes will have None in their position.

        Raises:
            Exception: If an error occurs during batch node retrieval.

        Note:
            The returned node data includes a "clusters" field containing community
            membership information formatted as JSON.
        """
        if not node_ids:
            return {}

        result_dict = {node_id: None for node_id in node_ids}

        try:
            async with self.async_driver.session() as session:
                result = await session.run(
                    f"""
                    UNWIND $node_ids AS node_id
                    MATCH (n:`{self.namespace}`)
                    WHERE n.id = node_id
                    RETURN node_id, properties(n) AS node_data
                    """,
                    node_ids=node_ids
                )

                async for record in result:
                    node_id = record["node_id"]
                    raw_node_data = record["node_data"]

                    if raw_node_data:
                        raw_node_data["clusters"] = json.dumps(
                            [
                                {
                                    "level": index,
                                    "cluster": cluster_id,
                                }
                                for index, cluster_id in enumerate(
                                    raw_node_data.get("communityIds", [])
                                )
                            ]
                        )
                        result_dict[node_id] = raw_node_data
            return [result_dict[node_id] for node_id in node_ids]
        except Exception as e:
            logger.error(f"Error in batch node retrieval: {e}")
            raise e

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """Retrieve an edge's data by its source and target node identifiers.

        Args:
            source_node_id: The unique identifier of the source node.
            target_node_id: The unique identifier of the target node.

        Returns:
            A dictionary containing the edge's properties, or None if the edge
            doesn't exist.
        """
        results = await self.get_edges_batch([(source_node_id, target_node_id)])
        return results[0] if results else None

    async def get_edges_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> list[Union[dict, None]]:
        """Retrieve multiple edges' data in a single batch operation.

        Args:
            edge_pairs: List of tuples, where each tuple contains (source_id, target_id).

        Returns:
            List of dictionaries containing edge properties, in the same order as
            input edge_pairs. Returns empty list if edge_pairs is empty. Non-existent
            edges will have None in their position. On error, returns a list of None
            values with the same length as edge_pairs.

        Note:
            This batch operation is more efficient than calling get_edge()
            multiple times when querying many edges.
        """
        if not edge_pairs:
            return []

        result_dict = {tuple(edge_pair): None for edge_pair in edge_pairs}

        edges_params = [{"source_id": src, "target_id": tgt} for src, tgt in edge_pairs]

        try:
            async with self.async_driver.session() as session:
                result = await session.run(
                    f"""
                    UNWIND $edges AS edge
                    MATCH (s:`{self.namespace}`)-[r]->(t:`{self.namespace}`)
                    WHERE s.id = edge.source_id AND t.id = edge.target_id
                    RETURN edge.source_id AS source_id, edge.target_id AS target_id, properties(r) AS edge_data
                    """,
                    edges=edges_params
                )

                async for record in result:
                    source_id = record["source_id"]
                    target_id = record["target_id"]
                    edge_data = record["edge_data"]

                    edge_pair = (source_id, target_id)
                    result_dict[edge_pair] = edge_data

            return [result_dict[tuple(edge_pair)] for edge_pair in edge_pairs]
        except Exception as e:
            logger.error(f"Error in batch edge retrieval: {e}")
            return [None] * len(edge_pairs)

    async def get_node_edges(
        self, source_node_id: str
    ) -> list[tuple[str, str]]:
        """Get all outgoing edges from a specific node.

        Args:
            source_node_id: The unique identifier of the source node.

        Returns:
            List of tuples representing edges, where each tuple is (source_id, target_id).
            Returns empty list if the node has no outgoing edges.
        """
        results = await self.get_nodes_edges_batch([source_node_id])
        return results[0] if results else []

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> list[list[tuple[str, str]]]:
        """Get all outgoing edges for multiple nodes in a single batch operation.

        Args:
            node_ids: List of node identifiers to query.

        Returns:
            List of lists of edge tuples, in the same order as input node_ids.
            Each inner list contains tuples of (source_id, target_id) for that node's
            outgoing edges. Returns empty list if node_ids is empty. On error, returns
            a list of empty lists with the same length as node_ids.

        Note:
            This batch operation is more efficient than calling get_node_edges()
            multiple times when querying many nodes.
        """
        if not node_ids:
            return []

        result_dict = {node_id: [] for node_id in node_ids}

        try:
            async with self.async_driver.session() as session:
                result = await session.run(
                    f"""
                    UNWIND $node_ids AS node_id
                    MATCH (s:`{self.namespace}`)-[r]->(t:`{self.namespace}`)
                    WHERE s.id = node_id
                    RETURN s.id AS source_id, t.id AS target_id
                    """,
                    node_ids=node_ids
                )

                async for record in result:
                    source_id = record["source_id"]
                    target_id = record["target_id"]

                    if source_id in result_dict:
                        result_dict[source_id].append((source_id, target_id))

            return [result_dict[node_id] for node_id in node_ids]
        except Exception as e:
            logger.error(f"Error in batch node edges retrieval: {e}")
            return [[] for _ in node_ids]

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        """Insert or update a single node in the graph.

        Args:
            node_id: The unique identifier of the node.
            node_data: Dictionary containing the node's properties.

        Note:
            If the node already exists, its properties will be updated.
            If it doesn't exist, a new node will be created.
        """
        await self.upsert_nodes_batch([(node_id, node_data)])

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, str]]]):
        """Insert or update multiple nodes in a single batch operation.

        Nodes are grouped by their entity_type for efficient batch insertion.
        Each node is labeled with both the namespace and its entity type.

        Args:
            nodes_data: List of tuples, where each tuple contains (node_id, node_data).
                       The node_data dictionary should include an "entity_type" key.

        Note:
            Nodes are automatically grouped by entity_type for optimal performance.
            If a node already exists, its properties will be updated using SET +=.
            If entity_type is not provided in node_data, "UNKNOWN" is used as default.
        """
        if not nodes_data:
            return []

        nodes_by_type = {}
        for node_id, node_data in nodes_data:
            node_type = node_data.get("entity_type", "UNKNOWN").strip('"')
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append((node_id, node_data))

        async with self.async_driver.session() as session:
            for node_type, type_nodes in nodes_by_type.items():
                params = [{"id": node_id, "data": node_data} for node_id, node_data in type_nodes]

                await session.run(
                    f"""
                    UNWIND $nodes AS node
                    MERGE (n:`{self.namespace}`:`{node_type}` {{id: node.id}})
                    SET n += node.data
                    """,
                    nodes=params
                )

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        """Insert or update a single edge in the graph.

        Args:
            source_node_id: The unique identifier of the source node.
            target_node_id: The unique identifier of the target node.
            edge_data: Dictionary containing the edge's properties.

        Note:
            Both source and target nodes must already exist in the graph.
            If the edge already exists, its properties will be updated.
            If it doesn't exist, a new edge will be created.
        """
        await self.upsert_edges_batch([(source_node_id, target_node_id, edge_data)])


    async def upsert_edges_batch(
        self, edges_data: list[tuple[str, str, dict[str, str]]]
    ):
        """Insert or update multiple edges in a single batch operation.

        Args:
            edges_data: List of tuples, where each tuple contains
                       (source_id, target_id, edge_data).

        Note:
            All referenced nodes must already exist in the graph.
            If an edge already exists, its properties will be updated using SET +=.
            A default weight of 0.0 is added to edge_data if not provided.
            All edges are created with the relationship type "RELATED".
        """
        if not edges_data:
            return

        edges_params = []
        for source_id, target_id, edge_data in edges_data:
            edge_data_copy = edge_data.copy()
            edge_data_copy.setdefault("weight", 0.0)

            edges_params.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_data": edge_data_copy
            })

        async with self.async_driver.session() as session:
            await session.run(
                f"""
                UNWIND $edges AS edge
                MATCH (s:`{self.namespace}`)
                WHERE s.id = edge.source_id
                WITH edge, s
                MATCH (t:`{self.namespace}`)
                WHERE t.id = edge.target_id
                MERGE (s)-[r:RELATED]->(t)
                SET r += edge.edge_data
                """,
                edges=edges_params
            )
        



    async def clustering(self, algorithm: str):
        """Perform graph clustering using the specified algorithm.

        Args:
            algorithm: The clustering algorithm to use. Currently only "leiden" is supported.

        Raises:
            ValueError: If an unsupported clustering algorithm is specified.

        Note:
            This method requires the Neo4j Graph Data Science (GDS) library to be installed.
            The Leiden algorithm is run with the following parameters:
                - includeIntermediateCommunities: True
                - relationshipWeightProperty: "weight"
                - maxLevels: From global_config['max_graph_cluster_size']
                - tolerance: 0.0001
                - gamma: 1.0
                - theta: 0.01
                - randomSeed: From global_config['graph_cluster_seed']

            The method creates a temporary projected graph, runs clustering, writes
            results to the communityIds property, and then drops the projected graph.
        """
        if algorithm != "leiden":
            raise ValueError(
                f"Clustering algorithm {algorithm} not supported in Neo4j implementation"
            )

        random_seed = self.global_config["graph_cluster_seed"]
        max_level = self.global_config["max_graph_cluster_size"]
        async with self.async_driver.session() as session:
            try:
                # Project the graph with undirected relationships
                await session.run(
                    f"""
                    CALL gds.graph.project(
                        'graph_{self.namespace}',
                        ['{self.namespace}'],
                        {{
                            RELATED: {{
                                orientation: 'UNDIRECTED',
                                properties: ['weight']
                            }}
                        }}
                    )
                    """
                )

                # Run Leiden algorithm
                result = await session.run(
                    f"""
                    CALL gds.leiden.write(
                        'graph_{self.namespace}',
                        {{
                            writeProperty: 'communityIds',
                            includeIntermediateCommunities: True,
                            relationshipWeightProperty: "weight",
                            maxLevels: {max_level},
                            tolerance: 0.0001,
                            gamma: 1.0,
                            theta: 0.01,
                            randomSeed: {random_seed}
                        }}
                    )
                    YIELD communityCount, modularities;
                    """
                )
                result = await result.single()
                community_count: int = result["communityCount"]
                modularities = result["modularities"]
                logger.info(
                    f"Performed graph clustering with {community_count} communities and modularities {modularities}"
                )
            finally:
                # Drop the projected graph
                await session.run(f"CALL gds.graph.drop('graph_{self.namespace}')")

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Extract and build the community schema from clustered graph data.

        Analyzes the graph after clustering to extract community structure information
        including nodes, edges, hierarchical relationships, and source document references.

        Returns:
            Dictionary mapping cluster IDs to community schema information. Each schema
            contains:
                - level: The hierarchical level of the community
                - title: Display title for the community (e.g., "Cluster {id}")
                - nodes: List of node IDs belonging to this community
                - edges: List of edges (as [source, target] pairs) within this community
                - chunk_ids: List of source document IDs associated with this community
                - occurrence: Normalized occurrence score (0.0 to 1.0) based on chunk_ids
                - sub_communities: List of cluster IDs that are sub-communities

        Note:
            This method should be called after clustering() has been executed.
            The occurrence score represents the ratio of this community's chunk_ids
            to the maximum number of chunk_ids across all communities.
            Sub-communities are identified as communities at higher levels whose
            nodes are subsets of the parent community's nodes.
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

        async with self.async_driver.session() as session:
            # Fetch community data
            result = await session.run(
                f"""
                MATCH (n:`{self.namespace}`)
                WITH n, n.communityIds AS communityIds, [(n)-[]-(m:`{self.namespace}`) | m.id] AS connected_nodes
                RETURN n.id AS node_id, n.source_id AS source_id, 
                       communityIds AS cluster_key,
                       connected_nodes
                """
            )

            # records = await result.fetch()

            max_num_ids = 0
            async for record in result:
                for index, c_id in enumerate(record["cluster_key"]):
                    node_id = str(record["node_id"])
                    source_id = record["source_id"]
                    level = index
                    cluster_key = str(c_id)
                    connected_nodes = record["connected_nodes"]

                    results[cluster_key]["level"] = level
                    results[cluster_key]["title"] = f"Cluster {cluster_key}"
                    results[cluster_key]["nodes"].add(node_id)
                    results[cluster_key]["edges"].update(
                        [
                            tuple(sorted([node_id, str(connected)]))
                            for connected in connected_nodes
                            if connected != node_id
                        ]
                    )
                    chunk_ids = source_id.split(GRAPH_FIELD_SEP)
                    results[cluster_key]["chunk_ids"].update(chunk_ids)
                    max_num_ids = max(
                        max_num_ids, len(results[cluster_key]["chunk_ids"])
                    )

            # Process results
            for k, v in results.items():
                v["edges"] = [list(e) for e in v["edges"]]
                v["nodes"] = list(v["nodes"])
                v["chunk_ids"] = list(v["chunk_ids"])
                v["occurrence"] = len(v["chunk_ids"]) / max_num_ids

            # Compute sub-communities (this is a simplified approach)
            for cluster in results.values():
                cluster["sub_communities"] = [
                    sub_key
                    for sub_key, sub_cluster in results.items()
                    if sub_cluster["level"] > cluster["level"]
                    and set(sub_cluster["nodes"]).issubset(set(cluster["nodes"]))
                ]

        return dict(results)

    async def index_done_callback(self):
        """Callback executed when indexing is complete.

        Performs cleanup operations by closing the Neo4j async driver connection.

        Note:
            This method is called automatically by the framework at the end
            of the indexing process.
        """
        await self.async_driver.close()

    async def _debug_delete_all_node_edges(self):
        """Delete all nodes and edges in the current namespace.

        WARNING: This is a destructive operation intended for debugging and testing only.
        It permanently removes all graph data associated with the current namespace.

        The deletion is performed in two steps:
            1. Delete all relationships (edges) in the namespace
            2. Delete all nodes in the namespace

        Raises:
            Exception: If an error occurs during the deletion process.

        Note:
            This method should only be used for debugging, testing, or cleanup purposes.
            Use with extreme caution in production environments.
        """
        async with self.async_driver.session() as session:
            try:
                # Delete all relationships in the namespace
                await session.run(f"MATCH (n:`{self.namespace}`)-[r]-() DELETE r")

                # Delete all nodes in the namespace
                await session.run(f"MATCH (n:`{self.namespace}`) DELETE n")

                logger.info(
                    f"All nodes and edges in namespace '{self.namespace}' have been deleted."
                )
            except Exception as e:
                logger.error(f"Error deleting nodes and edges: {str(e)}")
                raise
