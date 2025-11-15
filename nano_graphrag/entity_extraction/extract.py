"""Entity and relationship extraction module for nano-graphrag.

This module provides functionality for extracting entities and relationships from text chunks
using DSPy-based entity relationship extraction. It supports two main workflows:
1. Generating training datasets from text chunks for model fine-tuning
2. Extracting entities and relationships to populate a knowledge graph

The module uses asynchronous processing to efficiently handle multiple text chunks in parallel
and integrates with graph storage and vector storage backends for persistence.
"""
from typing import Union
import pickle
import asyncio
from openai import BadRequestError
from collections import defaultdict
import dspy
from nano_graphrag.base import (
    BaseGraphStorage,
    BaseVectorStorage,
    TextChunkSchema,
)
from nano_graphrag.prompt import PROMPTS
from nano_graphrag._utils import logger, compute_mdhash_id
from nano_graphrag.entity_extraction.module import TypedEntityRelationshipExtractor
from nano_graphrag._op import _merge_edges_then_upsert, _merge_nodes_then_upsert


async def generate_dataset(
    chunks: dict[str, TextChunkSchema],
    filepath: str,
    save_dataset: bool = True,
    global_config: dict = {},
) -> list[dspy.Example]:
    """Generate a training dataset of extracted entities and relationships from text chunks.

    This function processes text chunks to extract entities and relationships, creating
    a dataset of DSPy examples that can be used for training or fine-tuning entity
    relationship extraction models. Each example contains the input text and the
    extracted entities and relationships.

    Args:
        chunks: Dictionary mapping chunk IDs to TextChunkSchema objects containing
            the text content to process.
        filepath: Path where the dataset should be saved (as a pickle file).
        save_dataset: If True, saves the filtered examples to the specified filepath.
            Defaults to True.
        global_config: Configuration dictionary that may contain:
            - use_compiled_dspy_entity_relationship: Whether to use a pre-compiled model
            - entity_relationship_module_path: Path to the compiled model to load

    Returns:
        A list of DSPy Example objects containing the extracted entities and relationships.
        Examples with no entities or relationships are filtered out.

    Note:
        - The function processes chunks asynchronously in parallel for efficiency
        - Progress is printed to stdout showing number of chunks, entities, and relations processed
        - BadRequestError exceptions during extraction are logged and result in empty extractions
        - The returned examples only include chunks that yielded at least one entity and relationship
    """
    entity_extractor = TypedEntityRelationshipExtractor(num_refine_turns=1, self_refine=True)

    if global_config.get("use_compiled_dspy_entity_relationship", False):
        entity_extractor.load(global_config["entity_relationship_module_path"])

    ordered_chunks = list(chunks.items())
    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(
        chunk_key_dp: tuple[str, TextChunkSchema]
    ) -> dspy.Example:
        """Process a single text chunk to extract entities and relationships.

        Args:
            chunk_key_dp: Tuple containing the chunk key and TextChunkSchema object.

        Returns:
            A DSPy Example object with the input text and extracted entities/relationships.

        Note:
            Updates the nonlocal counters for already_processed, already_entities,
            and already_relations, and prints progress to stdout.
        """
        nonlocal already_processed, already_entities, already_relations
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        try:
            prediction = await asyncio.to_thread(entity_extractor, input_text=content)
            entities, relationships = prediction.entities, prediction.relationships
        except BadRequestError as e:
            logger.error(f"Error in TypedEntityRelationshipExtractor: {e}")
            entities, relationships = [], []
        example = dspy.Example(
            input_text=content, entities=entities, relationships=relationships
        ).with_inputs("input_text")
        already_entities += len(entities)
        already_relations += len(relationships)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return example

    examples = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    filtered_examples = [
        example
        for example in examples
        if len(example.entities) > 0 and len(example.relationships) > 0
    ]
    num_filtered_examples = len(examples) - len(filtered_examples)
    if save_dataset:
        with open(filepath, "wb") as f:
            pickle.dump(filtered_examples, f)
            logger.info(
                f"Saved {len(filtered_examples)} examples with keys: {filtered_examples[0].keys()}, filtered {num_filtered_examples} examples"
            )

    return filtered_examples


async def extract_entities_dspy(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """Extract entities and relationships from text chunks and populate a knowledge graph.

    This function processes text chunks to extract entities and their relationships,
    then merges and stores them in a graph database and optionally in a vector database
    for semantic search. The function handles deduplication and merging of entities
    and relationships that appear across multiple chunks.

    Args:
        chunks: Dictionary mapping chunk IDs to TextChunkSchema objects containing
            the text content to process.
        knwoledge_graph_inst: Graph storage instance where nodes (entities) and edges
            (relationships) will be stored.
        entity_vdb: Vector database instance for storing entity embeddings. If None,
            entities will not be stored in a vector database.
        global_config: Configuration dictionary that may contain:
            - use_compiled_dspy_entity_relationship: Whether to use a pre-compiled model
            - entity_relationship_module_path: Path to the compiled model to load

    Returns:
        The updated BaseGraphStorage instance if entities were successfully extracted,
        or None if no entities were extracted (which may indicate LLM issues).

    Note:
        - Processes chunks asynchronously in parallel for efficiency
        - Automatically merges duplicate entities and relationships from different chunks
        - Progress is printed to stdout showing processed chunks, entities, and relations
        - Each entity is stored in the vector DB with combined entity name and description
        - BadRequestError exceptions during extraction are logged and result in empty extractions
        - Entities are indexed in the vector DB using MD5 hash IDs with 'ent-' prefix
    """
    entity_extractor = TypedEntityRelationshipExtractor(num_refine_turns=1, self_refine=True)

    if global_config.get("use_compiled_dspy_entity_relationship", False):
        entity_extractor.load(global_config["entity_relationship_module_path"])

    ordered_chunks = list(chunks.items())
    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single text chunk and organize extracted entities and relationships.

        Args:
            chunk_key_dp: Tuple containing the chunk key (str) and TextChunkSchema object.

        Returns:
            A tuple of two dictionaries:
            - First dict maps entity names to lists of entity data dictionaries
            - Second dict maps (source_id, target_id) tuples to lists of relationship data dictionaries

        Note:
            Updates the nonlocal counters for already_processed, already_entities,
            and already_relations, and prints progress to stdout. Each extracted entity
            and relationship is tagged with the source chunk key.
        """
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        try:
            prediction = await asyncio.to_thread(entity_extractor, input_text=content)
            entities, relationships = prediction.entities, prediction.relationships
        except BadRequestError as e:
            logger.error(f"Error in TypedEntityRelationshipExtractor: {e}")
            entities, relationships = [], []

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        for entity in entities:
            entity["source_id"] = chunk_key
            maybe_nodes[entity["entity_name"]].append(entity)
            already_entities += 1

        for relationship in relationships:
            relationship["source_id"] = chunk_key
            maybe_edges[(relationship["src_id"], relationship["tgt_id"])].append(
                relationship
            )
            already_relations += 1

        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knwoledge_graph_inst
