"""
Core Operations Module for nano-graphrag

This module provides the core operations for the GraphRAG (Graph-based Retrieval-Augmented Generation) pipeline,
including:
- Text chunking and tokenization
- Entity and relationship extraction from text
- Community detection and report generation
- Query operations (local, global, and naive search)
- Context building for RAG queries

The module implements the main algorithmic components that power the nano-graphrag system,
handling everything from document preprocessing to intelligent query answering.
"""

import re
import json
import asyncio
from typing import Union
from collections import Counter, defaultdict
from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,

    TokenizerWrapper
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tokenizer_wrapper: TokenizerWrapper,
    overlap_token_size=128,
    max_token_size=1024,
):
    """
    Split documents into overlapping chunks based on token size.

    This function implements a simple sliding window approach to chunk tokenized documents.
    Each chunk has a maximum size and overlaps with adjacent chunks for better context continuity.

    Args:
        tokens_list: List of token sequences, where each sequence represents a document
        doc_keys: List of document identifiers corresponding to tokens_list
        tokenizer_wrapper: TokenizerWrapper instance for encoding/decoding tokens
        overlap_token_size: Number of tokens to overlap between consecutive chunks (default: 128)
        max_token_size: Maximum number of tokens per chunk (default: 1024)

    Returns:
        list[dict]: List of chunk dictionaries, each containing:
            - tokens: Number of tokens in the chunk
            - content: Decoded text content of the chunk
            - chunk_order_index: Sequential index of the chunk within its document
            - full_doc_id: Identifier of the source document
    """
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))


        chunk_texts = tokenizer_wrapper.decode_batch(chunk_token)

        for i, chunk in enumerate(chunk_texts):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )
    return results


def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tokenizer_wrapper: TokenizerWrapper,
    overlap_token_size=128,
    max_token_size=1024,
):
    """
    Split documents into chunks using semantic separators (like newlines, periods, etc.).

    This approach is more intelligent than simple token-based chunking, as it attempts to
    preserve semantic boundaries by splitting on natural text separators.

    Args:
        tokens_list: List of token sequences, where each sequence represents a document
        doc_keys: List of document identifiers corresponding to tokens_list
        tokenizer_wrapper: TokenizerWrapper instance for encoding/decoding tokens
        overlap_token_size: Number of tokens to overlap between consecutive chunks (default: 128)
        max_token_size: Maximum number of tokens per chunk (default: 1024)

    Returns:
        list[dict]: List of chunk dictionaries, each containing:
            - tokens: Number of tokens in the chunk
            - content: Decoded text content of the chunk
            - chunk_order_index: Sequential index of the chunk within its document
            - full_doc_id: Identifier of the source document
    """
    from .prompt import PROMPTS
    # NOTE: Directly use wrapper encoding instead of accessing underlying tokenizer
    separators = [tokenizer_wrapper.encode(s) for s in PROMPTS["default_text_separator"]]
    splitter = SeparatorSplitter(
        separators=separators,
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_tokens = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_tokens]

        decoded_chunks = tokenizer_wrapper.decode_batch(chunk_tokens)
        for i, chunk in enumerate(decoded_chunks):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )
    return results


def get_chunks(new_docs, chunk_func=chunking_by_token_size, tokenizer_wrapper: TokenizerWrapper = None, **chunk_func_params):
    """
    Process new documents and generate text chunks using the specified chunking function.

    This is a high-level function that coordinates the chunking process, handling tokenization
    and generating unique IDs for each chunk.

    Args:
        new_docs: Dictionary mapping document IDs to document data (containing 'content' field)
        chunk_func: Function to use for chunking (default: chunking_by_token_size)
        tokenizer_wrapper: TokenizerWrapper instance for tokenization
        **chunk_func_params: Additional parameters to pass to the chunking function

    Returns:
        dict: Dictionary mapping chunk IDs (MD5 hash) to chunk data dictionaries
    """
    inserting_chunks = {}
    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    tokens = [tokenizer_wrapper.encode(doc) for doc in docs]
    chunks = chunk_func(
        tokens, doc_keys=doc_keys, tokenizer_wrapper=tokenizer_wrapper, overlap_token_size=chunk_func_params.get("overlap_token_size", 128), max_token_size=chunk_func_params.get("max_token_size", 1024)
    )
    for chunk in chunks:
        inserting_chunks.update(
            {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
        )
    return inserting_chunks


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
    tokenizer_wrapper: TokenizerWrapper,
) -> str:
    """
    Summarize entity or relationship descriptions if they exceed the maximum token limit.

    This function checks if descriptions are too long and uses an LLM to generate
    a concise summary when necessary, helping to keep the knowledge graph manageable.

    Args:
        entity_or_relation_name: Name of the entity or relationship being summarized
        description: Full description text to potentially summarize
        global_config: Global configuration dictionary containing LLM settings
        tokenizer_wrapper: TokenizerWrapper instance for token counting

    Returns:
        str: Either the original description (if short enough) or an LLM-generated summary
    """
    use_llm_func: callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]


    tokens = tokenizer_wrapper.encode(description)
    if len(tokens) < summary_max_tokens:
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]

    use_description = tokenizer_wrapper.decode(tokens[:llm_max_tokens])
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """
    Parse and validate a single entity extraction record.

    Processes raw extraction attributes and validates them to create an entity record
    suitable for adding to the knowledge graph.

    Args:
        record_attributes: List of attribute strings from the LLM extraction
            Expected format: ["entity", name, type, description, ...]
        chunk_key: Identifier of the source text chunk

    Returns:
        dict or None: Entity dictionary with keys (entity_name, entity_type, description, source_id)
            Returns None if the record is invalid
    """
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # Add this record as a node in the graph
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """
    Parse and validate a single relationship extraction record.

    Processes raw extraction attributes to create a relationship/edge record
    for the knowledge graph.

    Args:
        record_attributes: List of attribute strings from the LLM extraction
            Expected format: ["relationship", source, target, description, ..., weight]
        chunk_key: Identifier of the source text chunk

    Returns:
        dict or None: Relationship dictionary with keys (src_id, tgt_id, weight, description, source_id)
            Returns None if the record is invalid
    """
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # Add this record as an edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    tokenizer_wrapper,
):
    """
    Merge multiple node records for the same entity and upsert to the knowledge graph.

    When the same entity appears in multiple chunks, this function consolidates
    all the information (types, descriptions, source IDs) and updates the graph.

    Args:
        entity_name: Name of the entity to merge and upsert
        nodes_data: List of node data dictionaries to merge
        knwoledge_graph_inst: Knowledge graph storage instance
        global_config: Global configuration dictionary
        tokenizer_wrapper: TokenizerWrapper instance for summarization

    Returns:
        dict: Merged node data with entity_name included
    """
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config, tokenizer_wrapper
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    tokenizer_wrapper,
):
    """
    Merge multiple edge records for the same relationship and upsert to the knowledge graph.

    When the same relationship appears in multiple chunks, this function consolidates
    all the information (weights, descriptions, source IDs) and updates the graph.

    Args:
        src_id: Source entity ID
        tgt_id: Target entity ID
        edges_data: List of edge data dictionaries to merge
        knwoledge_graph_inst: Knowledge graph storage instance
        global_config: Global configuration dictionary
        tokenizer_wrapper: TokenizerWrapper instance for summarization

    Returns:
        None
    """
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config, tokenizer_wrapper
    )
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    tokenizer_wrapper,
    global_config: dict,
    using_amazon_bedrock: bool=False,
) -> Union[BaseGraphStorage, None]:
    """
    Extract entities and relationships from text chunks using LLM-based extraction.

    This is the main entity extraction function that processes text chunks, uses an LLM
    to identify entities and relationships, and populates the knowledge graph. It implements
    a "gleaning" approach where the LLM is prompted multiple times to extract additional
    information.

    Args:
        chunks: Dictionary mapping chunk IDs to TextChunkSchema objects
        knwoledge_graph_inst: Knowledge graph storage instance to populate
        entity_vdb: Vector database for storing entity embeddings
        tokenizer_wrapper: TokenizerWrapper instance for tokenization
        global_config: Global configuration dictionary with LLM settings
        using_amazon_bedrock: Whether Amazon Bedrock is being used (affects message formatting)

    Returns:
        BaseGraphStorage or None: Updated knowledge graph instance, or None if no entities extracted
    """
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)
        if isinstance(final_result, list):
            final_result = final_result[0]["text"]

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result, using_amazon_bedrock)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result, using_amazon_bedrock)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config, tokenizer_wrapper)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config, tokenizer_wrapper)
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


def _pack_single_community_by_sub_communities(
    community: SingleCommunitySchema,
    max_token_size: int,
    already_reports: dict[str, CommunitySchema],
    tokenizer_wrapper: TokenizerWrapper,
) -> tuple[str, int, set, set]:
    """
    Package sub-community reports into a CSV format description.

    For hierarchical community structures, this function retrieves and formats
    sub-community reports, helping to provide context for higher-level communities.

    Args:
        community: Community schema containing sub_communities references
        max_token_size: Maximum tokens allowed for sub-community descriptions
        already_reports: Dictionary of already generated community reports
        tokenizer_wrapper: TokenizerWrapper for token counting

    Returns:
        tuple: (sub_communities_csv, token_count, included_nodes_set, included_edges_set)
    """ 
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
        tokenizer_wrapper=tokenizer_wrapper,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])
    

    return (
        sub_communities_describe,
        len(tokenizer_wrapper.encode(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    tokenizer_wrapper: "TokenizerWrapper",
    max_token_size: int = 12000,
    already_reports: dict[str, CommunitySchema] = {},
    global_config: dict = {},
) -> str:
    """
    Generate a comprehensive description of a single community for report generation.

    This function packages all relevant information about a community (nodes, edges,
    sub-community reports) into a formatted string suitable for LLM processing.
    It implements intelligent token budget management to fit within size constraints.

    Args:
        knwoledge_graph_inst: Knowledge graph storage instance
        community: Community schema to describe
        tokenizer_wrapper: TokenizerWrapper for token counting
        max_token_size: Maximum tokens allowed in the output (default: 12000)
        already_reports: Dictionary of already generated community reports
        global_config: Global configuration dictionary

    Returns:
        str: Formatted community description with reports, entities, and relationships in CSV format
    """

    # 1. Prepare raw data
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )


    # 2. 定义模板和固定开销
    final_template = """-----Reports-----
```csv
{reports}
```
-----Entities-----
```csv
{entities}
```
-----Relationships-----
```csv
{relationships}
```"""
    base_template_tokens = len(tokenizer_wrapper.encode(
        final_template.format(reports="", entities="", relationships="")
    ))
    remaining_budget = max_token_size - base_template_tokens

    # 3. 处理子社区报告
    report_describe = ""
    contain_nodes = set()
    contain_edges = set()
    
    # 启发式截断检测
    truncated = len(nodes_in_order) > 100 or len(edges_in_order) > 100
    
    need_to_use_sub_communities = (
        truncated and 
        community["sub_communities"] and 
        already_reports
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(f"Community {community['title']} using sub-communities")
        # 获取子社区报告及包含的节点/边
        result = _pack_single_community_by_sub_communities(
            community, remaining_budget, already_reports, tokenizer_wrapper
        )
        report_describe, report_size, contain_nodes, contain_edges = result
        remaining_budget = max(0, remaining_budget - report_size)

    # 4. 准备节点和边数据（过滤子社区已包含的）
    def format_row(row: list) -> str:
        return ','.join('"{}"'.format(str(item).replace('"', '""')) for item in row)

    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]

    # 获取度数并创建数据结构
    node_degrees = await knwoledge_graph_inst.node_degrees_batch(nodes_in_order)
    edge_degrees = await knwoledge_graph_inst.edge_degrees_batch(edges_in_order)
    
    # 过滤已存在于子社区的节点/边
    nodes_list_data = [
        [i, name, data.get("entity_type", "UNKNOWN"), 
         data.get("description", "UNKNOWN"), node_degrees[i]]
        for i, (name, data) in enumerate(zip(nodes_in_order, nodes_data))
        if name not in contain_nodes  # 关键过滤
    ]
    
    edges_list_data = [
        [i, edge[0], edge[1], data.get("description", "UNKNOWN"), edge_degrees[i]]
        for i, (edge, data) in enumerate(zip(edges_in_order, edges_data))
        if (edge[0], edge[1]) not in contain_edges  # 关键过滤
    ]
    
    # 按重要性排序
    nodes_list_data.sort(key=lambda x: x[-1], reverse=True)
    edges_list_data.sort(key=lambda x: x[-1], reverse=True)

    # 5. 动态分配预算
    # 计算表头开销
    header_tokens = len(tokenizer_wrapper.encode(
        list_of_list_to_csv([node_fields]) + "\n" + list_of_list_to_csv([edge_fields])
    ))



    data_budget = max(0, remaining_budget - header_tokens)
    total_items = len(nodes_list_data) + len(edges_list_data)
    node_ratio = len(nodes_list_data) / max(1, total_items)
    edge_ratio = 1 - node_ratio




    # 执行截断
    nodes_final = truncate_list_by_token_size(
        nodes_list_data, key=format_row, 
        max_token_size=int(data_budget * node_ratio), 
        tokenizer_wrapper=tokenizer_wrapper
    )
    edges_final = truncate_list_by_token_size(
        edges_list_data, key=format_row,
        max_token_size= int(data_budget * edge_ratio),
        tokenizer_wrapper=tokenizer_wrapper
    )

    # 6. 组装最终输出
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_final)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_final)



    final_output = final_template.format(
        reports=report_describe,
        entities=nodes_describe,
        relationships=edges_describe
    )

    return final_output


def _community_report_json_to_str(parsed_output: dict) -> str:
    """
    Convert community report JSON to formatted markdown string.

    Reference: Official GraphRAG implementation (index/graph/extractors/community_reports)

    Args:
        parsed_output: Dictionary containing title, summary, and findings

    Returns:
        str: Formatted markdown report
    """
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knwoledge_graph_inst: BaseGraphStorage,
    tokenizer_wrapper: TokenizerWrapper,
    global_config: dict,
):
    """
    Generate descriptive reports for all communities in the knowledge graph.

    This function processes communities hierarchically (from highest to lowest level),
    generating LLM-based summaries that describe the entities, relationships, and
    themes within each community.

    Args:
        community_report_kv: Key-value storage for persisting community reports
        knwoledge_graph_inst: Knowledge graph containing the community structure
        tokenizer_wrapper: TokenizerWrapper for token management
        global_config: Global configuration with LLM settings

    Returns:
        None (results are stored in community_report_kv)
    """
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    use_string_json_convert_func: callable = global_config["convert_response_to_json_func"]

    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = list(communities_schema.keys()), list(communities_schema.values())
    already_processed = 0

    prompt_template = PROMPTS["community_report"]

    prompt_overhead = len(tokenizer_wrapper.encode(prompt_template.format(input_text="")))

    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            tokenizer_wrapper=tokenizer_wrapper, 
            max_token_size=global_config["best_model_max_token_size"] - prompt_overhead -200, # extra token for chat template and prompt template
            already_reports=already_reports,
            global_config=global_config,
        )
        prompt = prompt_template.format(input_text=describe)


        response = await use_llm_func(prompt, **llm_extra_kwargs)
        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(f"{now_ticks} Processed {already_processed} communities\r", end="", flush=True)
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    print()  # clear the progress bar
    await community_report_kv.upsert(community_datas)


async def _find_most_related_community_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    community_reports: BaseKVStorage[CommunitySchema],
    tokenizer_wrapper,
):
    related_communities = []
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        related_communities.extend(json.loads(node_d["clusters"]))
    related_community_dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= query_param.level
    ]
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    _related_community_datas = await asyncio.gather(
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    related_community_keys = sorted(
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    sorted_community_datas = [
        related_community_datas[k] for k in related_community_keys
    ]

    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.local_max_token_for_community_report,
        tokenizer_wrapper=tokenizer_wrapper, 
    )
    if query_param.local_community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    tokenizer_wrapper,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await knowledge_graph_inst.get_nodes_edges_batch([dp["entity_name"] for dp in node_datas])
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await knowledge_graph_inst.get_nodes_batch(all_one_hop_nodes)
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.local_max_token_for_text_unit,
        tokenizer_wrapper=tokenizer_wrapper, # 传入 wrapper
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    tokenizer_wrapper,
):
    all_related_edges = await knowledge_graph_inst.get_nodes_edges_batch([dp["entity_name"] for dp in node_datas])
    
    all_edges = []
    seen = set()
    
    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge) 
                
    all_edges_pack = await knowledge_graph_inst.get_edges_batch(all_edges)
    all_edges_degree = await knowledge_graph_inst.edge_degrees_batch(all_edges)
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.local_max_token_for_local_context,
        tokenizer_wrapper=tokenizer_wrapper, 
    )
    return all_edges_data


async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return None
    node_datas = await knowledge_graph_inst.get_nodes_batch([r["entity_name"] for r in results])
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await knowledge_graph_inst.node_degrees_batch([r["entity_name"] for r in results])
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports, tokenizer_wrapper
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst, tokenizer_wrapper
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst, tokenizer_wrapper
    )
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
) -> str:
    """
    Perform a local knowledge graph query.

    Local search finds entities related to the query and retrieves their immediate
    context including connected entities, relationships, relevant text chunks, and
    community reports. This is best for specific questions about particular entities.

    Args:
        query: User query string
        knowledge_graph_inst: Knowledge graph storage instance
        entities_vdb: Vector database containing entity embeddings
        community_reports: Storage for community reports
        text_chunks_db: Storage for original text chunks
        query_param: Query configuration parameters
        tokenizer_wrapper: TokenizerWrapper for token management
        global_config: Global configuration with LLM settings

    Returns:
        str: LLM-generated response based on local context, or context only if requested
    """
    use_model_func = global_config["best_model_func"]
    context = await _build_local_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
        tokenizer_wrapper,
    )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response


async def _map_global_communities(
    query: str,
    communities_data: list[CommunitySchema],
    query_param: QueryParam,
    global_config: dict,
    tokenizer_wrapper,
):
    use_string_json_convert_func = global_config["convert_response_to_json_func"]
    use_model_func = global_config["best_model_func"]
    community_groups = []
    while len(communities_data):
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=query_param.global_max_token_for_community_report,
            tokenizer_wrapper=tokenizer_wrapper, # 传入 wrapper
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group) :]

    async def _process(community_truncated_datas: list[CommunitySchema]) -> dict:
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", 0),
                    c["occurrence"],
                ]
            )
        community_context = list_of_list_to_csv(communities_section_list)
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            **query_param.global_special_community_map_llm_kwargs,
        )
        data = use_string_json_convert_func(response)
        return data.get("points", [])

    logger.info(f"Grouping to {len(community_groups)} groups for global search")
    responses = await asyncio.gather(*[_process(c) for c in community_groups])
    return responses


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
) -> str:
    """
    Perform a global knowledge graph query.

    Global search uses a map-reduce approach over community reports to answer
    broad questions about the entire dataset. Multiple "analyst" perspectives are
    generated and then reduced into a final answer. Best for high-level questions
    requiring a holistic view.

    Args:
        query: User query string
        knowledge_graph_inst: Knowledge graph storage instance
        entities_vdb: Vector database containing entity embeddings (unused in global search)
        community_reports: Storage for community reports
        text_chunks_db: Storage for original text chunks (unused in global search)
        query_param: Query configuration parameters
        tokenizer_wrapper: TokenizerWrapper for token management
        global_config: Global configuration with LLM settings

    Returns:
        str: LLM-generated response based on global community analysis
    """
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    if not len(community_schema):
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]

    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info(f"Revtrieved {len(community_datas)} communities")

    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config, tokenizer_wrapper
    )
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        return PROMPTS["fail_response"]
    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
        tokenizer_wrapper=tokenizer_wrapper, # 传入 wrapper
    )
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
Importance Score: {dp['score']}
{dp['answer']}
"""
        )
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, response_type=query_param.response_type
        ),
    )
    return response


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
):
    """
    Perform a naive vector-based similarity search query.

    This is the simplest query mode that performs traditional vector similarity search
    without using the knowledge graph structure. It finds and retrieves the most similar
    text chunks and passes them to an LLM for answering.

    Args:
        query: User query string
        chunks_vdb: Vector database containing text chunk embeddings
        text_chunks_db: Storage for original text chunks
        query_param: Query configuration parameters
        tokenizer_wrapper: TokenizerWrapper for token management
        global_config: Global configuration with LLM settings

    Returns:
        str: LLM-generated response based on retrieved chunks, or chunks only if requested
    """
    use_model_func = global_config["best_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
        tokenizer_wrapper=tokenizer_wrapper, # 传入 wrapper
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response
