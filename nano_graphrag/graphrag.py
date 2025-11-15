"""GraphRAG implementation module.

This module provides the main GraphRAG class, which implements a graph-based retrieval
augmented generation (RAG) system. GraphRAG combines knowledge graph construction,
community detection, and vector-based retrieval to enable powerful document indexing
and querying capabilities.

The module supports:
- Document chunking and embedding
- Entity extraction and relation mapping
- Graph clustering and community detection
- Multiple query modes (local, global, naive)
- Customizable storage backends (vector DB, graph DB, key-value store)
- Integration with various LLM providers (OpenAI, Azure OpenAI, Amazon Bedrock)
"""
import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast



from ._llm import (
    amazon_bedrock_embedding,
    create_amazon_bedrock_complete_function,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
)
from ._op import (
    chunking_by_token_size,
    extract_entities,
    generate_community_report,
    get_chunks,
    local_query,
    global_query,
    naive_query,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
    TokenizerWrapper,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)


@dataclass
class GraphRAG:
    """Main GraphRAG class for document indexing and querying.

    GraphRAG is a graph-based retrieval augmented generation system that enables
    intelligent document indexing and querying. It processes documents by:
    1. Chunking text into manageable pieces
    2. Extracting entities and relationships to build a knowledge graph
    3. Creating community reports through graph clustering
    4. Enabling multiple query modes for different use cases

    The class supports both synchronous and asynchronous operations, with customizable
    storage backends, embedding functions, and LLM providers.

    Attributes:
        working_dir: Directory path for storing cache and data files. Defaults to a
            timestamped directory in the current location.
        enable_local: If True, enables local query mode using entity-based retrieval.
        enable_naive_rag: If True, enables naive RAG mode using simple chunk retrieval.
        tokenizer_type: Type of tokenizer to use ('tiktoken' or 'huggingface').
        tiktoken_model_name: Model name for tiktoken tokenizer.
        huggingface_model_name: Model name for HuggingFace tokenizer.
        chunk_func: Function to use for chunking documents.
        chunk_token_size: Maximum token size for each text chunk.
        chunk_overlap_token_size: Number of overlapping tokens between chunks.
        entity_extract_max_gleaning: Maximum number of entity extraction iterations.
        entity_summary_to_max_tokens: Maximum tokens for entity summaries.
        graph_cluster_algorithm: Algorithm for graph clustering (e.g., 'leiden').
        max_graph_cluster_size: Maximum size for graph clusters.
        graph_cluster_seed: Random seed for reproducible clustering.
        node_embedding_algorithm: Algorithm for node embeddings (e.g., 'node2vec').
        node2vec_params: Parameters for node2vec algorithm.
        special_community_report_llm_kwargs: Additional kwargs for community report LLM.
        embedding_func: Function to generate embeddings for text.
        embedding_batch_num: Batch size for embedding operations.
        embedding_func_max_async: Maximum concurrent embedding operations.
        query_better_than_threshold: Similarity threshold for query results.
        using_azure_openai: If True, use Azure OpenAI instead of standard OpenAI.
        using_amazon_bedrock: If True, use Amazon Bedrock for LLM operations.
        best_model_id: Model ID for the best (most capable) LLM.
        cheap_model_id: Model ID for the cheap (faster/cheaper) LLM.
        best_model_func: Function for the best LLM completion.
        best_model_max_token_size: Maximum token size for best model.
        best_model_max_async: Maximum concurrent calls to best model.
        cheap_model_func: Function for the cheap LLM completion.
        cheap_model_max_token_size: Maximum token size for cheap model.
        cheap_model_max_async: Maximum concurrent calls to cheap model.
        entity_extraction_func: Function to extract entities from text.
        key_string_value_json_storage_cls: Storage class for key-value JSON data.
        vector_db_storage_cls: Storage class for vector database.
        vector_db_storage_cls_kwargs: Additional kwargs for vector DB storage.
        graph_storage_cls: Storage class for graph data.
        enable_llm_cache: If True, enables caching of LLM responses.
        always_create_working_dir: If True, creates working directory if it doesn't exist.
        addon_params: Additional custom parameters for extensions.
        convert_response_to_json_func: Function to convert LLM responses to JSON.
    """
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = False

    # text chunking
    tokenizer_type: str = "tiktoken"  # or 'huggingface'
    tiktoken_model_name: str = "gpt-4o"
    huggingface_model_name: str = "bert-base-uncased"  # default HF model
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            TokenizerWrapper,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # text embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2

    # LLM
    using_azure_openai: bool = False
    using_amazon_bedrock: bool = False
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"
    best_model_func: callable = gpt_4o_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

    # entity extraction
    entity_extraction_func: callable = extract_entities

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        """Initialize GraphRAG instance after dataclass initialization.

        This method is automatically called after the dataclass __init__ method.
        It performs the following initialization tasks:
        1. Logs the configuration parameters
        2. Initializes the tokenizer wrapper
        3. Configures LLM providers (Azure OpenAI or Amazon Bedrock if specified)
        4. Creates the working directory if needed
        5. Initializes all storage instances (docs, chunks, graphs, vector DBs)
        6. Sets up rate-limited LLM and embedding functions

        Note:
            This method modifies several attributes based on configuration flags
            (using_azure_openai, using_amazon_bedrock).
        """
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        self.tokenizer_wrapper = TokenizerWrapper(
            tokenizer_type=self.tokenizer_type,
            model_name=self.tiktoken_model_name if self.tokenizer_type == "tiktoken" else self.huggingface_model_name
        )

        if self.using_azure_openai:
            # If there's no OpenAI API key, use Azure OpenAI
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )

        if self.using_amazon_bedrock:
            self.best_model_func = create_amazon_bedrock_complete_function(self.best_model_id)
            self.cheap_model_func = create_amazon_bedrock_complete_function(self.cheap_model_id)
            self.embedding_func = amazon_bedrock_embedding
            logger.info(
                "Switched the default openai funcs to Amazon Bedrock"
            )

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )



    def insert(self, string_or_strings):
        """Insert documents into the GraphRAG system (synchronous wrapper).

        This is a synchronous wrapper around the async ainsert method. It processes
        the input documents by chunking, extracting entities, building a knowledge
        graph, and generating community reports.

        Args:
            string_or_strings: Either a single document string or a list of document
                strings to be inserted into the system.

        Returns:
            None. The method processes and stores the documents in the configured
            storage backends.

        Note:
            This method creates or reuses an event loop to run the async operation.
            For better performance in async contexts, use ainsert directly.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        """Query the GraphRAG system (synchronous wrapper).

        This is a synchronous wrapper around the async aquery method. It executes
        a query against the indexed documents using the specified query mode.

        Args:
            query: The query string to search for.
            param: Query parameters specifying the search mode and other options.
                Defaults to a new QueryParam instance with default settings.

        Returns:
            The query response, which varies based on the query mode:
            - local mode: Entity-based retrieval results
            - global mode: Community-based retrieval results
            - naive mode: Simple chunk-based retrieval results

        Note:
            This method creates or reuses an event loop to run the async operation.
            For better performance in async contexts, use aquery directly.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """Query the GraphRAG system asynchronously.

        Executes a query against the indexed documents using the specified query mode.
        The method supports three query modes:
        - local: Uses entity-based retrieval for focused, specific queries
        - global: Uses community reports for broad, comprehensive queries
        - naive: Uses simple chunk-based retrieval (traditional RAG)

        Args:
            query: The query string to search for.
            param: Query parameters specifying the search mode and other options.
                Defaults to a new QueryParam instance with default settings.

        Returns:
            The query response containing relevant information from the indexed
            documents. The response format depends on the query mode selected.

        Raises:
            ValueError: If local mode is requested but enable_local is False,
                or if naive mode is requested but enable_naive_rag is False,
                or if an unknown mode is specified.

        Note:
            After the query completes, the method calls _query_done() to perform
            cleanup operations like saving the LLM response cache.
        """
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if param.mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
            )
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, string_or_strings):
        """Insert documents into the GraphRAG system asynchronously.

        This method performs the complete document indexing pipeline:
        1. Filters out duplicate documents based on content hash
        2. Chunks documents into smaller pieces for processing
        3. Extracts entities and relationships from chunks
        4. Builds/updates the knowledge graph
        5. Performs graph clustering to identify communities
        6. Generates community reports
        7. Updates all storage backends

        Args:
            string_or_strings: Either a single document string or a list of document
                strings to be inserted into the system.

        Returns:
            None. The method processes and stores the documents in the configured
            storage backends.

        Note:
            - Duplicate documents (based on MD5 hash) are automatically filtered out
            - Community reports are regenerated on each insert (incremental updates
              not yet supported)
            - The method calls _insert_start() at the beginning and _insert_done()
              at the end to manage storage indexing callbacks
            - If no new documents or chunks are found, the method returns early

        Raises:
            Any exceptions from the entity extraction or storage operations are
            propagated to the caller after _insert_done() is called in the finally block.
        """
        await self._insert_start()
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking

            inserting_chunks = get_chunks(
                new_docs=new_docs,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
                tokenizer_wrapper=self.tokenizer_wrapper,
            )

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: don't support incremental update for communities now, so we have to drop all
            await self.community_reports.drop()

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.entity_extraction_func(
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                tokenizer_wrapper=self.tokenizer_wrapper,
                global_config=asdict(self),
                using_amazon_bedrock=self.using_amazon_bedrock,
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- update clusterings of graph
            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, self.tokenizer_wrapper, asdict(self)
            )

            # ---------- commit upsertings and indexing
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_start(self):
        """Prepare storage instances for document insertion.

        This method is called at the beginning of the insert operation to notify
        storage instances that an indexing operation is starting. This allows
        storage backends to perform any necessary preparation, such as opening
        connections or preparing batch operations.

        Returns:
            None. The method executes all index_start_callback() methods
            concurrently and waits for them to complete.

        Note:
            Currently only triggers callbacks for the chunk_entity_relation_graph.
            Storage instances that are None are skipped.
        """
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        """Finalize storage instances after document insertion.

        This method is called at the end of the insert operation (in a finally block)
        to notify storage instances that an indexing operation is complete. This
        allows storage backends to perform cleanup, flush buffers, save indexes,
        or close connections.

        Returns:
            None. The method executes all index_done_callback() methods
            concurrently and waits for them to complete.

        Note:
            This method is called in a finally block to ensure it runs even if
            errors occur during insertion. It triggers callbacks for all storage
            instances: full_docs, text_chunks, llm_response_cache, community_reports,
            entities_vdb, chunks_vdb, and chunk_entity_relation_graph.
            Storage instances that are None are skipped.
        """
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        """Finalize storage instances after query execution.

        This method is called at the end of query operations to notify storage
        instances to perform cleanup tasks. Primarily used to ensure the LLM
        response cache is properly saved after query execution.

        Returns:
            None. The method executes all index_done_callback() methods
            concurrently and waits for them to complete.

        Note:
            Currently only triggers callbacks for llm_response_cache to save
            any cached responses generated during the query.
            Storage instances that are None are skipped.
        """
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
