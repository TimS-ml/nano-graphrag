"""Large Language Model (LLM) integration module for nano-graphrag.

This module provides asynchronous interfaces to various LLM providers including OpenAI,
Azure OpenAI, and Amazon Bedrock. It implements caching mechanisms, retry logic, and
singleton client patterns for efficient API interactions.

The module supports:
- Text completion with multiple LLM providers (OpenAI GPT-4o, Azure OpenAI, Amazon Bedrock)
- Text embeddings generation (OpenAI, Azure OpenAI, Amazon Bedrock Titan)
- Response caching to reduce API calls and costs
- Automatic retry with exponential backoff for rate limits and connection errors
- Singleton pattern for API clients to manage connections efficiently

Key features:
- Cache-enabled completion functions to avoid redundant API calls
- Factory pattern for creating provider-specific completion functions
- Global client instances for connection pooling
- Support for conversation history and system prompts
"""

import json
import numpy as np
from typing import Optional, List, Any, Callable

import aioboto3
from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage

global_openai_async_client = None
global_azure_openai_async_client = None
global_amazon_bedrock_async_client = None


def get_openai_async_client_instance():
    """Get or create a singleton instance of the OpenAI async client.

    This function implements the singleton pattern to ensure only one OpenAI client
    instance exists throughout the application lifecycle, which helps manage connections
    efficiently and avoid creating redundant client instances.

    Returns:
        AsyncOpenAI: The global AsyncOpenAI client instance.

    Note:
        The client is initialized using environment variables for API key configuration.
        Refer to OpenAI's documentation for required environment variables (e.g., OPENAI_API_KEY).
    """
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    """Get or create a singleton instance of the Azure OpenAI async client.

    This function implements the singleton pattern to ensure only one Azure OpenAI client
    instance exists throughout the application lifecycle, which helps manage connections
    efficiently and avoid creating redundant client instances.

    Returns:
        AsyncAzureOpenAI: The global AsyncAzureOpenAI client instance.

    Note:
        The client is initialized using environment variables for API key and endpoint configuration.
        Required environment variables include AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.
    """
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client


def get_amazon_bedrock_async_client_instance():
    """Get or create a singleton instance of the Amazon Bedrock async client session.

    This function implements the singleton pattern to ensure only one Amazon Bedrock
    session exists throughout the application lifecycle, which helps manage connections
    efficiently and avoid creating redundant session instances.

    Returns:
        aioboto3.Session: The global aioboto3 Session instance for Amazon Bedrock.

    Note:
        The session is initialized using AWS credentials from environment variables or
        AWS configuration files. Required credentials include AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY, and optionally AWS_SESSION_TOKEN and AWS_REGION.
    """
    global global_amazon_bedrock_async_client
    if global_amazon_bedrock_async_client is None:
        global_amazon_bedrock_async_client = aioboto3.Session()
    return global_amazon_bedrock_async_client


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Generate text completion using OpenAI API with optional caching support.

    This function sends a completion request to OpenAI's chat completion API with support
    for caching responses to reduce API calls. It automatically retries on rate limit errors
    and connection issues with exponential backoff.

    Args:
        model: The OpenAI model identifier to use (e.g., "gpt-4o", "gpt-4o-mini").
        prompt: The user prompt/question to send to the model.
        system_prompt: Optional system prompt to set the behavior and context for the model.
            Defaults to None.
        history_messages: List of previous conversation messages in OpenAI format.
            Each message should be a dict with "role" and "content" keys. Defaults to [].
        **kwargs: Additional keyword arguments to pass to the OpenAI API.
            Special kwargs:
                - hashing_kv (BaseKVStorage): Optional key-value storage for caching responses.
                  If provided, responses will be cached and retrieved based on argument hash.
            Other kwargs are passed directly to the chat.completions.create() method.

    Returns:
        str: The generated completion text from the model.

    Note:
        - This function uses a global singleton OpenAI client instance.
        - Retries up to 5 times with exponential backoff (4-10 seconds) on rate limit
          or connection errors.
        - When hashing_kv is provided, the function checks cache before making API calls
          and stores new responses in the cache.
        - The cache key is computed from the model name and message history.
    """
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def amazon_bedrock_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Generate text completion using Amazon Bedrock API with optional caching support.

    This function sends a completion request to Amazon Bedrock's Converse API with support
    for caching responses to reduce API calls. It automatically retries on rate limit errors
    and connection issues with exponential backoff.

    Args:
        model: The Amazon Bedrock model identifier to use
            (e.g., "us.anthropic.claude-3-sonnet-20240229-v1:0").
        prompt: The user prompt/question to send to the model.
        system_prompt: Optional system prompt to set the behavior and context for the model.
            Defaults to None.
        history_messages: List of previous conversation messages in Bedrock format.
            Each message should be a dict with "role" and "content" keys. Defaults to [].
        **kwargs: Additional keyword arguments.
            Special kwargs:
                - hashing_kv (BaseKVStorage): Optional key-value storage for caching responses.
                  If provided, responses will be cached and retrieved based on argument hash.
                - max_tokens (int): Maximum number of tokens to generate. Defaults to 4096.
            Other kwargs are not currently passed to the Bedrock API.

    Returns:
        str: The generated completion text from the model.

    Note:
        - This function uses a global singleton aioboto3 Session instance.
        - Retries up to 5 times with exponential backoff (4-10 seconds) on rate limit
          or connection errors.
        - The AWS region is determined by the AWS_REGION environment variable,
          defaulting to "us-east-1".
        - When hashing_kv is provided, the function checks cache before making API calls
          and stores new responses in the cache.
        - The inference configuration uses temperature=0 for deterministic outputs.
    """
    amazon_bedrock_async_client = get_amazon_bedrock_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": [{"text": prompt}]})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    inference_config = {
        "temperature": 0,
        "maxTokens": 4096 if "max_tokens" not in kwargs else kwargs["max_tokens"],
    }

    async with amazon_bedrock_async_client.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        if system_prompt:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
                system=[{"text": system_prompt}]
            )
        else:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
            )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response["output"]["message"]["content"][0]["text"], "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response["output"]["message"]["content"][0]["text"]


def create_amazon_bedrock_complete_function(model_id: str) -> Callable:
    """
    Factory function to dynamically create completion functions for Amazon Bedrock

    Args:
        model_id (str): Amazon Bedrock model identifier (e.g., "us.anthropic.claude-3-sonnet-20240229-v1:0")

    Returns:
        Callable: Generated completion function
    """
    async def bedrock_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Any] = [],
        **kwargs
    ) -> str:
        return await amazon_bedrock_complete_if_cache(
            model_id,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
    
    # Set function name for easier debugging
    bedrock_complete.__name__ = f"{model_id}_complete"
    
    return bedrock_complete


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Generate text completion using OpenAI's GPT-4o model.

    This is a convenience wrapper around openai_complete_if_cache that uses the
    "gpt-4o" model by default. Supports caching and automatic retries.

    Args:
        prompt: The user prompt/question to send to the model.
        system_prompt: Optional system prompt to set the behavior and context for the model.
            Defaults to None.
        history_messages: List of previous conversation messages in OpenAI format.
            Each message should be a dict with "role" and "content" keys. Defaults to [].
        **kwargs: Additional keyword arguments passed to openai_complete_if_cache.
            See openai_complete_if_cache for details on supported arguments.

    Returns:
        str: The generated completion text from GPT-4o.

    Note:
        See openai_complete_if_cache for details on retry logic and caching behavior.
    """
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Generate text completion using OpenAI's GPT-4o-mini model.

    This is a convenience wrapper around openai_complete_if_cache that uses the
    "gpt-4o-mini" model by default. GPT-4o-mini is a faster and more cost-effective
    variant of GPT-4o. Supports caching and automatic retries.

    Args:
        prompt: The user prompt/question to send to the model.
        system_prompt: Optional system prompt to set the behavior and context for the model.
            Defaults to None.
        history_messages: List of previous conversation messages in OpenAI format.
            Each message should be a dict with "role" and "content" keys. Defaults to [].
        **kwargs: Additional keyword arguments passed to openai_complete_if_cache.
            See openai_complete_if_cache for details on supported arguments.

    Returns:
        str: The generated completion text from GPT-4o-mini.

    Note:
        See openai_complete_if_cache for details on retry logic and caching behavior.
    """
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def amazon_bedrock_embedding(texts: list[str]) -> np.ndarray:
    """Generate text embeddings using Amazon Bedrock's Titan Embed Text v2 model.

    This function converts a list of text strings into numerical vector embeddings
    using Amazon Bedrock's embedding model. The embeddings can be used for semantic
    search, similarity comparison, and other NLP tasks.

    Args:
        texts: A list of text strings to convert into embeddings.

    Returns:
        np.ndarray: A 2D numpy array of shape (len(texts), 1024) containing the
            embedding vectors. Each row represents the embedding for the corresponding
            input text.

    Note:
        - This function uses the "amazon.titan-embed-text-v2:0" model.
        - Embedding dimension is 1024.
        - Maximum token size is 8192 tokens per text.
        - Retries up to 5 times with exponential backoff (4-10 seconds) on rate limit
          or connection errors.
        - The AWS region is determined by the AWS_REGION environment variable,
          defaulting to "us-east-1".
        - Embeddings are generated sequentially for each text in the input list.
    """
    amazon_bedrock_async_client = get_amazon_bedrock_async_client_instance()

    async with amazon_bedrock_async_client.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        embeddings = []
        for text in texts:
            body = json.dumps(
                {
                    "inputText": text,
                    "dimensions": 1024,
                }
            )
            response = await bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0", body=body,
            )
            response_body = await response.get("body").read()
            embeddings.append(json.loads(response_body))
    return np.array([dp["embedding"] for dp in embeddings])


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    """Generate text embeddings using OpenAI's text-embedding-3-small model.

    This function converts a list of text strings into numerical vector embeddings
    using OpenAI's embedding model. The embeddings can be used for semantic search,
    similarity comparison, clustering, and other NLP tasks.

    Args:
        texts: A list of text strings to convert into embeddings.

    Returns:
        np.ndarray: A 2D numpy array of shape (len(texts), 1536) containing the
            embedding vectors. Each row represents the embedding for the corresponding
            input text.

    Note:
        - This function uses the "text-embedding-3-small" model.
        - Embedding dimension is 1536.
        - Maximum token size is 8192 tokens per text.
        - Retries up to 5 times with exponential backoff (4-10 seconds) on rate limit
          or connection errors.
        - This function uses a global singleton OpenAI client instance.
        - All texts are processed in a single API call for efficiency.
    """
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Generate text completion using Azure OpenAI API with optional caching support.

    This function sends a completion request to Azure OpenAI's chat completion API with
    support for caching responses to reduce API calls. It automatically retries on rate
    limit errors and connection issues with exponential backoff.

    Args:
        deployment_name: The Azure OpenAI deployment name to use (e.g., "gpt-4o").
            This corresponds to the deployment you created in Azure Portal.
        prompt: The user prompt/question to send to the model.
        system_prompt: Optional system prompt to set the behavior and context for the model.
            Defaults to None.
        history_messages: List of previous conversation messages in OpenAI format.
            Each message should be a dict with "role" and "content" keys. Defaults to [].
        **kwargs: Additional keyword arguments to pass to the Azure OpenAI API.
            Special kwargs:
                - hashing_kv (BaseKVStorage): Optional key-value storage for caching responses.
                  If provided, responses will be cached and retrieved based on argument hash.
            Other kwargs are passed directly to the chat.completions.create() method.

    Returns:
        str: The generated completion text from the model.

    Note:
        - This function uses a global singleton Azure OpenAI client instance.
        - Retries up to 3 times with exponential backoff (4-10 seconds) on rate limit
          or connection errors.
        - When hashing_kv is provided, the function checks cache before making API calls
          and stores new responses in the cache.
        - The cache key is computed from the deployment name and message history.
        - Requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.
    """
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Generate text completion using Azure OpenAI's GPT-4o deployment.

    This is a convenience wrapper around azure_openai_complete_if_cache that uses the
    "gpt-4o" deployment name by default. Supports caching and automatic retries.

    Args:
        prompt: The user prompt/question to send to the model.
        system_prompt: Optional system prompt to set the behavior and context for the model.
            Defaults to None.
        history_messages: List of previous conversation messages in OpenAI format.
            Each message should be a dict with "role" and "content" keys. Defaults to [].
        **kwargs: Additional keyword arguments passed to azure_openai_complete_if_cache.
            See azure_openai_complete_if_cache for details on supported arguments.

    Returns:
        str: The generated completion text from GPT-4o.

    Note:
        See azure_openai_complete_if_cache for details on retry logic and caching behavior.
        Requires an Azure OpenAI deployment named "gpt-4o" in your Azure subscription.
    """
    return await azure_openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Generate text completion using Azure OpenAI's GPT-4o-mini deployment.

    This is a convenience wrapper around azure_openai_complete_if_cache that uses the
    "gpt-4o-mini" deployment name by default. GPT-4o-mini is a faster and more
    cost-effective variant of GPT-4o. Supports caching and automatic retries.

    Args:
        prompt: The user prompt/question to send to the model.
        system_prompt: Optional system prompt to set the behavior and context for the model.
            Defaults to None.
        history_messages: List of previous conversation messages in OpenAI format.
            Each message should be a dict with "role" and "content" keys. Defaults to [].
        **kwargs: Additional keyword arguments passed to azure_openai_complete_if_cache.
            See azure_openai_complete_if_cache for details on supported arguments.

    Returns:
        str: The generated completion text from GPT-4o-mini.

    Note:
        See azure_openai_complete_if_cache for details on retry logic and caching behavior.
        Requires an Azure OpenAI deployment named "gpt-4o-mini" in your Azure subscription.
    """
    return await azure_openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
    """Generate text embeddings using Azure OpenAI's text-embedding-3-small model.

    This function converts a list of text strings into numerical vector embeddings
    using Azure OpenAI's embedding model. The embeddings can be used for semantic search,
    similarity comparison, clustering, and other NLP tasks.

    Args:
        texts: A list of text strings to convert into embeddings.

    Returns:
        np.ndarray: A 2D numpy array of shape (len(texts), 1536) containing the
            embedding vectors. Each row represents the embedding for the corresponding
            input text.

    Note:
        - This function uses the "text-embedding-3-small" model deployment.
        - Embedding dimension is 1536.
        - Maximum token size is 8192 tokens per text.
        - Retries up to 3 times with exponential backoff (4-10 seconds) on rate limit
          or connection errors.
        - This function uses a global singleton Azure OpenAI client instance.
        - All texts are processed in a single API call for efficiency.
        - Requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.
    """
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
