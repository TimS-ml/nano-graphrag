"""Utility functions and classes for nano-graphrag.

This module provides various utility functions and classes used throughout the
nano-graphrag package, including:
- JSON parsing and extraction utilities
- Tokenization wrappers for different tokenizer backends
- String manipulation and cleaning functions
- File I/O helpers for JSON data
- Hashing utilities
- Async function decorators and helpers
- Embedding function wrappers

The utilities support both OpenAI-style (tiktoken) and HuggingFace tokenizers,
and provide robust handling of malformed JSON responses from LLMs.
"""

import asyncio
import html
import json
import logging
import os
import re
import numbers
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, Literal

import numpy as np
import tiktoken


from transformers import AutoTokenizer

logger = logging.getLogger("nano-graphrag")
logging.getLogger("neo4j").setLevel(logging.ERROR)

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an asyncio event loop.

    This function ensures that an event loop is always available, even in
    sub-threads where one might not exist. If an event loop already exists,
    it returns that loop. If not (e.g., in a sub-thread), it creates a new
    event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.

    Note:
        This is particularly useful when working with asyncio in multi-threaded
        environments where event loops may not be automatically created.
    """
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If in a sub-thread, create a new event loop.
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def extract_first_complete_json(s: str):
    """Extract the first complete JSON object from a string.

    This function uses a stack-based approach to track opening and closing braces
    to identify the first valid JSON object in a string. This is useful when parsing
    LLM responses that may contain JSON objects embedded in other text.

    Args:
        s (str): The input string that may contain one or more JSON objects.

    Returns:
        dict | None: The first complete JSON object found in the string, or None
            if no valid JSON object could be extracted.

    Note:
        - The function removes newlines before attempting to parse JSON.
        - If JSON decoding fails, an error is logged with the first 50 characters
          of the attempted JSON string.
        - Only the first complete JSON object is returned, even if multiple exist.
    """
    stack = []
    first_json_start = None

    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    first_json_str = s[first_json_start:i+1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}...")
                        return None
                    finally:
                        first_json_start = None
    logger.warning("No complete JSON object found in the input string.")
    return None

def parse_value(value: str):
    """Convert a string value to its appropriate Python type.

    This function acts as a safer alternative to eval() for parsing string
    values into their appropriate types. It handles JSON-like literals and
    numeric types.

    Args:
        value (str): The string value to parse.

    Returns:
        Any: The parsed value as one of the following types:
            - None: if value is "null"
            - bool: if value is "true" or "false"
            - float: if value contains a decimal point and is numeric
            - int: if value is a whole number
            - str: if value cannot be converted to other types (quotes removed)

    Note:
        This function strips leading/trailing whitespace and removes surrounding
        quotes from string values.
    """
    value = value.strip()

    if value == "null":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    else:
        # Try to convert to int or float
        try:
            if '.' in value:  # If there's a dot, it might be a float
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If conversion fails, return the value as-is (likely a string)
            return value.strip('"')  # Remove surrounding quotes if they exist

def extract_values_from_json(json_string, keys=["reasoning", "answer", "data"], allow_no_quotes=False):
    """Extract key-value pairs from a non-standard or malformed JSON string.

    This function handles JSON-like strings that may not be properly formatted,
    including those with unquoted keys or values, and nested objects. It's
    particularly useful for parsing LLM responses that may not produce perfect JSON.

    Args:
        json_string (str): The JSON-like string to parse.
        keys (list[str], optional): List of keys to extract. Defaults to
            ["reasoning", "answer", "data"]. Note: Currently not actively used
            in filtering, all found keys are extracted.
        allow_no_quotes (bool, optional): Whether to allow unquoted values.
            Defaults to False. Note: The regex pattern supports both quoted
            and unquoted values regardless of this parameter.

    Returns:
        dict: A dictionary containing the extracted key-value pairs. Nested
            JSON objects are recursively parsed into nested dictionaries.

    Note:
        - The function uses regex to match key-value patterns.
        - Nested objects (values starting with '{' and ending with '}') are
          recursively parsed.
        - Values are automatically converted to appropriate types using parse_value().
        - If no values can be extracted, a warning is logged.
    """
    extracted_values = {}

    # Enhanced pattern to match both quoted and unquoted values, as well as nested objects
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'

    for match in re.finditer(regex_pattern, json_string, re.DOTALL):
        key = match.group('key').strip('"')  # Strip quotes from key
        value = match.group('value').strip()

        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith('{') and value.endswith('}'):
            extracted_values[key] = extract_values_from_json(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value(value)

    if not extracted_values:
        logger.warning("No values could be extracted from the string.")

    return extracted_values


def convert_response_to_json(response: str) -> dict:
    """Convert a response string to a JSON dictionary with fallback strategies.

    This function attempts to extract JSON data from a response string using
    multiple strategies. First, it tries to extract a complete, well-formed JSON
    object. If that fails, it falls back to extracting values from malformed or
    non-standard JSON strings.

    Args:
        response (str): The response string to parse, typically from an LLM.

    Returns:
        dict: The extracted JSON data as a dictionary. Returns an empty dict
            or None if extraction fails completely.

    Note:
        - First attempts to extract a complete JSON object using
          extract_first_complete_json().
        - Falls back to extract_values_from_json() for malformed JSON.
        - Logs appropriate info/error messages at each step.
        - Particularly useful for handling unpredictable LLM response formats.
    """
    prediction_json = extract_first_complete_json(response)

    if prediction_json is None:
        logger.info("Attempting to extract values from a non-standard JSON string...")
        prediction_json = extract_values_from_json(response, allow_no_quotes=True)

    if not prediction_json:
        logger.error("Unable to extract meaningful data from the response.")
    else:
        logger.info("JSON data successfully extracted.")

    return prediction_json




class TokenizerWrapper:
    """A unified wrapper for different tokenizer backends.

    This class provides a consistent interface for tokenization across different
    backends, supporting both OpenAI's tiktoken and HuggingFace tokenizers. The
    tokenizer is lazy-loaded on first use to avoid unnecessary initialization.

    Attributes:
        tokenizer_type (Literal["tiktoken", "huggingface"]): The type of tokenizer
            to use.
        model_name (str): The name of the model whose tokenizer to load.
        _tokenizer: The underlying tokenizer object (lazy-loaded).

    Note:
        The tokenizer is not loaded until the first encoding/decoding operation,
        which helps reduce initialization time when the tokenizer may not be used.
    """

    def __init__(self, tokenizer_type: Literal["tiktoken", "huggingface"] = "tiktoken", model_name: str = "gpt-4o"):
        """Initialize the TokenizerWrapper.

        Args:
            tokenizer_type (Literal["tiktoken", "huggingface"], optional): The type
                of tokenizer to use. Defaults to "tiktoken".
            model_name (str, optional): The name of the model whose tokenizer to
                load. For tiktoken, this should be an OpenAI model name. For
                huggingface, this should be a HuggingFace model identifier.
                Defaults to "gpt-4o".
        """
        self.tokenizer_type = tokenizer_type
        self.model_name = model_name
        self._tokenizer = None
        self._lazy_load_tokenizer()

    def _lazy_load_tokenizer(self):
        """Lazy-load the tokenizer on first use.

        This internal method loads the appropriate tokenizer based on the
        tokenizer_type. It only loads once, subsequent calls return immediately.

        Raises:
            ImportError: If huggingface tokenizer is requested but transformers
                library is not installed.
            ValueError: If an unknown tokenizer_type is specified.

        Note:
            This method is called automatically by other methods and should not
            be called directly by users.
        """
        if self._tokenizer is not None:
            return
        logger.info(f"Loading tokenizer: type='{self.tokenizer_type}', name='{self.model_name}'")
        if self.tokenizer_type == "tiktoken":
            self._tokenizer = tiktoken.encoding_for_model(self.model_name)
        elif self.tokenizer_type == "huggingface":
            if AutoTokenizer is None:
                raise ImportError("`transformers` is not installed. Please install it via `pip install transformers` to use HuggingFace tokenizers.")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        else:
            raise ValueError(f"Unknown tokenizer_type: {self.tokenizer_type}")

    def get_tokenizer(self):
        """Get the underlying tokenizer object.

        This method provides direct access to the underlying tokenizer object
        for special use cases that require tokenizer-specific functionality.

        Returns:
            The underlying tokenizer object (tiktoken.Encoding or
            transformers.PreTrainedTokenizer).

        Note:
            Use this method sparingly. Prefer using the encode(), decode(), and
            decode_batch() methods for most use cases to maintain abstraction.
        """
        self._lazy_load_tokenizer()
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        self._lazy_load_tokenizer()
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back into text.

        Args:
            tokens (list[int]): The list of token IDs to decode.

        Returns:
            str: The decoded text.
        """
        self._lazy_load_tokenizer()
        return self._tokenizer.decode(tokens)

    def decode_batch(self, tokens_list: list[list[int]]) -> list[str]:
        """Decode multiple sequences of token IDs into text efficiently.

        This method provides batch decoding functionality for improved efficiency
        when decoding multiple sequences.

        Args:
            tokens_list (list[list[int]]): A list of token ID sequences to decode.

        Returns:
            list[str]: A list of decoded text strings corresponding to each input
                sequence.

        Raises:
            ValueError: If an unknown tokenizer_type is encountered.

        Note:
            - For HuggingFace tokenizers, uses the native batch_decode() method.
            - For tiktoken, simulates batch decoding using list comprehension.
            - HuggingFace batch decoding skips special tokens by default.
        """
        self._lazy_load_tokenizer()
        # HuggingFace tokenizer has batch_decode, but tiktoken doesn't, so we simulate it
        if self.tokenizer_type == "tiktoken":
            return [self._tokenizer.decode(tokens) for tokens in tokens_list]
        elif self.tokenizer_type == "huggingface":
            return self._tokenizer.batch_decode(tokens_list, skip_special_tokens=True)
        else:
             raise ValueError(f"Unknown tokenizer_type: {self.tokenizer_type}")
        


def truncate_list_by_token_size(
    list_data: list,
    key: callable,
    max_token_size: int,
    tokenizer_wrapper: TokenizerWrapper
):
    """Truncate a list of items to fit within a maximum token size.

    This function truncates a list by counting tokens for each item until the
    maximum token size is reached. It's useful for fitting data into LLM context
    windows with token limits.

    Args:
        list_data (list): The list of data items to truncate.
        key (callable): A function that extracts the string to tokenize from
            each data item. Should take one data item and return a string.
        max_token_size (int): The maximum number of tokens allowed. If <= 0,
            returns an empty list.
        tokenizer_wrapper (TokenizerWrapper): The tokenizer wrapper to use for
            encoding text into tokens.

    Returns:
        list: A truncated list containing items from the beginning of list_data
            that fit within max_token_size. Returns empty list if max_token_size <= 0.

    Note:
        - Adds 1 token per item to simulate newline separation when items are
          concatenated.
        - Truncates conservatively: stops at the first item that would exceed
          the limit.
        - If the first item exceeds max_token_size, returns an empty list.
    """
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(tokenizer_wrapper.encode(key(data))) + 1  # +1 to simulate newline separation
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def compute_mdhash_id(content, prefix: str = ""):
    """Compute an MD5 hash ID for the given content.

    Args:
        content (str): The content to hash.
        prefix (str, optional): A prefix to prepend to the hash. Defaults to "".

    Returns:
        str: The MD5 hash of the content (hexadecimal), with the optional prefix.

    Note:
        The content is encoded to UTF-8 before hashing.
    """
    return prefix + md5(content.encode()).hexdigest()


def write_json(json_obj, file_name):
    """Write a JSON object to a file.

    Args:
        json_obj: The JSON-serializable object to write.
        file_name (str): The path to the file to write to.

    Note:
        - The file is written with UTF-8 encoding.
        - JSON is formatted with 2-space indentation for readability.
        - Non-ASCII characters are preserved (ensure_ascii=False).
    """
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def load_json(file_name):
    """Load a JSON object from a file.

    Args:
        file_name (str): The path to the JSON file to load.

    Returns:
        The deserialized JSON object, or None if the file doesn't exist.

    Note:
        The file is read with UTF-8 encoding.
    """
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


# it's dirty to type, so it's a good way to have fun
def pack_user_ass_to_openai_messages(prompt: str, generated_content: str, using_amazon_bedrock: bool):
    """Pack user prompt and assistant response into OpenAI-compatible message format.

    This function formats messages according to the API requirements of either
    Amazon Bedrock or standard OpenAI-compatible APIs.

    Args:
        prompt (str): The user's prompt text.
        generated_content (str): The assistant's generated response text.
        using_amazon_bedrock (bool): If True, formats for Amazon Bedrock API.
            If False, formats for standard OpenAI API.

    Returns:
        list[dict]: A list of message dictionaries in the appropriate format:
            - Bedrock: Content wrapped in {"text": ...} objects
            - OpenAI: Content as plain strings

    Note:
        Amazon Bedrock requires content to be wrapped in a list of dictionaries
        with a "text" key, while OpenAI uses plain strings.
    """
    if using_amazon_bedrock:
        return [
            {"role": "user", "content": [{"text": prompt}]},
            {"role": "assistant", "content": [{"text": generated_content}]},
        ]
    else:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": generated_content},
        ]


def is_float_regex(value):
    """Check if a string value matches the pattern of a floating-point number.

    Args:
        value (str): The string value to check.

    Returns:
        bool: True if the value matches a float pattern (including integers),
            False otherwise.

    Note:
        The pattern matches optional +/- sign, optional digits before decimal,
        optional decimal point, and required digits. Examples: "123", "12.34",
        "-0.5", "+123.456".
    """
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def compute_args_hash(*args):
    """Compute an MD5 hash of the given arguments.

    Args:
        *args: Variable number of arguments to hash together.

    Returns:
        str: The MD5 hash (hexadecimal) of the string representation of all args.

    Note:
        Arguments are converted to their string representation before hashing.
        Useful for creating cache keys based on function arguments.
    """
    return md5(str(args).encode()).hexdigest()


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple delimiter markers.

    This function splits a string using any of the provided markers as delimiters,
    removing empty results and trimming whitespace.

    Args:
        content (str): The string to split.
        markers (list[str]): A list of delimiter strings to split by.

    Returns:
        list[str]: A list of non-empty, stripped string segments.

    Note:
        - If markers is empty, returns the original content in a list.
        - All markers are regex-escaped to handle special characters.
        - Empty segments and whitespace-only segments are filtered out.
    """
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string value with double quotes.

    Args:
        content (Any): The content to enclose. Can be any type.

    Returns:
        str: The content as a string enclosed in double quotes. Numbers are
            returned as strings without quotes.

    Note:
        - Numbers (int, float, etc.) are returned as strings without quotes.
        - Existing single or double quotes at the beginning/end are removed
          before adding new double quotes.
        - Leading and trailing whitespace is stripped.
    """
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'


def list_of_list_to_csv(data: list[list]):
    """Convert a list of lists into a CSV-formatted string.

    Args:
        data (list[list]): A 2D list where each inner list represents a row.

    Returns:
        str: A CSV-formatted string with comma-tab delimiters and quoted values.

    Note:
        - Each value is enclosed in quotes using enclose_string_with_quotes().
        - Values within a row are separated by ",\t" (comma + tab).
        - Rows are separated by newlines.
        - Numbers are not quoted, but strings are.
    """
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )


# -----------------------------------------------------------------------------------
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes and control characters.

    This function sanitizes text by unescaping HTML entities and removing
    control characters. It's adapted from the official Microsoft GraphRAG
    implementation.

    Args:
        input (Any): The input to clean. Typically a string, but can be any type.

    Returns:
        str: The cleaned string with HTML entities unescaped and control
            characters removed. If input is not a string, returns it unchanged.

    Note:
        - Non-string inputs are returned as-is without processing.
        - HTML entities (like &amp;, &lt;, etc.) are converted to their
          corresponding characters.
        - Control characters (ASCII 0x00-0x1f and 0x7f-0x9f) are removed.
        - Leading and trailing whitespace is stripped before processing.

    Reference:
        https://github.com/microsoft/graphrag
        https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    """
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


# Utils types -----------------------------------------------------------------------
@dataclass
class EmbeddingFunc:
    """A wrapper dataclass for embedding functions with metadata.

    This dataclass encapsulates an embedding function along with its configuration
    parameters, providing a standardized interface for different embedding models.

    Attributes:
        embedding_dim (int): The dimensionality of the embedding vectors produced
            by this function.
        max_token_size (int): The maximum number of tokens this embedding function
            can handle in a single call.
        func (callable): The actual async embedding function that takes text and
            returns embedding vectors.

    Note:
        The __call__ method allows instances to be used as async functions,
        delegating to the wrapped func.
    """
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        """Call the wrapped embedding function.

        Args:
            *args: Positional arguments to pass to the embedding function.
            **kwargs: Keyword arguments to pass to the embedding function.

        Returns:
            np.ndarray: The embedding vector(s) as a NumPy array.
        """
        return await self.func(*args, **kwargs)


# Decorators ------------------------------------------------------------------------
def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Decorator to limit the number of concurrent async function calls.

    This decorator restricts the maximum number of concurrent executions of an
    async function, implementing a simple semaphore-like behavior without using
    asyncio.Semaphore (to avoid nest-asyncio issues).

    Args:
        max_size (int): The maximum number of concurrent calls allowed.
        waitting_time (float, optional): The sleep time in seconds between
            checks when waiting for an available slot. Defaults to 0.0001.

    Returns:
        callable: A decorator function that wraps async functions with
            concurrency limiting.

    Example:
        @limit_async_func_call(max_size=5)
        async def my_async_func():
            # This function will have at most 5 concurrent executions
            pass

    Note:
        - Uses a polling approach with asyncio.sleep() instead of asyncio.Semaphore
          to avoid compatibility issues with nest-asyncio.
        - The __current_size counter tracks active concurrent calls.
        - Functions wait in a loop until a slot becomes available.
    """

    def final_decro(func):
        """Inner decorator that wraps the actual function.

        Note:
            Not using async.Semaphore to avoid nest-asyncio issues.
        """
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Decorator to wrap an embedding function with metadata attributes.

    This decorator creates an EmbeddingFunc instance from a function and its
    configuration parameters, providing a standardized interface for embedding
    functions.

    Args:
        **kwargs: Keyword arguments to pass to EmbeddingFunc constructor.
            Should include:
            - embedding_dim (int): Dimension of embeddings
            - max_token_size (int): Maximum token size
            - func (callable): The embedding function (added automatically)

    Returns:
        callable: A decorator function that wraps the function into an
            EmbeddingFunc instance.

    Example:
        @wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=512)
        async def my_embedding_func(texts):
            # Function implementation
            return embeddings

    Note:
        The wrapped function becomes an EmbeddingFunc instance, which can be
        called like a normal async function while also carrying metadata about
        embedding dimensions and token limits.
    """

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro
