"""
Text Splitter Module for nano-graphrag

This module provides token-based text splitting functionality with support for
semantic separators (like newlines, periods, etc.). The splitter ensures chunks
respect token size limits while attempting to preserve natural text boundaries.
"""

from typing import List, Optional, Union, Literal

class SeparatorSplitter:
    """
    Token-based text splitter that respects semantic separators.

    This class splits a sequence of tokens into chunks based on:
    1. Natural separators (e.g., newlines, periods) to preserve semantic boundaries
    2. Maximum chunk size constraints
    3. Overlapping tokens between chunks for better context continuity

    The splitter first attempts to split on separators, then merges splits to fit
    within the chunk size limit, and finally enforces overlap between consecutive chunks.
    """
    def __init__(
        self,
        separators: Optional[List[List[int]]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = "end",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: callable = len,
    ):
        """
        Initialize the SeparatorSplitter.

        Args:
            separators: List of token sequences to use as split points (e.g., newline tokens)
            keep_separator: Whether/how to keep separators in chunks:
                - True or "end": Append separator to the end of the preceding chunk
                - "start": Prepend separator to the start of the following chunk
                - False: Discard separators
            chunk_size: Maximum number of tokens per chunk (default: 4000)
            chunk_overlap: Number of tokens to overlap between consecutive chunks (default: 200)
            length_function: Function to calculate length of token sequences (default: len)
        """
        self._separators = separators or []
        self._keep_separator = keep_separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    def split_tokens(self, tokens: List[int]) -> List[List[int]]:
        """
        Split a token sequence into chunks.

        Main entry point that orchestrates the splitting process:
        1. Split on separators
        2. Merge splits to respect chunk size
        3. Enforce overlap between chunks

        Args:
            tokens: List of token IDs to split

        Returns:
            List of token ID lists, where each inner list is a chunk
        """
        splits = self._split_tokens_with_separators(tokens)
        return self._merge_splits(splits)

    def _split_tokens_with_separators(self, tokens: List[int]) -> List[List[int]]:
        """
        Split tokens on separator boundaries.

        Scans through the token sequence and creates splits at each separator occurrence,
        handling separator retention based on the keep_separator setting.

        Args:
            tokens: List of token IDs to split

        Returns:
            List of token splits (before merging/overlap enforcement)
        """
        splits = []
        current_split = []
        i = 0
        while i < len(tokens):
            separator_found = False
            for separator in self._separators:
                if tokens[i:i+len(separator)] == separator:
                    if self._keep_separator in [True, "end"]:
                        current_split.extend(separator)
                    if current_split:
                        splits.append(current_split)
                        current_split = []
                    if self._keep_separator == "start":
                        current_split.extend(separator)
                    i += len(separator)
                    separator_found = True
                    break
            if not separator_found:
                current_split.append(tokens[i])
                i += 1
        if current_split:
            splits.append(current_split)
        return [s for s in splits if s]

    def _merge_splits(self, splits: List[List[int]]) -> List[List[int]]:
        """
        Merge splits to respect chunk size constraints.

        Combines smaller splits until they approach the chunk_size limit,
        then handles overlap enforcement if configured.

        Args:
            splits: List of token splits from separator-based splitting

        Returns:
            List of merged chunks with size constraints enforced
        """
        if not splits:
            return []

        merged_splits = []
        current_chunk = []

        for split in splits:
            if not current_chunk:
                current_chunk = split
            elif self._length_function(current_chunk) + self._length_function(split) <= self._chunk_size:
                current_chunk.extend(split)
            else:
                merged_splits.append(current_chunk)
                current_chunk = split

        if current_chunk:
            merged_splits.append(current_chunk)

        if len(merged_splits) == 1 and self._length_function(merged_splits[0]) > self._chunk_size:
            return self._split_chunk(merged_splits[0])

        if self._chunk_overlap > 0:
            return self._enforce_overlap(merged_splits)
        
        return merged_splits

    def _split_chunk(self, chunk: List[int]) -> List[List[int]]:
        """
        Split a single oversized chunk into smaller overlapping chunks.

        Used when a chunk exceeds the maximum size and cannot be split by separators.
        Creates overlapping chunks using a sliding window approach.

        Args:
            chunk: Token list that exceeds chunk_size

        Returns:
            List of smaller overlapping chunks
        """
        result = []
        for i in range(0, len(chunk), self._chunk_size - self._chunk_overlap):
            new_chunk = chunk[i:i + self._chunk_size]
            if len(new_chunk) > self._chunk_overlap:  # Only add if chunk length exceeds overlap
                result.append(new_chunk)
        return result

    def _enforce_overlap(self, chunks: List[List[int]]) -> List[List[int]]:
        """
        Ensure overlap between consecutive chunks.

        Prepends tokens from the end of the previous chunk to the beginning of
        each subsequent chunk, ensuring context continuity across chunk boundaries.

        Args:
            chunks: List of chunks to add overlap to

        Returns:
            List of chunks with enforced overlap
        """
        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                overlap = chunks[i-1][-self._chunk_overlap:]
                new_chunk = overlap + chunk
                if self._length_function(new_chunk) > self._chunk_size:
                    new_chunk = new_chunk[:self._chunk_size]
                result.append(new_chunk)
        return result

