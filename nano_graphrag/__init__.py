"""
nano-graphrag: A lightweight implementation of GraphRAG (Graph-based Retrieval-Augmented Generation)

This package provides a simplified implementation of Microsoft's GraphRAG approach for
knowledge graph construction and retrieval-augmented generation. It enables building
knowledge graphs from text documents and querying them using various search strategies.

Main Components:
    - GraphRAG: Main class for building and querying knowledge graphs
    - QueryParam: Configuration for query operations

For more information and examples, visit: https://github.com/gusye1234/nano-graphrag
"""

from .graphrag import GraphRAG, QueryParam

__version__ = "0.0.8.2"
__author__ = "Jianbai Ye"
__url__ = "https://github.com/gusye1234/nano-graphrag"

# Note: 'dp' in variable names stands for "data pack"
