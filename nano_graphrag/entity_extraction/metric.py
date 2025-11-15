"""
Evaluation Metrics for Entity Extraction

This module provides evaluation metrics for assessing the quality of entity and
relationship extraction. The metrics are designed to work with DSPy's evaluation
framework and can be used for training and optimizing extraction models.
"""

import dspy
from nano_graphrag.entity_extraction.module import Relationship


class AssessRelationships(dspy.Signature):
    """
    Assess the similarity between gold and predicted relationships:
    1. Match relationships based on src_id and tgt_id pairs, allowing for slight variations in entity names.
    2. For matched pairs, compare:
       a) Description similarity (semantic meaning)
       b) Weight similarity
       c) Order similarity
    3. Consider unmatched relationships as penalties.
    4. Aggregate scores, accounting for precision and recall.
    5. Return a final similarity score between 0 (no similarity) and 1 (perfect match).

    Key considerations:
    - Prioritize matching based on entity pairs over exact string matches.
    - Use semantic similarity for descriptions rather than exact matches.
    - Weight the importance of different aspects (e.g., entity matching, description, weight, order).
    - Balance the impact of matched and unmatched relationships in the final score.
    """

    gold_relationships: list[Relationship] = dspy.InputField(
        desc="The gold-standard relationships to compare against."
    )
    predicted_relationships: list[Relationship] = dspy.InputField(
        desc="The predicted relationships to compare against the gold-standard relationships."
    )
    similarity_score: float = dspy.OutputField(
        desc="Similarity score between 0 and 1, with 1 being the highest similarity."
    )


def relationships_similarity_metric(
    gold: dspy.Example, pred: dspy.Prediction, trace=None
) -> float:
    """
    Compute similarity score between gold and predicted relationships.

    Uses an LLM-based approach (via DSPy ChainOfThought) to assess the similarity
    between ground truth and predicted relationships, considering entity pairs,
    descriptions, weights, and orders.

    Args:
        gold: DSPy Example containing gold-standard relationships
        pred: DSPy Prediction containing predicted relationships
        trace: Optional trace for DSPy debugging

    Returns:
        float: Similarity score between 0 and 1, where 1 is perfect match
    """
    model = dspy.ChainOfThought(AssessRelationships)
    gold_relationships = [Relationship(**item) for item in gold["relationships"]]
    predicted_relationships = [Relationship(**item) for item in pred["relationships"]]
    similarity_score = float(
        model(
            gold_relationships=gold_relationships,
            predicted_relationships=predicted_relationships,
        ).similarity_score
    )
    return similarity_score


def entity_recall_metric(
    gold: dspy.Example, pred: dspy.Prediction, trace=None
) -> float:
    """
    Compute recall score for entity extraction.

    Calculates what proportion of gold-standard entities were successfully
    extracted by the model. Recall = TP / (TP + FN), where TP is true positives
    and FN is false negatives.

    Args:
        gold: DSPy Example containing gold-standard entities
        pred: DSPy Prediction containing predicted entities
        trace: Optional trace for DSPy debugging

    Returns:
        float: Recall score between 0 and 1, where 1 means all gold entities were found
    """
    true_set = set(item["entity_name"] for item in gold["entities"])
    pred_set = set(item["entity_name"] for item in pred["entities"])
    true_positives = len(pred_set.intersection(true_set))
    false_negatives = len(true_set - pred_set)
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    return recall
