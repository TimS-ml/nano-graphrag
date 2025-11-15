"""Entity extraction module for knowledge graph construction.

This module provides components for extracting entities and relationships from text documents
using DSPy-based language models. It implements a sophisticated extraction pipeline with
optional self-refinement capabilities to improve the quality of extracted knowledge.

The module supports:
    - Extraction of typed entities (people, organizations, locations, etc.)
    - Relationship extraction with different orders (direct, second-order, third-order)
    - Self-refinement through critique and refinement iterations
    - Exception handling for robust extraction

Typical usage example:
    extractor = TypedEntityRelationshipExtractor(
        lm=my_language_model,
        self_refine=True,
        num_refine_turns=2
    )
    result = extractor(input_text="Your text here")
    entities = result.entities
    relationships = result.relationships
"""

import dspy
from pydantic import BaseModel, Field
from nano_graphrag._utils import clean_str
from nano_graphrag._utils import logger


# Obtained from:
# https://github.com/SciPhi-AI/R2R/blob/6e958d1e451c1cb10b6fc868572659785d1091cb/r2r/providers/prompts/defaults.jsonl
ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "DATE",
    "TIME",
    "MONEY",
    "PERCENTAGE",
    "PRODUCT",
    "EVENT",
    "LANGUAGE",
    "NATIONALITY",
    "RELIGION",
    "TITLE",
    "PROFESSION",
    "ANIMAL",
    "PLANT",
    "DISEASE",
    "MEDICATION",
    "CHEMICAL",
    "MATERIAL",
    "COLOR",
    "SHAPE",
    "MEASUREMENT",
    "WEATHER",
    "NATURAL_DISASTER",
    "AWARD",
    "LAW",
    "CRIME",
    "TECHNOLOGY",
    "SOFTWARE",
    "HARDWARE",
    "VEHICLE",
    "FOOD",
    "DRINK",
    "SPORT",
    "MUSIC_GENRE",
    "INSTRUMENT",
    "ARTWORK",
    "BOOK",
    "MOVIE",
    "TV_SHOW",
    "ACADEMIC_SUBJECT",
    "SCIENTIFIC_THEORY",
    "POLITICAL_PARTY",
    "CURRENCY",
    "STOCK_SYMBOL",
    "FILE_TYPE",
    "PROGRAMMING_LANGUAGE",
    "MEDICAL_PROCEDURE",
    "CELESTIAL_BODY",
]


class Entity(BaseModel):
    """Represents an extracted entity from text with its type and metadata.

    An Entity captures a named entity (person, organization, location, etc.) identified
    in the source text along with its type, detailed description, and importance score.
    This model is used as the output format for entity extraction processes.

    Attributes:
        entity_name: The name of the entity as it appears in the text.
        entity_type: The classification type of the entity (e.g., PERSON, ORGANIZATION).
        description: A detailed and comprehensive description of the entity, including
            its role, significance, characteristics, relationships, and any relevant
            historical or contextual information.
        importance_score: A normalized score (0.0 to 1.0) indicating the entity's
            importance in the context, where 1.0 represents the highest importance.
    """

    entity_name: str = Field(..., description="The name of the entity.")
    entity_type: str = Field(..., description="The type of the entity.")
    description: str = Field(
        ..., description="The description of the entity, in details and comprehensive."
    )
    importance_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Importance score of the entity. Should be between 0 and 1 with 1 being the most important.",
    )

    def to_dict(self):
        """Converts the entity to a dictionary with cleaned and normalized values.

        This method prepares the entity data for storage or further processing by:
        - Converting entity names and types to uppercase
        - Cleaning strings to remove extra whitespace and special characters
        - Ensuring the importance score is a float type

        Returns:
            dict: A dictionary containing the cleaned entity data with keys:
                - entity_name: Uppercase, cleaned entity name
                - entity_type: Uppercase, cleaned entity type
                - description: Cleaned description text
                - importance_score: Float value between 0.0 and 1.0
        """
        return {
            "entity_name": clean_str(self.entity_name.upper()),
            "entity_type": clean_str(self.entity_type.upper()),
            "description": clean_str(self.description),
            "importance_score": float(self.importance_score),
        }


class Relationship(BaseModel):
    """Represents a relationship between two entities in the knowledge graph.

    A Relationship captures the connection between a source entity and a target entity,
    including the nature of their relationship, its strength, and its order (direct vs.
    indirect connections). This model supports multi-hop relationship reasoning.

    Attributes:
        src_id: The name/identifier of the source entity in the relationship.
        tgt_id: The name/identifier of the target entity in the relationship.
        description: A detailed and comprehensive description of the relationship,
            including its nature (e.g., familial, professional, causal), impact,
            significance, historical context, evolution over time, and any notable
            events or actions resulting from the relationship.
        weight: A normalized score (0.0 to 1.0) indicating the strength of the
            relationship, where 1.0 represents the strongest possible connection.
        order: The degree of separation between entities:
            - 1: Direct/immediate relationships
            - 2: Second-order/indirect relationships
            - 3: Third-order/further indirect relationships
    """

    src_id: str = Field(..., description="The name of the source entity.")
    tgt_id: str = Field(..., description="The name of the target entity.")
    description: str = Field(
        ...,
        description="The description of the relationship between the source and target entity, in details and comprehensive.",
    )
    weight: float = Field(
        ...,
        ge=0,
        le=1,
        description="The weight of the relationship. Should be between 0 and 1 with 1 being the strongest relationship.",
    )
    order: int = Field(
        ...,
        ge=1,
        le=3,
        description="The order of the relationship. 1 for direct relationships, 2 for second-order, 3 for third-order.",
    )

    def to_dict(self):
        """Converts the relationship to a dictionary with cleaned and normalized values.

        This method prepares the relationship data for storage or further processing by:
        - Converting entity IDs to uppercase for consistency
        - Cleaning strings to remove extra whitespace and special characters
        - Ensuring numeric values are of the correct type

        Returns:
            dict: A dictionary containing the cleaned relationship data with keys:
                - src_id: Uppercase, cleaned source entity name
                - tgt_id: Uppercase, cleaned target entity name
                - description: Cleaned description text
                - weight: Float value between 0.0 and 1.0
                - order: Integer value (1, 2, or 3)
        """
        return {
            "src_id": clean_str(self.src_id.upper()),
            "tgt_id": clean_str(self.tgt_id.upper()),
            "description": clean_str(self.description),
            "weight": float(self.weight),
            "order": int(self.order),
        }


class CombinedExtraction(dspy.Signature):
    """
    Given a text document that is potentially relevant to this activity and a list of entity types,
    identify all entities of those types from the text and all relationships among the identified entities.

    Entity Guidelines:
    1. Each entity name should be an actual atomic word from the input text.
    2. Avoid duplicates and generic terms.
    3. Make sure descriptions are detailed and comprehensive. Use multiple complete sentences for each point below:
        a). The entity's role or significance in the context
        b). Key attributes or characteristics
        c). Relationships to other entities (if applicable)
        d). Historical or cultural relevance (if applicable)
        e). Any notable actions or events associated with the entity
    4. All entity types from the text must be included.
    5. IMPORTANT: Only use entity types from the provided 'entity_types' list. Do not introduce new entity types.

    Relationship Guidelines:
    1. Make sure relationship descriptions are detailed and comprehensive. Use multiple complete sentences for each point below:
        a). The nature of the relationship (e.g., familial, professional, causal)
        b). The impact or significance of the relationship on both entities
        c). Any historical or contextual information relevant to the relationship
        d). How the relationship evolved over time (if applicable)
        e). Any notable events or actions that resulted from this relationship
    2. Include direct relationships (order 1) as well as higher-order relationships (order 2 and 3):
        a). Direct relationships: Immediate connections between entities.
        b). Second-order relationships: Indirect effects or connections that result from direct relationships.
        c). Third-order relationships: Further indirect effects that result from second-order relationships.
    3. The "src_id" and "tgt_id" fields must exactly match entity names from the extracted entities list.
    """

    input_text: str = dspy.InputField(
        desc="The text to extract entities and relationships from."
    )
    entity_types: list[str] = dspy.InputField(
        desc="List of entity types used for extraction."
    )
    entities: list[Entity] = dspy.OutputField(
        desc="List of entities extracted from the text and the entity types."
    )
    relationships: list[Relationship] = dspy.OutputField(
        desc="List of relationships extracted from the text and the entity types."
    )


class CritiqueCombinedExtraction(dspy.Signature):
    """
    Critique the current extraction of entities and relationships from a given text.
    Focus on completeness, accuracy, and adherence to the provided entity types and extraction guidelines.

    Critique Guidelines:
    1. Evaluate if all relevant entities from the input text are captured and correctly typed.
    2. Check if entity descriptions are comprehensive and follow the provided guidelines.
    3. Assess the completeness of relationship extractions, including higher-order relationships.
    4. Verify that relationship descriptions are detailed and follow the provided guidelines.
    5. Identify any inconsistencies, errors, or missed opportunities in the current extraction.
    6. Suggest specific improvements or additions to enhance the quality of the extraction.
    """

    input_text: str = dspy.InputField(
        desc="The original text from which entities and relationships were extracted."
    )
    entity_types: list[str] = dspy.InputField(
        desc="List of valid entity types for this extraction task."
    )
    current_entities: list[Entity] = dspy.InputField(
        desc="List of currently extracted entities to be critiqued."
    )
    current_relationships: list[Relationship] = dspy.InputField(
        desc="List of currently extracted relationships to be critiqued."
    )
    entity_critique: str = dspy.OutputField(
        desc="Detailed critique of the current entities, highlighting areas for improvement for completeness and accuracy.."
    )
    relationship_critique: str = dspy.OutputField(
        desc="Detailed critique of the current relationships, highlighting areas for improvement for completeness and accuracy.."
    )


class RefineCombinedExtraction(dspy.Signature):
    """
    Refine the current extraction of entities and relationships based on the provided critique.
    Improve completeness, accuracy, and adherence to the extraction guidelines.

    Refinement Guidelines:
    1. Address all points raised in the entity and relationship critiques.
    2. Add missing entities and relationships identified in the critique.
    3. Improve entity and relationship descriptions as suggested.
    4. Ensure all refinements still adhere to the original extraction guidelines.
    5. Maintain consistency between entities and relationships during refinement.
    6. Focus on enhancing the overall quality and comprehensiveness of the extraction.
    """

    input_text: str = dspy.InputField(
        desc="The original text from which entities and relationships were extracted."
    )
    entity_types: list[str] = dspy.InputField(
        desc="List of valid entity types for this extraction task."
    )
    current_entities: list[Entity] = dspy.InputField(
        desc="List of currently extracted entities to be refined."
    )
    current_relationships: list[Relationship] = dspy.InputField(
        desc="List of currently extracted relationships to be refined."
    )
    entity_critique: str = dspy.InputField(
        desc="Detailed critique of the current entities to guide refinement."
    )
    relationship_critique: str = dspy.InputField(
        desc="Detailed critique of the current relationships to guide refinement."
    )
    refined_entities: list[Entity] = dspy.OutputField(
        desc="List of refined entities, addressing the entity critique and improving upon the current entities."
    )
    refined_relationships: list[Relationship] = dspy.OutputField(
        desc="List of refined relationships, addressing the relationship critique and improving upon the current relationships."
    )


class TypedEntityRelationshipExtractorException(dspy.Module):
    """Exception handler wrapper for entity and relationship extraction.

    This module wraps a predictor to provide graceful exception handling during
    the extraction process. When specified exception types occur, it returns empty
    results instead of propagating the exception, allowing the extraction pipeline
    to continue processing other inputs.

    This is particularly useful for handling validation errors or LLM output
    parsing failures without interrupting batch processing.

    Attributes:
        predictor: The underlying DSPy module that performs the actual extraction.
        exception_types: Tuple of exception types to catch and handle gracefully.
    """

    def __init__(
        self,
        predictor: dspy.Module,
        exception_types: tuple[type[Exception]] = (Exception,),
    ):
        """Initializes the exception handler wrapper.

        Args:
            predictor: The DSPy module to wrap with exception handling.
            exception_types: Tuple of exception types to catch and convert to empty
                results. Other exceptions will be re-raised. Defaults to (Exception,).
        """
        super().__init__()
        self.predictor = predictor
        self.exception_types = exception_types

    def copy(self):
        """Creates a shallow copy of this exception handler.

        Returns:
            TypedEntityRelationshipExtractorException: A new instance wrapping the
                same predictor with the same exception handling configuration.
        """
        return TypedEntityRelationshipExtractorException(self.predictor)

    def forward(self, **kwargs):
        """Executes the wrapped predictor with exception handling.

        Attempts to run the predictor with the provided arguments. If an exception
        of the specified types occurs, returns an empty prediction instead of failing.
        All other exceptions are re-raised.

        Args:
            **kwargs: Arbitrary keyword arguments to pass to the wrapped predictor.

        Returns:
            dspy.Prediction: The predictor's result, or an empty prediction (with
                empty entities and relationships lists) if a handled exception occurs.

        Raises:
            Exception: Any exception that is not in the configured exception_types.
        """
        try:
            prediction = self.predictor(**kwargs)
            return prediction

        except Exception as e:
            if isinstance(e, self.exception_types):
                return dspy.Prediction(entities=[], relationships=[])

            raise e


class TypedEntityRelationshipExtractor(dspy.Module):
    """Main module for extracting typed entities and relationships from text.

    This module orchestrates the complete entity and relationship extraction pipeline
    using language models via DSPy. It supports basic extraction with Chain-of-Thought
    reasoning and optional self-refinement through iterative critique and improvement.

    The extraction process:
    1. Initial extraction: Identifies entities and relationships from input text
    2. (Optional) Critique: Analyzes the extraction for completeness and accuracy
    3. (Optional) Refinement: Improves the extraction based on the critique
    4. Steps 2-3 can be repeated for multiple refinement turns

    This approach helps improve extraction quality by allowing the model to review
    and enhance its own outputs.

    Attributes:
        lm: The language model to use for extraction. If None, uses the default
            from dspy.settings.
        entity_types: List of valid entity types to extract (e.g., PERSON, ORGANIZATION).
        self_refine: Whether to enable the self-refinement process.
        num_refine_turns: Number of critique-refine iterations to perform.
        extractor: The core extraction module (ChainOfThought wrapped with exception handling).
        critique: The critique module (only initialized if self_refine is True).
        refine: The refinement module (only initialized if self_refine is True).
    """

    def __init__(
        self,
        lm: dspy.LM = None,
        max_retries: int = 3,
        entity_types: list[str] = ENTITY_TYPES,
        self_refine: bool = False,
        num_refine_turns: int = 1,
    ):
        """Initializes the entity and relationship extractor.

        Args:
            lm: The DSPy language model to use for extraction. If None, the default
                language model from dspy.settings will be used. Defaults to None.
            max_retries: Maximum number of retry attempts for each LLM call when
                parsing or validation fails. Defaults to 3.
            entity_types: List of entity type strings to extract. Entities not matching
                these types will be filtered out. Defaults to the global ENTITY_TYPES list
                which includes 60+ common entity types.
            self_refine: Whether to enable self-refinement through critique and
                refinement iterations. This improves quality but increases LLM calls.
                Defaults to False.
            num_refine_turns: Number of critique-and-refine iterations to perform
                when self_refine is True. More turns may improve quality but increase
                cost and latency. Defaults to 1.
        """
        super().__init__()
        self.lm = lm
        self.entity_types = entity_types
        self.self_refine = self_refine
        self.num_refine_turns = num_refine_turns

        self.extractor = dspy.ChainOfThought(
            signature=CombinedExtraction, max_retries=max_retries
        )
        self.extractor = TypedEntityRelationshipExtractorException(
            self.extractor, exception_types=(ValueError,)
        )

        if self.self_refine:
            self.critique = dspy.ChainOfThought(
                signature=CritiqueCombinedExtraction, max_retries=max_retries
            )
            self.refine = dspy.ChainOfThought(
                signature=RefineCombinedExtraction, max_retries=max_retries
            )

    def forward(self, input_text: str) -> dspy.Prediction:
        """Extracts entities and relationships from the provided text.

        This is the main entry point for the extraction process. It performs the following:
        1. Runs the initial entity and relationship extraction using Chain-of-Thought
        2. If self_refine is enabled, iteratively critiques and refines the extraction
        3. Converts the final Entity and Relationship objects to dictionaries
        4. Returns the results in a DSPy Prediction object

        The method uses the configured language model and entity types, and applies
        the self-refinement process if enabled during initialization.

        Args:
            input_text: The text document from which to extract entities and relationships.
                This can be a paragraph, article, or any text content containing
                entities and their relationships.

        Returns:
            dspy.Prediction: A prediction object containing two fields:
                - entities: List of dictionaries, each representing an extracted entity
                  with keys: entity_name, entity_type, description, importance_score
                - relationships: List of dictionaries, each representing an extracted
                  relationship with keys: src_id, tgt_id, description, weight, order

        Note:
            If extraction fails with a ValueError (e.g., due to LLM output parsing issues),
            the exception handler will return empty lists for both entities and relationships
            rather than raising an exception. This allows batch processing to continue.

            Debug logging is enabled during refinement to track the number of entities
            and relationships before and after each refinement turn.
        """
        with dspy.context(lm=self.lm if self.lm is not None else dspy.settings.lm):
            extraction_result = self.extractor(
                input_text=input_text, entity_types=self.entity_types
            )

            current_entities: list[Entity] = extraction_result.entities
            current_relationships: list[Relationship] = extraction_result.relationships

            if self.self_refine:
                for _ in range(self.num_refine_turns):
                    critique_result = self.critique(
                        input_text=input_text,
                        entity_types=self.entity_types,
                        current_entities=current_entities,
                        current_relationships=current_relationships,
                    )
                    refined_result = self.refine(
                        input_text=input_text,
                        entity_types=self.entity_types,
                        current_entities=current_entities,
                        current_relationships=current_relationships,
                        entity_critique=critique_result.entity_critique,
                        relationship_critique=critique_result.relationship_critique,
                    )
                    logger.debug(
                        f"entities: {len(current_entities)} | refined_entities: {len(refined_result.refined_entities)}"
                    )
                    logger.debug(
                        f"relationships: {len(current_relationships)} | refined_relationships: {len(refined_result.refined_relationships)}"
                    )
                    current_entities = refined_result.refined_entities
                    current_relationships = refined_result.refined_relationships

        entities = [entity.to_dict() for entity in current_entities]
        relationships = [
            relationship.to_dict() for relationship in current_relationships
        ]

        return dspy.Prediction(entities=entities, relationships=relationships)
