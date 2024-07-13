import json
from pathlib import Path
from typing import Any


def collect_entity_synonyms(entity: str, entites_path: Path) -> list[str]:
    """Locate an entity by the path name and return all synonyms.

    Parameters:
    - entity: Name of entity.
    - entities_path: Path object for entities directory.

    Returns:
    - List of entity synonyms.
    """
    entity_file = entites_path / f"{entity}.json"
    with entity_file.open(mode="r", encoding="utf-8") as source:
        entity_dict = json.load(source)
    synonyms = []
    if entity_dict["id"] == entity:
        for label in entity_dict["labels"]:
            for custom_entities in label["custom_entities"].values():
                synonyms.extend(custom_entities)
    return synonyms


def join_training_text(patterns: list[dict], entities_path: Path) -> list[str]:
    """Join patterns into a training string.

    Parameters:
    - patterns: A list of training patterns containing:
      - 'entity': The name of annotation entity.
      - 'text': The text within the pattern if applicable.
    - entities_path: Path object for entities directory.

    Returns:
    - List of string training patterns.
    """
    text_strings = []
    for pattern in patterns:
        if "entity" in pattern:
            synonyms = collect_entity_synonyms(pattern["entity"], entities_path)
            extended_strings = []
            for s in synonyms:
                if text_strings:
                    for text in text_strings:
                        extended_strings.append(text + s)
                else:
                    extended_strings.append(s)
            if len(text_strings) == 1:
                text_strings = [s for s in extended_strings]
            else:
                text_strings.extend(extended_strings)
        else:
            if text_strings:
                text_strings = [text + pattern["text"] for text in text_strings]
            else:
                text_strings.append(pattern["text"])
    return text_strings


def expand_training(
    training: dict[str, list[dict[str, Any]]], entities_path: Path
) -> tuple[list[str], list[str]]:
    """Expand training phrases to have instances of each entity pattern.

    Parameters:
    - training: a dictionary of training extracted from JSON source files.
    - entities_path: Path object for entities directory.

    Returns
    - A tuple of training patterns and intents within lists.
    """
    intents = []
    patterns = []
    utterances = training["utterances"]
    for utterance in utterances:
        training_strings = join_training_text(utterance["patterns"], entities_path)
        patterns.extend(training_strings)
        intents.extend([utterance["intent"]] * len(training_strings))
    return patterns, intents


def get_training(training_dir: Path, entities_path: Path) -> tuple[list, list]:
    """Get training examples in a sepcified path.

    Parameters:
    - training_dir: Path object for the training examples directory.
    - entities_dir: Path object for the entities directory.

    Returns:
    - Tuple of two lists containing training patterns and their ground truths.
    """
    all_patterns = []
    all_intents = []

    for file in training_dir.iterdir():
        with file.open(mode="r", encoding="utf-8") as training_file:
            training = json.load(training_file)
            patterns, intents = expand_training(training, entities_path)
            all_patterns.extend(patterns)
            all_intents.extend(intents)
    return all_patterns, all_intents
