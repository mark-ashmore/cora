"""Main pipeline for training and updating the classifier model."""

import csv
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import spacy
from spacy.matcher import PhraseMatcher

from main_pipeline.expand_training import get_training
from utils import paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
f_handler = logging.FileHandler(paths.MAIN_PIPELINE_LOG)
f_format = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(filename)s | %(funcName)s | "
    "%(lineno)s | %(message)s"
)
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)
logger.setLevel(level=logging.INFO)


def prepare_training(
    entities_path: Path, training_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the training data for the model."""
    # TODO: Add in real test data or use train_test_split to randomly split the
    # training data into training and validation sets.
    texts, labels = get_training(training_path, entities_path)
    train_df = pd.DataFrame({"text": texts, "label": labels})
    test_df = pd.DataFrame({"text": texts, "label": labels})
    logger.info("Training and test data prepared")
    logger.info("Training data shape: %s", str(train_df.shape))
    logger.info("Test data shape: %s", str(test_df.shape))
    return train_df, test_df


def get_label_map(df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    """Get a label map for the training data."""
    label2id = {label: i for i, label in enumerate(df["label"].unique())}
    id2label = {i: label for i, label in enumerate(df["label"].unique())}
    logger.info("Label map created")
    logger.info("Label map: %s", str(label2id))
    return label2id, id2label


def convert_labels(
    train_df: pd.DataFrame, test_df: pd.DataFrame, label2id: dict[str, int]
) -> None:
    """Convert the labels to integers."""
    for label in label2id:
        train_df.loc[train_df["label"] == label, "label"] = label2id[label]
        test_df.loc[test_df["label"] == label, "label"] = label2id[label]
    logger.info("Labels converted in training and test data dataframes")


def tokenize_data(df: pd.DataFrame, tokenizer: BertTokenizer):
    encoded_dict = tokenizer(
        df["text"].tolist(), padding=True, truncation=True, return_tensors="pt"
    )
    labels = torch.tensor(df["label"].tolist())
    return encoded_dict["input_ids"], encoded_dict["attention_mask"], labels


def initialize_scheduler(
    optimizer: AdamW, train_dataloader: DataLoader, epochs: int
) -> lr_scheduler.LambdaLR:
    """Initialize the learning rate scheduler."""
    total_steps = len(train_dataloader) * epochs
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )


def train_bert_model(
    epochs: int,
    model: BertForSequenceClassification,
    train_dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: lr_scheduler.LambdaLR,
    device: torch.device,
) -> None:
    """Train the model."""
    logger.info("Training started.")
    for _ in range(epochs):
        logger.info(f"Starting epoch {_ + 1}/{epochs}")
        model.train()
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()


def evaluate_bert_model(
    model: BertForSequenceClassification,
    test_dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Evaluate the model."""
    logger.info("Evaluating test data.")
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }
            outputs = model(**inputs)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = batch[2].to("cpu").numpy()
            predictions.extend(logits.argmax(axis=-1))
            true_labels.extend(label_ids)
    return predictions, true_labels


def update_bert_model(
    entities_path: Path,
    training_path: Path,
    report_path: Path,
    bert_path: Path,
    tokenizer_path: Path,
    labels_path: Path,
) -> None:
    """Update the BERT model."""
    train_df, test_df = prepare_training(entities_path, training_path)
    label2id, id2label = get_label_map(train_df)
    convert_labels(train_df, test_df, label2id)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    train_input_ids, train_attention_mask, train_labels = tokenize_data(
        train_df, tokenizer
    )
    test_input_ids, test_attention_mask, test_labels = tokenize_data(test_df, tokenizer)
    # DataLoaders for efficient batching
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    # Model setup
    model: BertForSequenceClassification | Any = (
        BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=len(train_df["label"].unique())
        )
    )
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 4
    scheduler = initialize_scheduler(optimizer, train_dataloader, epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_bert_model(epochs, model, train_dataloader, optimizer, scheduler, device)
    predictions, true_labels = evaluate_bert_model(model, test_dataloader, device)
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(
        true_labels,
        predictions,
        target_names=list(id2label.values()),
        output_dict=True,
        zero_division=0,
    )
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    pd.DataFrame(report).transpose().to_csv(report_path)
    model.save_pretrained(bert_path)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Model saved to {bert_path}")
    logger.info(f"Tokenizer saved to {tokenizer_path}")
    with labels_path.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "id"])
        for label, id in label2id.items():
            writer.writerow([label, id])
    logger.info(f"Labels saved to {labels_path}")


def prepare_entity_terms(entities_path: Path) -> list[tuple[tuple[str], str]]:
    """Prepare entity terms for training."""
    entity_terms = []
    for entity_file in entities_path.iterdir():
        if entity_file.is_file():
            with entity_file.open(mode="r", encoding="utf-8") as source:
                entity_dict = json.load(source)
            for label in entity_dict["labels"]:
                synonyms = []
                eid = label["label"]
                for custom_entities in label["custom_entities"].values():
                    synonyms.extend(custom_entities)
                entity_terms.append((tuple(synonyms), eid))
    return entity_terms


def update_entity_model(entities_path: Path, spacy_model_path: Path) -> None:
    """Update the entity model."""
    nlp = spacy.load("en_core_web_sm")
    matcher = PhraseMatcher(nlp.vocab)
    entity_terms = prepare_entity_terms(entities_path)
    for term in entity_terms:
        patterns = [nlp.make_doc(text) for text in term[0]]
        matcher.add(term[1], patterns)
    nlp.to_disk((spacy_model_path / "nlp"))
    pickle.dump(matcher, (spacy_model_path / "matcher.pkl").open(mode="wb"))
    logger.info("Entity model updated at %s", str(datetime.now().strftime("%H:%M:%S")))


def update_agent_models():
    """Update Bert and Spacy models for agent use."""
    update_bert_model(
        paths.ENTITIES_PATH,
        paths.MODEL_TRAINING_PATH,
        paths.MODEL_REPORT_PATH,
        paths.BERT_PATH,
        paths.TOKENIZER_PATH,
        paths.LABELS_PATH,
    )
    update_entity_model(paths.ENTITIES_PATH, paths.ENTITY_MODEL_PATH)
