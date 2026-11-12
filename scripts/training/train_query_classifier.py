"""Train the query-intent classifier (BERT [CLS] embeddings + logistic regression).

Run from anywhere:
    python scripts/training/train_query_classifier.py

Reads:  data/datasets/query_classifier_dataset.csv  (columns: query, category)
Writes: local/query_classifier_model.pkl

Note: the application does not load this model at runtime yet. See
smartscholar/query_processing/query_classifier.py.
"""

import os
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = PROJECT_ROOT / "local"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "datasets" / "query_classifier_dataset.csv"
CATEGORIES = ["factual", "explanatory", "technical", "conceptual", "general"]


class QueryClassifierTrainer:
    def __init__(self, model_path=None, bert_model="bert-base-uncased"):
        self.model_path = str(model_path or (LOCAL_DIR / "query_classifier_model.pkl"))
        self.bert_model = bert_model
        self.tokenizer, self.bert = self.load_bert()
        self.label_encoder = LabelEncoder()

    def load_bert(self):
        """Load BERT tokenizer and model, downloading them into ./local if needed."""
        os.makedirs(LOCAL_DIR, exist_ok=True)
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, cache_dir=str(LOCAL_DIR))
        bert = BertModel.from_pretrained(self.bert_model, cache_dir=str(LOCAL_DIR))
        return tokenizer, bert

    def encode_query(self, query):
        """Convert a query into its BERT [CLS] embedding."""
        tokens = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            output = self.bert(**tokens)
        return output.last_hidden_state[:, 0, :].squeeze().numpy()

    def train_model(self, dataset_path=DEFAULT_DATASET):
        """Train logistic regression on BERT embeddings and save it with joblib."""
        dataset = pd.read_csv(dataset_path)
        print(dataset.head())

        # Fixed category order so label indices are stable across runs.
        self.label_encoder.fit(CATEGORIES)

        X = [self.encode_query(q) for q in dataset["query"]]
        y = self.label_encoder.transform(dataset["category"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        classifier = LogisticRegression(max_iter=500)
        classifier.fit(X_train, y_train)

        joblib.dump(classifier, self.model_path)
        print(f"Model trained and saved to {self.model_path}")
        return classifier


if __name__ == "__main__":
    trainer = QueryClassifierTrainer()
    trainer.train_model()
