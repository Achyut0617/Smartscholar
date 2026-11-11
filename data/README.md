# Data

`datasets/query_classifier_dataset.csv` contains 999 labeled queries with the columns `query` and `category`. Categories: `factual`, `explanatory`, `technical`, `conceptual`, `general`. It is used by `scripts/training/train_query_classifier.py`.

Model weights are not stored here. They go in `local/`, which is git-ignored.
