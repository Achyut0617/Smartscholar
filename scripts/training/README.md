# Scripts

Operational scripts that run outside the application. Reusable application logic belongs in `src/smartscholar/`.

| Script | Purpose |
|---|---|
| `training/train_query_classifier.py` | Trains the query-intent classifier from `data/datasets/query_classifier_dataset.csv`. Requires pandas, scikit-learn, torch and transformers. |

Other categories (evaluation, maintenance, development) will be created when a real script needs them.
