# Colab artifact setup (run after Cell 13)

1. Download `fraud_streaming_deployment_bundle.zip` from the chat.
2. Run `files.upload()` in Colab and upload that ZIP.
3. Extract only the `deployment/` folder into the existing Drive project root.
4. Save a label-free training reference sample from `X_train` to `data/processed/baseline_reference.csv`.
5. Run `deployment/feast_repo/generate_feast_repo.py` to create `feast_features.parquet`, `feast_repo/feature_repo.py`, and `feast_repo/feature_store.yaml`.
6. The Docker Compose stack runs `feast_repo/bootstrap.py` to apply Feast definitions and materialize the offline rows into its SQLite online store.

The ground-truth label CSV stays outside the API/consumer containers.
