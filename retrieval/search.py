
import os
import faiss
import numpy as np
import pandas as pd

from PIL import Image

from retrieval.feature_extractor import DenseNet121FeatureExtractor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

INDEX_PATH = os.path.join(ARTIFACT_DIR, "chexpert_visual.index")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "chexpert_metadata.csv")

CHECKPOINT_PATH = os.path.join(ARTIFACT_DIR, "densenet121_chexpert.pth")


class VisualRetriever:

    def __init__(self):

        self.index = faiss.read_index(INDEX_PATH)
        self.metadata = pd.read_csv(METADATA_PATH)

        self.extractor = DenseNet121FeatureExtractor(
            CHECKPOINT_PATH
        )

    def search(self, image_path, top_k=3):

        query_embedding = self.extractor.extract_image(image_path)
        query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            row = self.metadata.iloc[int(idx)]

            results.append({
                "image_path": str(row["image_path"]),
                "patient_id": str(row["patient_id"]),
                "similarity_score": float(score),
                "labels": {
                    col: (
                        None if pd.isna(row[col]) else float(row[col])
                    )
                    for col in [
                        "Cardiomegaly",
                        "Edema",
                        "Consolidation",
                        "Atelectasis",
                        "Pleural Effusion"
                    ]
                    if col in self.metadata.columns
                }
            })

        return results
