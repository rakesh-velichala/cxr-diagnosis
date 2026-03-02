"""Similarity-based retrieval of chest X-ray cases from the dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from app.config import settings
from data.loader import get_disease_columns
from utils.logging_config import logger


@dataclass
class RetrievedCase:
    """A single retrieved similar case."""

    image_id: str
    similarity: float
    labels: dict[str, int]

    @property
    def positive_findings(self) -> list[str]:
        """Return list of disease names with positive (1) labels."""
        return [k for k, v in self.labels.items() if v == 1]


class DatasetRetriever:
    """Retrieve the most similar X-ray cases using cosine similarity.

    Parameters
    ----------
    embeddings : np.ndarray
        Pre-computed CLIP embeddings of shape ``(N, D)``.
    df : pd.DataFrame
        Dataset CSV loaded as a DataFrame.
    top_k : int
        Number of similar cases to retrieve.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        df: pd.DataFrame,
        top_k: Optional[int] = None,
    ) -> None:
        self.embeddings = embeddings.astype(np.float32)
        self.df = df
        self.top_k = top_k or settings.top_k
        self.disease_cols = get_disease_columns(df)
        self.image_ids: list[str] = df[settings.image_col].tolist()

        # Pre-normalize for fast cosine similarity.
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        self._normed = self.embeddings / norms

        logger.info(
            "DatasetRetriever ready: %d samples, top_k=%d",
            len(self.image_ids),
            self.top_k,
        )

    def retrieve(self, query_embedding: np.ndarray) -> list[RetrievedCase]:
        """Find the top-k most similar cases to a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            CLIP embedding of the uploaded image, shape ``(D,)`` or ``(1, D)``.

        Returns
        -------
        list[RetrievedCase]
            Top-k retrieved cases ordered by descending similarity.
        """
        query = query_embedding.astype(np.float32).reshape(1, -1)
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        query_norm = np.clip(query_norm, 1e-8, None)
        query_normed = query / query_norm

        similarities = (self._normed @ query_normed.T).squeeze()  # (N,)

        top_indices = np.argsort(similarities)[::-1][: self.top_k]

        results: list[RetrievedCase] = []
        for idx in top_indices:
            image_id = self.image_ids[idx]
            sim_score = float(similarities[idx])
            row = self.df.iloc[idx]
            labels = {col: int(row[col]) for col in self.disease_cols}
            results.append(
                RetrievedCase(
                    image_id=image_id,
                    similarity=sim_score,
                    labels=labels,
                )
            )

        logger.info(
            "Retrieved %d cases, top similarity=%.4f",
            len(results),
            results[0].similarity if results else 0.0,
        )
        return results
