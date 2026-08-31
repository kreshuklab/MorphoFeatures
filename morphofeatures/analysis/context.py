"""Neighbor aggregation for MorphoContextFeatures."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.cluster import FeatureAgglomeration

from morphofeatures.data.contracts import EmbeddingTable


def aggregate_neighbors(
    embeddings: EmbeddingTable,
    neighbors: Mapping[int, Sequence[int]],
    include_self: bool = True,
    reducer: str = "mean",
) -> EmbeddingTable:
    if reducer not in {"mean", "max"}:
        raise ValueError("reducer must be 'mean' or 'max'")
    positions = {int(label_id): index for index, label_id in enumerate(embeddings.label_ids)}
    output_ids, output_features = [], []
    for label_id in embeddings.label_ids:
        member_ids = [int(value) for value in neighbors.get(int(label_id), ()) if int(value) in positions]
        if include_self:
            member_ids.insert(0, int(label_id))
        if not member_ids:
            continue
        rows = embeddings.features[[positions[value] for value in member_ids]]
        aggregated = rows.mean(axis=0) if reducer == "mean" else rows.max(axis=0)
        output_ids.append(int(label_id))
        output_features.append(aggregated)
    return EmbeddingTable(np.asarray(output_ids, dtype=np.int64), np.asarray(output_features))


def agglomerate_features(
    embeddings: EmbeddingTable, n_features: int = 200
) -> EmbeddingTable:
    if n_features <= 0 or n_features > embeddings.features.shape[1]:
        raise ValueError("n_features must be between 1 and the input feature count")
    transformed = FeatureAgglomeration(n_clusters=int(n_features)).fit_transform(
        embeddings.features
    )
    return EmbeddingTable(embeddings.label_ids, transformed)
