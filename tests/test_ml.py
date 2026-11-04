"""Fast unit tests for the ML helpers."""
import numpy as np
import pandas as pd

from config import ARCHETYPE_TEMPLATES, RADAR_CATEGORIES
from ml.archetypes import _name_clusters, _neighbor_preservation


def test_neighbor_preservation_is_perfect_for_identity():
    x = np.random.default_rng(0).normal(size=(60, 5))
    assert _neighbor_preservation(x, x, k=5) == 1.0


def test_cluster_names_are_unique_and_sensible():
    # cluster 0 looks like a rim protector, cluster 1 like a shooter
    prof = pd.DataFrame(0.0, index=[0, 1], columns=RADAR_CATEGORIES)
    prof.loc[0, ["Paint", "Defense"]] = 1.5
    prof.loc[0, "Shooting"] = -1.5
    prof.loc[1, "Shooting"] = 2.0
    names = _name_clusters(prof)
    assert names[0] == "Rim Protector"
    assert names[1] == "Movement Shooter"
    assert len(set(names.values())) == 2
    assert set(names.values()) <= set(ARCHETYPE_TEMPLATES)
