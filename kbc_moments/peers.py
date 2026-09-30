"""Inferred intent from peers: what did customers with the same profile do next?"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .synth import make_population

FEATURES = ["age", "income", "housing", "savings"]
EVENTS = {
    "bought_home_3y": "bought a home",
    "bought_car_3y": "bought a car",
    "study_or_move_abroad_3y": "studied or moved abroad",
    "started_pension_saving_3y": "started pension saving",
}


@lru_cache(maxsize=1)
def population():
    return make_population()


def peer_insights(profile: dict, k: int = 300) -> dict:
    pop = population()
    x = pop[FEATURES].to_numpy(float)
    mu, sd = x.mean(0), x.std(0)
    q = (np.array([profile[f] for f in FEATURES], float) - mu) / sd
    z = (x - mu) / sd
    z[:, 3] = np.log1p(pop["savings"]) / np.log1p(pop["savings"]).std()  # savings on log scale
    q[3] = np.log1p(profile["savings"]) / np.log1p(pop["savings"]).std()
    idx = np.argsort(((z - q) ** 2).sum(1))[:k]
    peers = pop.iloc[idx]
    return {label: float(peers[col].mean()) for col, label in EVENTS.items()}
