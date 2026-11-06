"""Local model construction and parent-preserving inference."""
from .network import Model

def create_predictor(**kwargs):
    return Model(**kwargs)
