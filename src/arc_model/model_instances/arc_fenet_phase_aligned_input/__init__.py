"""ArcFENet 相位对齐输入; independent fixed model instance."""
from .network import Model

def create_model(**kwargs):
    return Model(**kwargs)
