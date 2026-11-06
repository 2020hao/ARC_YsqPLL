"""rcr hierarchical snn wide; independent fixed model instance."""
from .network import Model

def create_model(**kwargs):
    return Model(**kwargs)
