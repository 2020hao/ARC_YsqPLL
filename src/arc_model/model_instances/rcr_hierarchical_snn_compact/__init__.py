"""rcr hierarchical snn compact; independent fixed model instance."""
from .network import Model

def create_model(**kwargs):
    return Model(**kwargs)
