"""rcr cnn snn peak dynamic affine calibrated; independent fixed model instance."""
from .network import Model

def create_model(**kwargs):
    return Model(**kwargs)
