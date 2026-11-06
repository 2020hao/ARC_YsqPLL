"""rcr cnn snn peak dynamic bias calibrated; independent fixed model instance."""
from .network import Model

def create_model(**kwargs):
    return Model(**kwargs)
