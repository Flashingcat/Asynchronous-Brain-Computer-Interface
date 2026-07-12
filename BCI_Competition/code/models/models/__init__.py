from .conformer import ConformerClassifier
from .deepcnn import DeepCNNClassifier
from .deformer import DeformerClassifier
from .eegnet import EEGNetClassifier
from .shallowconvnet import ShallowConvNetClassifier

__all__ = [
    "ConformerClassifier",
    "DeepCNNClassifier",
    "DeformerClassifier",
    "EEGNetClassifier",
    "ShallowConvNetClassifier",
]
