"""Which published FlowHMR prediction a result may read.

A saved control is only a draft choice; the committed generation decides what
placement, preview and export actually use. The two optional finger passes are
mutually exclusive, and a selected pass without its own published prediction is
an error instead of a silent substitution of another prediction file.
"""
from .flow_hamer import prediction_name as hamer_prediction_name
from . import flow_fah


def prediction_name(project, generated=False):
    selected = flow_fah.prediction_name(project, generated=generated)
    if selected:
        return selected
    return hamer_prediction_name(project, generated=generated)


def publish_hand_boxes(project, state=None):
    """SAM must keep the per-frame left/right boxes for either optional finger pass."""
    return flow_fah.hand_boxes_required(project, state)
