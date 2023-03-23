import warnings
import torch
from densepose.modeling.cse.utils import get_closest_vertices_mask_from_ES


def from_E_to_vertex(E, M, embed_map):
    """
        M is 1 for unkown regions
    """
    assert len(E.shape) == 4
    assert len(E.shape) == len(M.shape), (E.shape, M.shape)
    assert E.shape[0] == 1
    M = M.float()
    M = torch.cat([M, 1-M], dim=1)
    with warnings.catch_warnings():  # Ignore userError for pytorch interpolate from detectron2
        warnings.filterwarnings("ignore")
        vertices, _ = get_closest_vertices_mask_from_ES(
            E, M, E.shape[2], E.shape[3],
            embed_map, device=E.device)

    return vertices.long()
