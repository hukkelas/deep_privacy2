from dp2.metrics.ppl import calculate_ppl
from dp2.metrics.torch_metrics import compute_metrics_iteratively
from dp2.metrics.fid_clip import compute_fid_clip


def final_eval_fn(*args, **kwargs):
    result = compute_metrics_iteratively(*args, **kwargs)
    result2 = calculate_ppl(*args, **kwargs,)
    result3 = compute_fid_clip(*args, **kwargs)
    assert all(key not in result for key in result2)
    result.update(result2)
    result.update(result3)
    return result


def train_eval_fn(*args, **kwargs):
    result = compute_metrics_iteratively(*args, **kwargs)
    result2 = compute_fid_clip(*args, **kwargs)
    assert all(key not in result for key in result2)
    result.update(result2)
    return result