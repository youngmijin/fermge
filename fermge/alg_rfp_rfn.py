import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = ["calc_rfp_rfn"]


@njit
def calc_rfp_rfn(
    total_tn: float,
    total_fp: float,
    total_fn: float,
    total_tp: float,
    group_tn: NDArray[np.float_],
    group_fp: NDArray[np.float_],
    group_fn: NDArray[np.float_],
    group_tp: NDArray[np.float_],
) -> tuple[NDArray[np.float_], NDArray[np.float_], float, float]:
    """Calculates rFP, rFN values for total and each groups."""

    assert group_tn.shape == group_fp.shape == group_fn.shape == group_tp.shape

    total_rfp: float = total_fp / (total_fp + total_tn)
    total_rfn: float = total_fn / (total_fn + total_tp)
    group_rfp: NDArray[np.float_] = group_fp / (group_fp + group_tn)
    group_rfn: NDArray[np.float_] = group_fn / (group_fn + group_tp)

    return (group_rfp, group_rfn, total_rfp, total_rfn)
