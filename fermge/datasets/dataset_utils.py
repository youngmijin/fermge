from itertools import product
from typing import Callable, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rich import print

__all__ = [
    "make_group_indices",
    "encode_onehot_columns",
    "one_way_normalizer",
]

# NOTE: A tuple of (
#           grouping criteria name,
#           (
#               group name,
#               values associated with group or function handling pd.Series to determine group
#           )
#       )
GroupCriteria = tuple[str, dict[str, list[int] | Callable[[pd.Series], pd.Series]]]


def make_group_indices(
    X_train: pd.DataFrame, X_valid: pd.DataFrame, *criterias: GroupCriteria
) -> dict[str, tuple[NDArray[np.intp], NDArray[np.intp]]]:
    group_names: list[list[str]] = []
    for _, group_mapping_dict in criterias:
        group_names.append(list(group_mapping_dict.keys()))
    group_name_combinations: list[list[str]] = [list(x) for x in product(*group_names)]

    group_indices: dict[str, tuple[NDArray[np.intp], NDArray[np.intp]]] = {}
    for gnc in group_name_combinations:
        group_divisors_train: list[pd.Series[bool]] = []
        group_divisors_valid: list[pd.Series[bool]] = []
        for group_name, (group_key, group_mapping_dict) in zip(gnc, criterias):
            group_name_criteria = group_mapping_dict[group_name]
            if isinstance(group_name_criteria, list):
                group_divisors_train.append(X_train[group_key].isin(group_name_criteria))
                group_divisors_valid.append(X_valid[group_key].isin(group_name_criteria))
            else:
                train_criteria = group_name_criteria(X_train[group_key])
                assert train_criteria.dtype == bool
                valid_criteria = group_name_criteria(X_valid[group_key])
                assert valid_criteria.dtype == bool
                group_divisors_train.append(train_criteria)
                group_divisors_valid.append(valid_criteria)
        group_divisor_train = np.all(group_divisors_train, axis=0)
        group_divisor_valid = np.all(group_divisors_valid, axis=0)

        if sum(group_divisor_train) == 0:
            print(f"skipping group index {gnc} because it has no train data")
            continue
        if sum(group_divisor_valid) == 0:
            print(f"skipping group index {gnc} because it has no valid data")
            continue

        group_indices["-".join(gnc)] = (
            X_train.index[group_divisor_train].to_numpy(),
            X_valid.index[group_divisor_valid].to_numpy(),
        )

    return group_indices


def encode_onehot_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        for new_colname in df[column].unique():
            df[new_colname] = 0
            df.loc[df[column] == new_colname, new_colname] = 1
        df = df.drop(columns=column)
    return df


T = TypeVar("T", bound=np.generic)


def one_way_normalizer(X_train: NDArray[T], X_valid: NDArray[T]) -> tuple[NDArray[T], NDArray[T]]:
    assert X_train.ndim == 2, "X_train must be 2D"
    assert X_valid.ndim == 2, "X_valid must be 2D"
    assert X_train.shape[1] == X_valid.shape[1], "X_train and X_valid must have same shape"
    X_train_float = X_train.astype(float)
    X_valid_float = X_valid.astype(float)
    X_train_mean = np.mean(X_train_float, axis=0).astype(float)
    X_train_std = np.std(X_train_float, axis=0).astype(float)
    X_train_std = np.where(X_train_std == 0, 1.0, X_train_std)
    X_train_norm = ((X_train_float - X_train_mean) / X_train_std).astype(X_train.dtype)
    X_valid_norm = ((X_valid_float - X_train_mean) / X_train_std).astype(X_valid.dtype)
    return X_train_norm, X_valid_norm
