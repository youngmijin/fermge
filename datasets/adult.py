import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from datasets.dataset import Dataset
from datasets.dataset_utils import (
    GroupCriteria,
    encode_onehot_columns,
    make_group_indices,
    one_way_normalizer,
)

__all__ = ["Adult"]


class Adult(Dataset):
    """
    Kohavi, R. (1996).
    Scaling up the accuracy of Naive-Bayes classifiers: a decision-tree hybrid.
    In Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data mining, Portland, 1996 (pp. 202-207).
    """

    def __init__(self):
        self.X_train: NDArray[np.float_] | None = None
        self.y_train: NDArray[np.float_] | None = None
        self.X_valid: NDArray[np.float_] | None = None
        self.y_valid: NDArray[np.float_] | None = None

        self.group_indices: dict[str, tuple[NDArray[np.intp], NDArray[np.intp]]] | None = None

    @property
    def name(self) -> str:
        return "adult"

    @property
    def file_local_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "adult.csv")

    @property
    def file_remote_url(self) -> str:
        return "https://drive.google.com/uc?id=1xpNXKrPR-VUQXNV9FrqIK5xe0yLkP46B"

    @property
    def file_md5_hash(self) -> str:
        return "46a9b0988c83b02d27640bf9ced3ab95"

    def load(self, *group_criterias: GroupCriteria):
        adult = pd.read_csv(self.file_local_path)
        adult = adult.drop(columns=["fnlwgt"])
        adult = adult.replace({"?": np.nan})
        adult = adult.dropna()
        adult = adult[adult["race"].isin(["Black", "White"])]
        adult = adult.replace({"<=50K": 0, ">50K": 1})
        adult = adult.replace({"Female": 0, "Male": 1})
        adult = adult.replace({"Black": 0, "White": 1})
        adult = adult.reset_index(drop=True)

        adult = encode_onehot_columns(
            adult,
            [
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "native-country",
            ],
        )

        X = adult.drop(columns=["income"])
        y = adult["income"]

        X_train: pd.DataFrame
        X_valid: pd.DataFrame
        y_train: pd.Series
        y_valid: pd.Series
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

        X_train = X_train.reset_index(drop=True)
        X_valid = X_valid.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_valid = y_valid.reset_index(drop=True)

        self.group_indices = make_group_indices(X_train, X_valid, *group_criterias)

        self.X_train, self.X_valid = one_way_normalizer(
            X_train.to_numpy().astype(np.float32),
            X_valid.to_numpy().astype(np.float32),
        )

        self.y_train = y_train.to_numpy().astype(np.float32)
        self.y_valid = y_valid.to_numpy().astype(np.float32)

    def get_group_criterias(self, n_groups: int) -> list[GroupCriteria]:
        if n_groups == 2:
            return [("gender", {"female": [0], "male": [1]})]
        else:
            raise NotImplementedError

    @property
    def train_data(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        assert self.X_train is not None, "X_train is not loaded"
        assert self.y_train is not None, "y_train is not loaded"
        return self.X_train, self.y_train

    @property
    def valid_data(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        assert self.X_valid is not None, "X_valid is not loaded"
        assert self.y_valid is not None, "y_valid is not loaded"
        return self.X_valid, self.y_valid

    @property
    def train_group_indices(self) -> dict[str, NDArray[np.intp]]:
        assert self.group_indices is not None, "group_indices is not loaded"
        return {k: v[0] for k, v in self.group_indices.items()}

    @property
    def valid_group_indices(self) -> dict[str, NDArray[np.intp]]:
        assert self.group_indices is not None, "group_indices is not loaded"
        return {k: v[1] for k, v in self.group_indices.items()}
