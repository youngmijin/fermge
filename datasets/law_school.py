import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from datasets.dataset import Dataset
from datasets.dataset_utils import GroupCriteria, make_group_indices, one_way_normalizer

__all__ = ["LawSchool"]


class LawSchool(Dataset):
    """
    Wightman, L. F. (1998).
    LSAC national longitudinal bar passage study.
    LSAC research report series.
    """

    def __init__(self):
        self.X_train: NDArray[np.float_] | None = None
        self.y_train: NDArray[np.float_] | None = None
        self.X_valid: NDArray[np.float_] | None = None
        self.y_valid: NDArray[np.float_] | None = None

        self.group_indices: dict[str, tuple[NDArray[np.intp], NDArray[np.intp]]] | None = None

    @property
    def name(self) -> str:
        return "law_school"

    @property
    def file_local_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "law_dataset.csv")

    @property
    def file_remote_url(self) -> str:
        return "https://drive.google.com/file/d/1xvt9Pzykyp_mWLzGkWJ7qpb5ZawbeSz1/view"

    @property
    def file_md5_hash(self) -> str:
        return "3296294f79ddd38d8f5fe31499f6ee12"

    def load(self, group_size: int = 2):
        law = pd.read_csv(self.file_local_path)
        law = law.dropna()
        law = law.reset_index(drop=True)

        X = law.drop(columns=["pass_bar"])
        y = law["pass_bar"]

        X_train: pd.DataFrame
        X_valid: pd.DataFrame
        y_train: pd.Series
        y_valid: pd.Series
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

        X_train = X_train.reset_index(drop=True)
        X_valid = X_valid.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_valid = y_valid.reset_index(drop=True)

        male_gc: GroupCriteria = (
            "male",
            {
                "X": lambda x: x == 0.0,
                "O": lambda x: x != 0.0,
            },
        )

        if group_size == 2:
            self.group_indices = make_group_indices(X_train, X_valid, male_gc)
        else:
            raise ValueError("Invalid group size")

        self.X_train, self.X_valid = one_way_normalizer(
            X_train.to_numpy().astype(np.float32),
            X_valid.to_numpy().astype(np.float32),
        )

        self.y_train = y_train.to_numpy().astype(np.float32)
        self.y_valid = y_valid.to_numpy().astype(np.float32)

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
