import os

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset as HuggingFaceDataset
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    TextDataset class to load text data from a file and return it as a dataset
    """

    def __init__(self, text_file_path: str, huggingface_ds: bool = False, shuffle: bool = False,
                 format_type: str = None):
        self._test_data = None
        self._train_data = None
        self.text_file_path = text_file_path
        # Abs file path
        if not os.path.isabs(self.text_file_path):
            self.text_file_path = os.path.abspath(self.text_file_path)

        self.text = self.load_text_file()

        if shuffle:
            np.random.shuffle(self.text)

        if huggingface_ds:
            self.text = self.to_huggingface_dataset()

        if huggingface_ds is True and format_type is not None:
            self.text = self.text.with_format(format_type)

    def load_text_file(self):
        _text_array = np.array([])
        with open(self.text_file_path, 'r') as f:
            text = [line.strip() for line in f.readlines()]
            _text_array = np.array(list(map(lambda x: x, text)))
        return _text_array

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]

    def __repr__(self):
        return f"TextDataLoader(text_file_path={self.text_file_path})"

    def __str__(self):
        return f"TextDataLoader(text_file_path={self.text_file_path})"

    def train_test_split(self, train_size=0.8):
        self._train_data = self.text[:int(len(self.text) * train_size)]
        self._test_data = self.text[int(len(self.text) * train_size):]
        return self._train_data, self._test_data

    def to_huggingface_dataset(self):
        # Abs file path
        if not os.path.isabs(self.text_file_path):
            self.text_file_path = os.path.abspath(self.text_file_path)
        return HuggingFaceDataset.from_text(self.text_file_path)


class ParquetDataset(Dataset):
    """
    ParquetDataset class to load data from a parquet file and return it as a dataset
    """

    def __init__(self, file_path: str, huggingface_ds: bool = False, format_type: str = None):
        self.parquet_file_path = file_path
        if not os.path.isabs(self.parquet_file_path):
            self.parquet_file_path = os.path.abspath(self.parquet_file_path)

        self.parquet_data = pd.read_parquet(self.parquet_file_path)

        if huggingface_ds:
            self.parquet_data = self.df_to_huggingface_dataset()

        if huggingface_ds is True and format_type is not None:
            self.parquet_data = self.parquet_data.with_format(format_type)

    def __len__(self):
        return len(self.parquet_data)

    def __getitem__(self, idx):
        return self.parquet_data.iloc[idx]

    def df_to_huggingface_dataset(self):
        return HuggingFaceDataset.from_pandas(self.parquet_data)


class VisualLoader(Dataset):
    def __init__(self, download: bool = False, percent: int = 30):
        self.download = download
        self.data = None

        if self.download:
            self.data = self.download_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def download_data(path: str = "../data/imagenet_data") -> HuggingFaceDataset:
        path = os.path.abspath(path)
        os.environ["HF_DATASETS_CACHE"] = f"{path}"
        dataset = load_dataset("phiyodr/inpaintCOCO")
        return dataset
