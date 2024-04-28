import os

import numpy as np
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, text_file_path: str, shuffle: bool = False, transform=None, target_transform=None):
        self._test_data = None
        self._train_data = None
        self.transform = transform
        self.target_transform = target_transform
        self.text_file_path = text_file_path
        # Abs file path
        if not os.path.isabs(self.text_file_path):
            self.text_file_path = os.path.abspath(self.text_file_path)

        self.text = self.load_text_file()

        if shuffle:
            np.random.shuffle(self.text)

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


class LLMDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        super(LLMDataLoader, self).__init__(dataset, batch_size, shuffle)

    @staticmethod
    def from_huggingface_dataset(dataset, batch_size, shuffle=True):
        return LLMDataLoader(dataset, batch_size, shuffle)
