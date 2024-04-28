import unittest

from src.data_load import TextDataset


class TestDataLoader(unittest.TestCase):
    def test_load_file(self):
        file_path = '../data/maliciousInstruct.txt'
        dataset = TextDataset(file_path)
        self.assertNotEqual(len(dataset), 0)

    def test_first_line_no_shuffle(self):
        file_path = '../data/maliciousInstruct.txt'
        dataset = TextDataset(file_path)
        # Read the first line  from the file
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        self.assertEqual(dataset[0], first_line)

    def test_first_line_shuffle(self):
        file_path = '../data/maliciousInstruct.txt'
        dataset = TextDataset(file_path, shuffle=True)
        # Read the first line  from the file
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        self.assertNotEqual(dataset[0], first_line)

    def test_split_dataset(self):
        file_path = '../data/maliciousInstruct.txt'
        dataset = TextDataset(file_path)
        train_data, test_data = dataset.train_test_split()
        self.assertEqual(len(train_data) + len(test_data), len(dataset))

    def test_file_not_found(self):
        file_path = 'non_existent_file.txt'
        with self.assertRaises(FileNotFoundError):
            dataset = TextDataset(file_path)
