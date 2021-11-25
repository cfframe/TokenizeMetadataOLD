# meta_data_tools_test_case.py
from pathlib import Path
import pandas as pd
import os
import unittest
from src.meta_data_tools import MetaDataTools as MDT


class MetaDataToolsTestCase(unittest.TestCase):
    def setUp(self):
        """Fixtures used by tests."""
        self.Root = Path(__file__).parent
        self.TestDataDir = os.path.join(self.Root, 'test_data')
        self.TsvFilePath = os.path.join(self.TestDataDir, 'test_tsv.txt')
        self.TestDataFrame = MDT.read_raw_data(self.TsvFilePath)

    def test_read_raw_data__raw_data_has_expected_shape(self):
        df = MDT.read_raw_data(self.TsvFilePath)

        pd.set_option('display.max_colwidth', 100)
        print(df.head())
        self.assertEqual(df.shape, (2, 4))

    def test_remove_punctuation_from_text__returns_text_without_punctuation(self):
        test_text = 'Fred. was* here.'
        expected = 'Fred was here'
        actual = MDT.remove_punctuation_from_text(test_text)
        self.assertEqual(expected, actual)

    def test_remove_punctuation_from_dataframe__returns_dataframe_without_punctuation(self):
        test_df = self.TestDataFrame
        new_df = MDT.remove_punctuation_from_dataframe(test_df)

        expected = 'institution'
        actual = new_df['Description'][0]
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
