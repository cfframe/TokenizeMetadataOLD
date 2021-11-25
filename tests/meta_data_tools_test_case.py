# meta_data_tools_test_case.py
from pathlib import Path

import nltk
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
        self.NoDescTsvFilePath = os.path.join(self.TestDataDir, 'test_no_desc_tsv.txt')
        self.TestNoDescDataFrame = MDT.read_raw_data(self.NoDescTsvFilePath)

    def test_read_raw_data__raw_data_has_expected_shape(self):
        df = MDT.read_raw_data(self.TsvFilePath)

        pd.set_option('display.max_colwidth', 100)
        print(df.head())
        self.assertEqual(df.shape, (2, 4))

    def test_cleanse_text_in_dataframe__returns_dataframe_without_punctuation(self):
        test_df = self.TestDataFrame
        new_df = MDT.cleanse_text_in_dataframe(test_df)

        expected = ['institution']
        actual = new_df['Some Description'][0]
        self.assertEqual(expected, actual)

    def test_cleanse_text__remove_specified_punctuation(self):
        expected = 'dogs kennels paint'
        with self.subTest(self, testing_for='Apostrophe'):
            test_text = "Dog's kennel's paint"
            actual = ' '.join(MDT.cleanse_text(test_text))
            self.assertEqual(expected, actual)

        with self.subTest(self, testing_for='Double quote'):
            test_text = 'Dogs "kennels" paint'
            actual = ' '.join(MDT.cleanse_text(test_text))
            self.assertEqual(expected, actual)

    def test_cleanse_text__replace_other_punctuation(self):
        expected = 'dogs kennel dog house'.split(' ')
        sub_tests = [['Hyphen', "Dog's kennel dog-house"],
                     ['Equals sign', "dog's kennel=dog house"],
                     ['Adjacent punctuation', "Dog's kennel=dog****house&&&"]]

        for sub_test in sub_tests:
            with self.subTest(self, testing_for=sub_test[0]):
                test_text = sub_test[1]
                actual = MDT.cleanse_text(test_text)
                self.assertEqual(expected, actual)

    def test_stemmer__by_stemmer(self):
        test_words = ['describes', 'describe', 'descriptor', 'description']
        sub_tests = [['Porter', nltk.PorterStemmer(), ['describ', 'describ', 'descriptor', 'descript']],
                     ['Lancaster', nltk.LancasterStemmer(), ['describ', 'describ', 'describ', 'describ']]]
        for sub_test in sub_tests:
            with self.subTest(self, testing_for=sub_test[0]):
                stemmer = sub_test[1]
                expected = sub_test[2]
                actual = MDT.stemming(test_words, stemmer)
                self.assertEqual(expected, actual)

    def test_identify_descriptor_column__when_valid_column_name_exists__returns_index_and_name(self):
        df = self.TestDataFrame
        expected = 1
        # Column index is first part of pairing returned
        actual = MDT.identify_descriptor_column(self.TestDataFrame)[0]

        self.assertEqual(expected, actual)

    def test_identify_descriptor_column__when_valid_column_name_not_exists__returns_dummy_index_and_name(self):
        df = self.TestNoDescDataFrame

        expected = -1
        # Column index is first part of pairing returned
        actual = MDT.identify_descriptor_column(self.TestNoDescDataFrame)[0]

        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
