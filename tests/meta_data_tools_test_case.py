# meta_data_tools_test_case.py
from pathlib import Path
from src.file_tools import FileTools
import nltk
import pandas as pd
import os
import unittest
from src.meta_data_tools import MetaDataTools as MDT
from src.custom_exceptions import DataFrameException


class MetaDataToolsTestCase(unittest.TestCase):
    def setUp(self):
        """Fixtures used by tests."""
        self.Root = Path(__file__).parent
        self.Temp = os.path.join(self.Root, 'temp_meta_data_tools')
        self.TestDataDir = os.path.join(self.Root, 'test_data')
        self.Test2ColFile = os.path.join(self.TestDataDir, 'test_tsv_2_cols.txt')
        self.Test5ColIncLabelDataFrame = MDT.read_raw_data(os.path.join(self.TestDataDir, 'test_tsv_5_cols_inc_labels.txt'))
        self.TestNoDescDataFrame = MDT.read_raw_data(os.path.join(self.TestDataDir, 'test_no_desc_tsv.txt'))
        self.Test1ColDataFrame = MDT.read_raw_data(os.path.join(self.TestDataDir, 'test_tsv_1_col.txt'))
        self.Test2ColDataFrame = MDT.read_raw_data(self.Test2ColFile)
        self.ExcelXlsxFilePath = os.path.join(self.TestDataDir, 'test_xls.xlsx')
        self.ExcelXlsmFilePath = os.path.join(self.TestDataDir, 'test_xlsm.xlsm')

        FileTools.ensure_empty_directory(self.Temp)

    def tearDown(self) -> None:
        FileTools.ensure_empty_directory(self.Temp)

    def test_read_raw_data__raw_data_has_expected_shape(self):
        df = self.Test5ColIncLabelDataFrame

        pd.set_option('display.max_colwidth', 100)
        print(df.head())
        self.assertEqual((2, 5), df.shape)

    def test_cleanse_text_in_dataframe__returns_dataframe_without_punctuation(self):
        test_df = self.Test5ColIncLabelDataFrame
        new_df = MDT.cleanse_text_in_dataframe(test_df)

        expected = ['institution']
        actual = new_df['Some Description'][0]
        self.assertEqual(expected, actual)

    def test_cleanse_text__remove_specified_punctuation(self):
        expected = 'dogs kennels paint'
        with self.subTest(self):
            print(f'Testing for: Apostrophe')
            test_text = "Dog's kennel's paint"
            actual = ' '.join(MDT.cleanse_text(test_text))
            self.assertEqual(expected, actual)

        with self.subTest(self):
            print(f'Testing for: Double quote')
            test_text = 'Dogs "kennels" paint'
            actual = ' '.join(MDT.cleanse_text(test_text))
            self.assertEqual(expected, actual)

    def test_cleanse_text__replace_other_punctuation(self):
        expected = 'dogs kennel dog house'.split(' ')
        sub_tests = [['Hyphen', "Dog's kennel dog-house"],
                     ['Equals sign', "dog's kennel=dog house"],
                     ['Adjacent punctuation', "Dog's kennel=dog****house&&&"]]

        for sub_test in sub_tests:
            with self.subTest(self):
                print(f'Testing for: {sub_test[0]}')
                test_text = sub_test[1]
                actual = MDT.cleanse_text(test_text)
                self.assertEqual(expected, actual)

    def test_stemmer__by_stemmer(self):
        test_words = ['describes', 'describe', 'descriptor', 'description']
        sub_tests = [['Porter', nltk.PorterStemmer(), ['describ', 'describ', 'descriptor', 'descript']],
                     ['Lancaster', nltk.LancasterStemmer(), ['describ', 'describ', 'describ', 'describ']]]
        for sub_test in sub_tests:
            with self.subTest(self):
                print(f'Testing for: {sub_test[0]}')
                stemmer = sub_test[1]
                expected = sub_test[2]
                actual = MDT.stemming(test_words, stemmer)
                self.assertEqual(expected, actual)

    def test_identify_descriptor_column__when_valid_column_name_exists__returns_index_and_name(self):
        df = self.Test5ColIncLabelDataFrame
        expected = 1
        # Column index is first part of pairing returned
        actual = MDT.identify_descriptor_column(self.Test5ColIncLabelDataFrame)[0]

        self.assertEqual(expected, actual)

    def test_identify_descriptor_column__when_valid_column_name_not_exists__returns_dummy_index_and_name(self):
        expected = -1
        # Column index is first part of pairing returned
        actual = MDT.identify_descriptor_column(self.TestNoDescDataFrame)[0]

        self.assertEqual(expected, actual)

    def test_field_tokenized_descriptor_list_from_df__when_valid__returns_paired_series_list(self):
        # Column index is first part of pairing returned
        sub_tests = [['2 columns', self.Test2ColDataFrame], ['5 columns', self.Test5ColIncLabelDataFrame]]
        for sub_test in sub_tests:
            with self.subTest(self):
                print(f'Testing for: {sub_test[0]}')
                fdl = MDT.field_tokenized_descriptor_list_from_df(sub_test[1])

                self.assertTrue(len(fdl) == 2)
                self.assertTrue(fdl[0].__class__.__name__ == 'Series')
                self.assertTrue(fdl[1].__class__.__name__ == 'list')

    def test_field_tokenized_descriptor_list_from_df__when_invalid__raises_exception(self):
        # Column index is first part of pairing returned
        sub_tests = [['1 column', self.Test1ColDataFrame], ['No descriptor', self.TestNoDescDataFrame]]
        for sub_test in sub_tests:
            with self.subTest(self):
                print(f'Testing for: {sub_test[0]}')
                self.assertRaises(DataFrameException, MDT.field_tokenized_descriptor_list_from_df, sub_test[1])

    def test_field_tokenized_descriptor_list_from_df__last_element_is_expected_type(self):
        sub_tests = [['treat as not labelled', False, 'list'], ['treat as labelled', True, 'Series']]
        df = self.Test5ColIncLabelDataFrame
        for sub_test in sub_tests:
            testing_for = sub_test[0]
            is_labelled = sub_test[1]
            expected = sub_test[2]

            print(f'Testing for: {testing_for}')
            result = MDT.field_tokenized_descriptor_list_from_df(df, is_labelled)

            self.assertEqual(expected, type(result[-1]).__name__)

    def test_field_tokenized_descriptor_list_from_df__when_valid__tokenizes_data(self):
        fdl = MDT.field_tokenized_descriptor_list_from_df(self.Test2ColDataFrame)
        expected = ['institution']
        actual = fdl[1][0]
        self.assertEqual(expected, actual)

    def test_field_tokenized_descriptor_df_from_df__when_valid__tokenizes_data(self):
        fdl = MDT.field_tokenized_descriptor_df_from_df(self.Test2ColDataFrame, 'test_name')

        expected = ['institution']
        actual = fdl['TokenizedDescriptors'][0]
        self.assertEqual(expected, actual)

    def test_field_tokenized_descriptor_df_from_df__when_labelled__includes_label_column(self):
        fdl = MDT.field_tokenized_descriptor_df_from_df(self.Test5ColIncLabelDataFrame, 'test_name', is_labelled=True)

        self.assertTrue(fdl.columns[len(fdl.columns) - 1] == 'Labels')

    def test_field_descriptors_df_from_file__when_valid_file(self):
        src_path = self.Test2ColFile
        target_dir = self.Temp
        prefix = 'dummy'

        with self.subTest(self):
            print('Testing for: empty directory pre-test')
            expected = 0
            actual = len([name for name in os.listdir(self.Temp) if os.path.isfile(os.path.join(self.Temp, name))])
            self.assertEqual(expected, actual)

        result = MDT.field_descriptors_df_from_file(src_path, target_dir, prefix, to_save=True)

        with self.subTest(self):
            print('Testing for: return DataFrame')
            expected = 'DataFrame'
            actual = result.__class__.__name__
            self.assertEqual(expected, actual)

        with self.subTest(self):
            print('Testing for: save processed file')
            expected = 1
            actual = len([name for name in os.listdir(self.Temp) if os.path.isfile(os.path.join(self.Temp, name))])
            self.assertTrue(expected, actual)

    def test_dict_of_field_descriptors_dfs_from_files(self):
        src_path = self.TestDataDir
        target_dir = self.Temp
        prefix = 'dummy'

        dataframes, errors = MDT.dict_of_field_descriptors_dfs_from_files(src_path, target_dir, prefix, to_save=True)

        with self.subTest():
            testing_for = 'Expected number of returned DataFrames'
            print(f'Testing for: {testing_for}')
            expected = 2
            actual = len(dataframes)

            self.assertEqual(expected, actual)

        with self.subTest():
            testing_for = 'Expected number of errors'
            print(f'Testing for: {testing_for}')
            # expected = len([name for name in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, name))])
            expected = 2
            actual = len(errors)

            self.assertEqual(expected, actual)

    def test_list_of_field_descriptors_dfs_from_files(self):
        src_path = self.TestDataDir
        target_dir = self.Temp
        prefix = 'dummy'

        dataframes, errors = MDT.list_of_field_descriptors_dfs_from_files(src_path, target_dir, prefix, to_save=True)

        with self.subTest():
            testing_for = 'Expected number of returned DataFrames'
            print(f'Testing for: {testing_for}')
            expected = 2
            actual = len(dataframes)

            self.assertEqual(expected, actual)

        with self.subTest():
            testing_for = 'Expected number of errors'
            print(f'Testing for: {testing_for}')
            # expected = len([name for name in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, name))])
            expected = 2
            actual = len(errors)

            self.assertEqual(expected, actual)

    def test_collate_dfs_from_list(self):
        dataframes, errors = MDT.list_of_field_descriptors_dfs_from_files(
            src_path=self.TestDataDir, target_dir=self.Temp, prefix='from_list', to_save=False)

        MDT.collate_dfs_from_list(df_list=dataframes, save_dir=self.Temp, save_name='all_data.txt', prefix='from_list_')
        self.assertTrue(Path(os.path.join(self.Temp, 'from_list_all_data.txt')).is_file())


if __name__ == '__main__':
    unittest.main()
