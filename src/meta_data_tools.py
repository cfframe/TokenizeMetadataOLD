# meta_data_tools.py
import datetime
import nltk
import os
from pathlib import Path
import pandas as pd
import re
import string
from src.custom_exceptions import DataFrameException


class MetaDataTools:
    """Static methods to work with Meta Data"""

    @staticmethod
    def cleanse_text_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleanse textual data held in a DataFrame. Excludes column names.

        :param df: DataFrame to process.
        :returns: DataFrame with text cleansed.
        """        
        new_df = pd.DataFrame()
        new_df[df.columns[0]] = df[df.columns[0]].apply(lambda x: str.lower(x))
        for column in df.columns[1:]:
            new_df[column] = df[column].apply(lambda x: MetaDataTools.cleanse_text(x))

        return new_df

    @staticmethod
    def read_raw_data(source_path: str) -> pd.DataFrame:
        raw_data = pd.read_csv(source_path, sep='\t', header='infer')
        return raw_data

    @staticmethod
    def cleanse_text(text: str) -> list:
        """Pre process text prior to tokenizing.

        Make lower case.
        Remove apostrophes and quotes, replace other punctuation with a space.

        :param text: Original text.
        :return: Processed text as list.
        """
        # Make all lower case first
        text = str.lower(text)
        # Punctuation removal then swap out other marks for spaces
        to_remove = ["'", '"']

        for item in to_remove:
            text = text.replace(item, '')

        for item in string.punctuation:
            text = text.replace(item, ' ')

        # Tokenize, without stop words
        stopwords = nltk.corpus.stopwords.words('english')
        tokens = re.split('\W+', text)
        text = [word for word in tokens if word not in stopwords and len(word) > 0]

        return text

    @staticmethod
    def stemming(tokenized_text: list, stemmer=nltk.LancasterStemmer()):
        """Stem stemmed_text by optionally chosen stemmer.

        :param tokenized_text: List of words to be stemmed; assumes already pre-processed.
        :param stemmer: NLTK stemmer (default LancasterStemmer)
        :return: List of stemmed words.
        """
        stemmed_text = [stemmer.stem(word) for word in tokenized_text]
        return stemmed_text

    @staticmethod
    def identify_descriptor_column(df: pd.DataFrame) -> list:
        """Attempt to identify the column of a DataFrame holding descriptions of field names.

        Returns the first column found that may hold a description, going to column name only.

        :param df: DataFrame to process.
        :return: List [descriptor column index, original descriptor column name]. If none found, returns [-1, ''].
        """
        lancaster = nltk.LancasterStemmer()
        descriptor_stem = lancaster.stem('description')

        clean_columns = []
        for column in df.columns:
            clean_columns.append(MetaDataTools.cleanse_text(column))

        stemmed_columns = [MetaDataTools.stemming(column, lancaster) for column in clean_columns]

        descriptor_column = ''
        descriptor_column_index = -1

        for column in stemmed_columns:
            if descriptor_stem in column:
                descriptor_column_index = stemmed_columns.index(column)
                descriptor_column = df.columns[descriptor_column_index]

                break

        return [descriptor_column_index, descriptor_column]

    @staticmethod
    def field_tokenized_descriptor_list_from_df(df: pd.DataFrame) -> list:
        """Derive a list of field names against descriptions from a DataFrame.

        Assume that Field names are the first column, and that if only two columns then
        the second column is the descriptors.

        :param df: Source DataFrame.
        :returns: See description.
        """

        if len(df.columns) < 2:
            raise DataFrameException('Data set has too few columns, should have at least two.')
        elif len(df.columns) == 2:
            descriptor_column_index = 1
        else:
            descriptor_column_index = MetaDataTools.identify_descriptor_column(df)[0]
            if descriptor_column_index < 0:
                raise DataFrameException('No descriptor column identified for DataFrame.')

        field_names = df[df.columns[0]]
        descriptions = [MetaDataTools.cleanse_text(text) for text in df[df.columns[descriptor_column_index]]]

        return [field_names, descriptions]

    @staticmethod
    def field_tokenized_descriptor_df_from_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Derive a reduced DataFrame of field names against tokenized descriptions from a source DataFrame.

        Assume that Field names are the first column, and that if only two columns then
        the second column is the descriptors.

        :param df: Source DataFrame.
        :param source: Source of data.
        :returns: DataFrame. See description.
        """

        if len(df.columns) < 2:
            raise DataFrameException('Data set has too few columns, should have at least two.')
        elif len(df.columns) == 2:
            descriptor_column_index = 1
        else:
            descriptor_column_index = MetaDataTools.identify_descriptor_column(df)[0]
            if descriptor_column_index < 0:
                raise DataFrameException('No descriptor column identified for DataFrame.')

        cleansed_df = df[df.columns[[0, descriptor_column_index]]]
        cleansed_df = MetaDataTools.cleanse_text_in_dataframe(cleansed_df)

        new_df = pd.DataFrame()

        new_df['Fields'] = cleansed_df[cleansed_df.columns[0]]
        new_df['TokenizedDescriptors'] = cleansed_df[cleansed_df.columns[1]]
        new_df.insert(0, 'Source', source)

        return new_df

    @staticmethod
    def field_descriptors_df_from_file(src_path: str, target_dir: str, prefix: str = '',
                                       to_save: bool = False) -> pd.DataFrame:
        """Create DataFrame of field descriptors from file

        :param src_path: Path to source file.
        :param target_dir: Path to directory for saving files.
        :param prefix: Optional string to use as a common prefix for saving files.
        :param to_save: Bool - whether to save the file to the working directory (default: False).
        :return: DataFrame of processed data.
        """
        df = MetaDataTools.read_raw_data(src_path)

        field_descriptors = MetaDataTools.field_tokenized_descriptor_df_from_df(df, Path(src_path).stem)
        if to_save:
            save_name = f'{prefix}_ProcessedDF {Path(src_path).stem}.tsv'
            save_path = os.path.join(Path(target_dir), save_name)
            # Open file with newline='' to prevent blank intermediate lines
            with open(save_path, 'w', encoding='utf-8', newline='') as outfile:
                outfile.write(field_descriptors.to_csv(sep='\t', index=False))
                print('Tokenized file saved to {}.'.format(save_path))

        return field_descriptors

    @staticmethod
    def dict_of_field_descriptors_dfs_from_files(src_path: str, target_dir: str, prefix: str = '',
                                                 to_save: bool = False, suffix: str = '.txt') -> (dict, list):
        """Process files in folder to generate a dictionary of DataFrames of fields vs tokenized descriptors

        :param src_path: Source path to directory holding files to process.
        :param target_dir: Folder where temporary and final files are to be saved.
        :param prefix: Optional string for prefixing the final filename.
        :param to_save: Whether to save the file to the working directory (default: False).
        :param suffix: Suffix of source files (default: .txt).
        :return: Dict, List. Dictionary of DataFrames and list of files with errors.
        """
        df_dict = {}
        errors = []
        for root, dirs, files in os.walk(src_path, topdown=False):
            for file_name in [file_name for file_name in files if Path(file_name).suffix == suffix]:
                file_path = os.path.join(src_path, file_name)
                try:
                    df = MetaDataTools.field_descriptors_df_from_file(file_path, target_dir, prefix, to_save=to_save)
                    key = Path(file_path).name
                    df_dict[key] = df
                except DataFrameException as ex:
                    errors.append([file_path, ex])

        return df_dict, errors

    @staticmethod
    def list_of_field_descriptors_dfs_from_files(src_path: str, target_dir: str, prefix: str = '',
                                                 to_save: bool = False, suffix: str = '.txt') -> (list, list):
        """Process files in folder to generate a list of DataFrames of fields vs tokenized descriptors

        :param src_path: Source path to directory holding files to process.
        :param target_dir: Folder where temporary and final files are to be saved.
        :param prefix: Optional string for prefixing the final filename.
        :param to_save: Whether to save the file to the working directory (default: False).
        :param suffix: Suffix of source files (default: .txt).
        :return: List, List. List of DataFrames and list of files with errors.
        """
        df_list = []
        errors = []
        for root, dirs, files in os.walk(src_path, topdown=False):
            for file_name in [file_name for file_name in files if Path(file_name).suffix == suffix]:
                file_path = os.path.join(src_path, file_name)
                try:
                    df = MetaDataTools.field_descriptors_df_from_file(file_path, target_dir, prefix, to_save=to_save)
                    df_list.append(df)
                except DataFrameException as ex:
                    errors.append([file_path, ex])

        return df_list, errors

    @staticmethod
    def save_dfs(df_list: list, save_dir: str, save_name: str, prefix: str = ''):
        """Save list of DataFrames

        :param df_list: list of DataFrames
        :param save_dir: target directory
        :param save_name: target file name
        :param prefix: optional prefix to main file name
        """

        if len(df_list) > 0:
            collate_dfs = pd.concat(df_list)
            collate_dfs.reset_index(inplace=True, drop=True)

            save_path = os.path.join(save_dir, f'{prefix}{save_name}')
            # Open file with newline='' to prevent blank intermediate lines
            with open(save_path, 'w', encoding='utf-8', newline='') as outfile:
                outfile.write(collate_dfs.to_csv(sep='\t', index=False))
                print('Data from dataframes  saved to {}.'.format(save_path))
        else:
            print('No data frames saved.')
