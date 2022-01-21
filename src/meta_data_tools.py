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
    def cleanse_text_in_dataframe(df: pd.DataFrame, columns_to_lower: list, columns_to_tokenize: list,
                                  sep: str = ',') -> pd.DataFrame:
        """
        Cleanse textual data held in a DataFrame. Excludes column names.

        Assumptions:
        - Always has columns 'Fields'
        - Optionally has 1 or more of 'Source' and 'Labels'

        :param df: DataFrame to process.
        :param columns_to_tokenize: list of column indices to tokenize.
        :param columns_to_lower: list of column indices to de-capitalise text.
        :param sep: separator for words in cleansed text.
        :returns: DataFrame with text cleansed.
        """
        new_df = pd.DataFrame()

        for i in columns_to_lower:
            new_df[df.columns[i]] = df[df.columns[i]].apply(lambda x: str.lower(str(x)))
        for i in columns_to_tokenize:
            new_df[df.columns[i]] = df[df.columns[i]]
            new_df['Tokenized ' + df.columns[i]] \
                = df[df.columns[i]].apply(lambda x: sep.join(MetaDataTools.cleanse_text(str(x))))

        # Arrange columns in logical order
        if 'Source' in new_df.columns:
            col = new_df.pop('Source')
            new_df.insert(0, col.name, col)

        if 'Labels' in new_df.columns:
            col = new_df.pop('Labels')
            new_df[col.name] = col

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
    def field_tokenized_descriptor_list_from_df(df: pd.DataFrame, is_labelled: bool = False) -> list:
        """Derive a list of field names against descriptions from a DataFrame.

        Assume that Field names are the first column, and that if only two columns then
        the second column is the descriptors.

        :param df: Source DataFrame.
        :param is_labelled: Whether DataFrame includes labels column. Assumed to be last column.
        :returns: See description.
        """
        min_column_count = 2
        if is_labelled:
            min_column_count = 3

        if len(df.columns) < min_column_count:
            raise DataFrameException(f'Data set has too few columns, should have at least {min_column_count}.')
        elif len(df.columns) == min_column_count:
            descriptor_column_index = 1
        else:
            descriptor_column_index = MetaDataTools.identify_descriptor_column(df)[0]
            if descriptor_column_index < 0:
                raise DataFrameException('No descriptor column identified for DataFrame.')

        field_names = df[df.columns[0]]
        descriptions = [','.join(MetaDataTools.cleanse_text(text)) for text in df[df.columns[descriptor_column_index]]]
        result = [field_names, descriptions]

        if is_labelled:
            labels = df[df.columns[min_column_count - 1]]
            result.append(labels)

        return result

    @staticmethod
    def field_tokenized_descriptor_df_from_df(df: pd.DataFrame, source: str, is_labelled: bool = False, sep: str = ',') -> pd.DataFrame:
        """Derive a reduced DataFrame of field names against tokenized descriptions from a source DataFrame.

        Assume that Field names are the first column, and that if only a minimum number of columns (2 if not labelled,
        3 if labelled) then the second column is the descriptors.

        If designated as labelled, assume the last column is the labels.

        :param df: Source DataFrame, added as first column in DataFrame.
        :param source: Source of data.
        :param is_labelled: Whether labelled (Default: False)
        :param sep: String separator in tokenized text. (Default ',')
        :returns: DataFrame. See description.
        """

        min_column_count = 2
        if is_labelled:
            min_column_count = 3

        if len(df.columns) < min_column_count:
            raise DataFrameException('Data set has too few columns, should have at least two.')
        elif len(df.columns) == min_column_count:
            descriptor_column_index = 1
        else:
            descriptor_column_index = MetaDataTools.identify_descriptor_column(df)[0]
            if descriptor_column_index < 0:
                raise DataFrameException('No descriptor column identified for DataFrame.')

        # Once fully generated, cleansed_df will have these columns, potentially with different names:
        # - 0 - Source
        # - 1 - Fields
        # - 2 - Descriptors
        # - 3 - [optional] Labels
        df_to_cleanse = pd.DataFrame()
        df_to_cleanse['Fields'] = df[df.columns[0]]
        df_to_cleanse['Descriptors'] = df[df.columns[descriptor_column_index]]

        columns_to_lower = [1]
        columns_to_tokenize = [0, 2]
        if is_labelled:
            df_to_cleanse['Labels'] = df[df.columns[len(df.columns) - 1]]
            columns_to_lower.append(3)
        df_to_cleanse.insert(0, 'Source', source)

        new_df = MetaDataTools.cleanse_text_in_dataframe(df_to_cleanse, columns_to_lower, columns_to_tokenize, sep)

        return new_df

    @staticmethod
    def field_descriptors_df_from_file(src_path: str, target_dir: str, prefix: str = '',
                                       to_save: bool = False) -> pd.DataFrame:
        """Create DataFrame of field descriptors from file

        :param src_path: Path to source file.
        :param target_dir: Path to directory for saving files.
        :param prefix: String to use as a common prefix for saving files (default: '').
        :param to_save: Whether to save the file to the working directory (default: False).
        :return: DataFrame of processed data.
        """
        df = MetaDataTools.read_raw_data(src_path)

        field_descriptors = MetaDataTools.field_tokenized_descriptor_df_from_df(df, Path(src_path).stem)
        if to_save:
            save_name = f'{prefix}_ProcessedDF {Path(src_path).stem}.txt'
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
        :param prefix: String for prefixing the final filename (default: '').
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
        :param prefix: String for prefixing the final filename (default: '').
        :param to_save: Whether to save the file to the working directory (default: False).
        :param suffix: Suffix of source files (default: '.txt').
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
    def collate_dfs_from_list(df_list: list) -> pd.DataFrame:
        """Save list of DataFrames

        :param df_list: List of DataFrames. Expect all to have same number of columns.
        :return: Collation of list elements as a single DataFrame.
        """

        collated_dfs = pd.DataFrame()

        if len(df_list) > 0:

            collated_dfs = pd.concat(df_list)
            collated_dfs.reset_index(inplace=True, drop=True)

        return collated_dfs

    @staticmethod
    def save_df(df: pd.DataFrame, save_dir: str = '', save_name: str = '', prefix: str = '', sep: str = '\t'):
        """Save list of DataFrames

        :param df: DataFrame.
        :param save_dir: Target directory (default: '').
        :param save_name: Target file name (default: '').
        :param prefix: Prefix to main file name (default: '').
        :param sep: Separator (default: '\t')
        """

        if len(df) > 0:

            if len(save_dir) > 0 and len(save_name) > 0:
                save_path = os.path.join(save_dir, f'{prefix}{save_name}')
                # Open file with newline='' to prevent blank intermediate lines
                with open(save_path, 'w', encoding='utf-8', newline='') as outfile:
                    outfile.write(df.to_csv(sep=sep, index=False))
                    print('Data from DataFrames saved to {}.'.format(save_path))
            else:
                print('No DataFrames saved.')
        else:
            print('No DataFrames saved.')

        return df

    @staticmethod
    def prep_df_for_bert(df: pd.DataFrame) -> pd.DataFrame:
        """Save list of DataFrames

        :param df: DataFrame.
        :return: DataFrame, first column is category and second is text.
        """
        new_df = pd.DataFrame()
        new_df['category'] = df['Labels']

        new_df['text'] = df[['Tokenized Source', 'Fields', 'Tokenized Descriptors']].agg(' '.join, axis=1)

        return new_df

