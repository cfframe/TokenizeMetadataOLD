# meta_data_tools.py
import string
import sys

import pandas as pd

class MetaDataTools:
    """Static methods to work with Meta Data"""

    @staticmethod
    def remove_punctuation_from_text(text: str) -> str:
        """Remove punctuation from text
        :param text: source text to process
        """
        text_no_punctuation = ''.join([char for char in text if char not in string.punctuation])
        return text_no_punctuation

    @staticmethod
    def remove_punctuation_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        new_df = pd.DataFrame()
        for column in df.columns:
            new_df[column] = df[column].apply(lambda x: MetaDataTools.remove_punctuation_from_text(str.lower(x)))

        return new_df

    @staticmethod
    def read_raw_data(source_path: str) -> pd.DataFrame:
        raw_data = pd.read_csv(source_path, sep='\t', header='infer')
        return raw_data

    @staticmethod
    def identify_descriptor_column(df: pd.DataFrame) -> str:
        for column in df.columns:
            column = str.lower(column)
        return 'STILL TO DO'
