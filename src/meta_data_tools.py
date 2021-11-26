# meta_data_tools.py
import nltk
import pandas as pd
import re
import string
import sys


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
        for column in df.columns:
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
            raise Exception('Data set has too few columns.')
        elif len(df.columns) == 2:
            descriptor_column_index = 1
        else:
            descriptor_column_index = MetaDataTools.identify_descriptor_column(df)[0]
            if descriptor_column_index < 0:
                raise Exception('No descriptor column identified for DataFrame.')

        field_names = df[df.columns[0]]
        descriptions = [MetaDataTools.cleanse_text(text) for text in df[df.columns[descriptor_column_index]]]

        return [field_names, descriptions]
