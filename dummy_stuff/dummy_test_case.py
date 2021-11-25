# dummy_test_case.py
from pathlib import Path
import pandas as pd
import os
import unittest
from src.meta_data_tools import MetaDataTools as MDT

import nltk


class MetaDataToolsTestCase(unittest.TestCase):

    def test_dummy(self):
        wn = nltk.WordNetLemmatizer()
        ps = nltk.PorterStemmer()

        words = ['describes', 'describe', 'descriptor', 'description']

        # Using list first is computationally cheaper
        ls = []
        for word in words:
            new_row = [word, ps.stem(word), wn.lemmatize(word)]
            ls.append(new_row)

        columns = ['word', 'PorterStemmer', 'WordNetLemmatizer']
        df = pd.DataFrame(ls, columns=columns)
        print(df)

        return True


if __name__ == '__main__':
    unittest.main()
