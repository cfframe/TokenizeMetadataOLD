# tokenize_meta_data.py
import argparse
import datetime

from src.file_tools import FileTools
from src.meta_data_tools import MetaDataTools
import os
import pandas as pd
from pathlib import Path

TEST_FILEPATH = os.path.join(Path(__file__).parent, 'tests', 'test_data', 'test_xls.xlsx')

"""
Description: Extract fields, tokenized descriptors and labels from data dictionary. 

Example usage:
py tokenize_labelled_meta_data.py -s "C:/temp/TableMetaData/Source/FCRB_Data Model_v0.5 CFF 1g.xlsm" -td C:/temp/TableMetaData/Results 
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Extract fields and tokenized descriptors from data dictionary.')
    parser.add_argument('-s', '--src_path', type=str, default=TEST_FILEPATH,
                        help='Source path for processing.')
    parser.add_argument('-td', '--target_dir', type=str, default=Path(__file__).parent,
                        help='Working directory for saving files etc')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # Declare args - helps with auto-completion. Convert to object?
    src_path = args.src_path
    target_dir = args.target_dir

    prefix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    command_filename = f'{prefix} Command args.txt'
    FileTools.save_command_args_to_file(vars(args),
                                        save_path=os.path.join(Path(target_dir).parent, command_filename),
                                        to_print=True
                                        )

    # read file
    wb = pd.ExcelFile(src_path)

    df_dict = pd.read_excel(wb, sheet_name=None)
    df_dict.pop('Status list', None)

    df_list = [MetaDataTools.field_tokenized_descriptor_df_from_df(df=v, source=k, is_labelled=True)
               for k, v in df_dict.items()
               if k != 'Status list']
    save_name = f'labelled.tsv'
    MetaDataTools.collate_dfs_from_list(df_list=df_list, save_name=save_name, save_dir=target_dir, prefix=prefix)


if __name__ == '__main__':
    main()
