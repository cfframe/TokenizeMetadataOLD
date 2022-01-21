# tokenize_meta_data.py
import argparse
import datetime

from src.file_tools import FileTools
from src.meta_data_tools import MetaDataTools
import os
import pandas as pd
from pathlib import Path

TEST_FILEPATH = os.path.join(Path(__file__).parent, 'test', 'test_data', 'test_xls.xlsx')

"""
Description: Extract fields, tokenized descriptors and labels from data dictionary. 

Example usage:
py tokenize_labelled_meta_data.py -s "C:/temp/TableMetaData/Source/FCRB_Data Model_v0.5 CFF 1g.xlsm" -td C:/temp/TableMetaData/Results 
py tokenize_labelled_meta_data.py -s "C:/temp/TableMetaData/Source/FCRB_Data Model_v0.5 CFF 1g.xlsm" -td C:/temp/TableMetaData/Results -ot bert
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Extract fields and tokenized descriptors from data dictionary.')
    parser.add_argument('-s', '--src_path', type=str, default=TEST_FILEPATH,
                        help='Source path for processing.')
    parser.add_argument('-td', '--target_dir', type=str, default=Path(__file__).parent,
                        help='Working directory for saving files etc')
    parser.add_argument('-ot', '--output_type', type=str, default='tokenized',
                        help='Output type (tokenized, bert')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # Declare args - helps with auto-completion. Convert to object?
    src_path = args.src_path
    target_dir = args.target_dir
    output_type = args.output_type

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

    # Word separator for tokenized text - default
    token_sep = ','
    column_save_sep = '\t'
    if output_type == 'bert':
        token_sep = ' '
        column_save_sep = ','

    df_list = [MetaDataTools.field_tokenized_descriptor_df_from_df(df=v, source=k, is_labelled=True, sep=token_sep)
               for k, v in df_dict.items()
               if k != 'Status list']
    save_name = f'labelled.txt'

    # If default output_type (i.e. 'tokenized') then leave as is.
    df = MetaDataTools.collate_dfs_from_list(df_list=df_list)
    if output_type == 'bert':
        df = MetaDataTools.prep_df_for_bert(df)

    MetaDataTools.save_df(df=df, save_name=save_name, save_dir=target_dir, prefix=prefix, sep=column_save_sep)


if __name__ == '__main__':
    main()
