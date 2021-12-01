# tokenize_meta_data.py
import argparse
import datetime

from src.file_tools import FileTools
from src.meta_data_tools import MetaDataTools
import os
from pathlib import Path

TEST_FILEPATH = os.path.join(Path(__file__).parent, 'tests', 'test_data', 'test_tsv_5_cols_inc_labels.txt')

"""
Description: Read in a data dictionary file and output a tokenized version.

Example usage:
py tokenize_meta_data.py -sf "C:/temp/TableMetaData/Source/SAP IS-H Case Attribute.txt" -wd C:/temp/TableMetaData/Results 
py tokenize_meta_data.py -sf "C:/temp/TableMetaData/Source" -d -wd C:/temp/TableMetaData/Results 
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Extract fields, labels and tokenized descriptors from Excel file.')
    parser.add_argument('-s', '--src_path', type=str, default=TEST_FILEPATH,
                        help='Source path for processing. Assume a file, but use is_directory flag if a folder.')
    parser.add_argument('-d', '--is_directory', action='store_true',
                        help='Indicates src_path is a directory and not a file')
    parser.add_argument('-td', '--target_dir', type=str, default=Path(__file__).parent,
                        help='Working directory for saving files etc')
    parser.add_argument('-ext', '--suffix', type=str, default='.txt',
                        help='Working directory for saving files etc')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # Declare args - helps with auto-completion. Convert to object?
    src_path = args.src_path
    target_dir = args.target_dir
    is_directory = args.is_directory
    suffix = args.suffix

    if input(f'WARNING Will clear directory {target_dir} if it exists.\n'
             f'Continue (n and Enter, or just Enter to continue)?').lower() == 'n':
        print('Chosen to quit.')
        quit()

    FileTools.ensure_empty_directory(target_dir)

    prefix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    command_filename = f'{prefix} Command args.txt'
    FileTools.save_command_args_to_file(vars(args),
                                        save_path=os.path.join(Path(target_dir).parent, command_filename),
                                        to_print=True
                                        )
    if not is_directory:
        MetaDataTools.field_descriptors_df_from_file(src_path, target_dir, prefix, to_save=True)
    else:
        df_list, errors = \
            MetaDataTools.list_of_field_descriptors_dfs_from_files(
                src_path, target_dir, prefix, to_save=True, suffix=suffix)

        for err in errors:
            print(err)

        save_name = f'collated.tsv'

        collated_dfs = MetaDataTools.collate_dfs_from_list(df_list=df_list, save_name=save_name, save_dir=target_dir, prefix=prefix)

        if len(collated_dfs) > 0:
            print('First few records in collated DataFrames:\n')
            print(collated_dfs.head())


if __name__ == '__main__':
    main()
