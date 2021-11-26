# main.py
import argparse
import csv
import datetime
from src.file_tools import FileTools
from src.meta_data_tools import MetaDataTools
import os
from pathlib import Path

TEST_FILEPATH = os.path.join(Path(__file__).parent, 'tests', 'test_data', 'test_tsv_4_cols.txt')

"""
Description: Read in a data dictionary file and output a tokenized version.

Example usage:
py main.py -sf "C:\temp\TableMetaData\Source\SAP IS-H Case Attribute.txt" -wd C:\temp\TableMetaData\Results 
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Extract fields and tokenized descriptors from data dictionary.')
    parser.add_argument('-sf', '--src_filepath', type=str, default=TEST_FILEPATH,
                        help="Source file for processing")
    parser.add_argument('-wd', '--working_dir', type=str, default=Path(__file__).parent,
                        help='Working directory for downloads etc (optional)')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # Declare args - helps with auto-completion. Convert to object?
    args.src_filepath = args.src_filepath
    args.working_dir = args.working_dir

    runtime_string = datetime.datetime.now().strftime('%y%m%d_%H%M%S_')
    command_filename = f'{runtime_string}Command args.txt'
    FileTools.save_command_args_to_file(vars(args),
                                        save_path=os.path.join(Path(args.working_dir).parent, command_filename),
                                        to_print=True
                                        )

    df = MetaDataTools.read_raw_data(args.src_filepath)

    field_descriptors = MetaDataTools.field_tokenized_descriptor_df_from_df(df)
    save_name = f'{runtime_string}{Path(args.src_filepath).stem}_ProcessedDF.txt'
    save_path = os.path.join(Path(args.working_dir), save_name)
    # Open file with newline='' to prevent blank intermediate lines
    with open(save_path, 'w', encoding='utf-8', newline='') as outfile:
        outfile.write(field_descriptors.to_csv(sep='\t'))
        print('2nd Tokenized file saved to {}.'.format(save_path))


if __name__ == '__main__':
    main()

