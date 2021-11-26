# main.py
import argparse
import datetime
from src.file_tools import FileTools
from src.meta_data_tools import MetaDataTools
import os
from pathlib import Path

TEST_FILEPATH = os.path.join(Path(__file__).parent, 'tests', 'test_data', 'test_tsv_4_cols.txt')

"""
Description: Read in a data dictionary file and output a tokenized version.

Example usage:
py main.py -sf "C:/temp/TableMetaData/Source/SAP IS-H Case Attribute.txt" -wd C:/temp/TableMetaData/Results 
py main.py -sf "C:/temp/TableMetaData/Source" -d -wd C:/temp/TableMetaData/Results 
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Extract fields and tokenized descriptors from data dictionary.')
    parser.add_argument('-sf', '--src_path', type=str, default=TEST_FILEPATH,
                        help='Source path for processing. Assume a file, but use is_directory flag if a folder.')
    parser.add_argument('-d', '--is_directory', action='store_true',
                        help='Indicates src_path is a directory and not a file')
    parser.add_argument('-wd', '--working_dir', type=str, default=Path(__file__).parent,
                        help='Working directory for downloads etc (optional)')

    args = parser.parse_args()

    return args


def process_file(src_path: str, working_dir: str, runtime_string: str):
    df = MetaDataTools.read_raw_data(src_path)

    field_descriptors = MetaDataTools.field_tokenized_descriptor_df_from_df(df)
    save_name = f'{runtime_string}_ProcessedDF {Path(src_path).stem}.txt'
    save_path = os.path.join(Path(working_dir), save_name)
    # Open file with newline='' to prevent blank intermediate lines
    with open(save_path, 'w', encoding='utf-8', newline='') as outfile:
        outfile.write(field_descriptors.to_csv(sep='\t'))
        print('Tokenized file saved to {}.'.format(save_path))

    return


def main():
    args = parse_args()
    # Declare args - helps with auto-completion. Convert to object?
    src_path = args.src_path
    working_dir = args.working_dir
    is_directory = args.is_directory

    runtime_string = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    command_filename = f'{runtime_string} Command args.txt'
    FileTools.save_command_args_to_file(vars(args),
                                        save_path=os.path.join(Path(working_dir).parent, command_filename),
                                        to_print=True
                                        )

    if not is_directory:
        process_file(src_path, working_dir, runtime_string)
    else:
        for root, dirs, files in os.walk(src_path, topdown=False):
            for file in files:
                file_path = os.path.join(src_path, file)
                process_file(file_path, working_dir, runtime_string)


if __name__ == '__main__':
    main()
