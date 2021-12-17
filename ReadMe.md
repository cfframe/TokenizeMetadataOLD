# ReadMe for WP2.6
## General Python project basics
Assume using a virtual environment and python 3.8.10.

Python package requirements defined in requirements.txt. Installation **can take a few minutes**, 
especially due to NLTK download.

One way of installing requirements: <br />
<code>py -m pip install --upgrade pip</code><br />
<code>pip install -r requirements.txt</code>

## Usage
### tokenize_meta_data.py
Extract fields, labels and tokenized descriptors from a TSV file or an Excel file.

Parameters:

-s, --src_path: Source path for processing. Assume a file, but use is_directory flag if a folder. 
Default: TEST_FILEPATH, as defined in tokenize_meta_data.py.  

-d, --is_directory: Indicates src_path is a directory and not a file. Default: true

-td, --target_dir: Working directory for saving files etc. Default: parent directory of tokenize_meta_data.py.

-ext, --suffix: Working directory for saving files etc. Default: '.txt'.

Example:<br />
<code>
py tokenize_meta_data.py -s "C:/temp/TableMetaData/Source/SAP IS-H Case Attribute.txt" -td C:/temp/TableMetaData/Results
</code>

### tokenize_labelled_meta_data.py
Extract fields, tokenized descriptors and labels from data dictionary (Excel workbook) and output a tokenized version 
as a TSV file.

Expected format of the input Excel workbook:
- Worksheet 'Status list'. This is ignored in processing.
- All other worksheets have a grid of fields, with these columns:
  - first column: field names
  - a subsequent column: descriptors (with a name similar to 'description' or 'descriptor')
  - last column: labels
  - all other columns are ignored.

Parameters:

-s, --src_path: Source path for processing. Default: TEST_FILEPATH, as defined in tokenize_labelled_meta_data.py.  

-td, --target_dir: Working directory for saving files etc. Default: parent directory of tokenize_labelled_meta_data.py.

Example:<br />
<code>
py tokenize_labelled_meta_data.py -s "C:/temp/TableMetaData/Source/FCRB_Data Model_v0.5 CFF 1g.xlsm" -td C:/temp/TableMetaData/Results
</code>
