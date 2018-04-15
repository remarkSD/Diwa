###
# SCRIPT: clean_data.py
# ABOUT THIS SCRIPT:
# This script outputs a .TSV file in the prescribed translation data format
# as agreed upon by the Deep Learning UP EEEI class (2sAY2017-2018)
# given an input .tsv file:
#   <english phrase/sentence> /t <filipino translation>
#
###

import os
import re
import sys

def clean_tsv_data(input_filename = "english-tagalogDB_School.tsv", output_filename = None):
    if output_filename is None:
        output_filename = os.path.splitext(input_filename)[0] + "_CLEANED.tsv"
    eos_regex = "[\t]+"

    print('Processing input TSV: %s...' % input_filename)

    input_file = open(input_filename, "r", encoding="utf8")
    output_file = open(output_filename, "w+", encoding="utf8")
    input_file_lines = input_file.readlines()
    for line in input_file_lines:
        split_line = re.split(eos_regex, line)
        entry_number = 0
        for entry in split_line:
            if '' == entry.rstrip():
                continue
            output_file.write(entry.rstrip())
            print('.', end='')
            if 0 == entry_number % 2:
                output_file.write("\t")
            else:
                output_file.write("\n")
            entry_number = entry_number + 1

    print('\nProcessing complete! See output TSV: %s' % output_filename)

def print_usage():
    print('USAGE: python3 clean_data.py <input.tsv> <output.tsv>')
    print('\t<input.tsv> : filename of the input TSV (REQUIRED)')
    print('\t<output.tsv> : filename of the output (OPTIONAL)')
    print('\t\tDefault: <input.tsv>_CLEANED.tsv')

if __name__ == "__main__":
    if 1 >= len(sys.argv):
        print_usage()
        sys.exit(1)
    input_filename = sys.argv[1]
    if 2 < len(sys.argv):
        output_filename = sys.argv[2]
    else:
        output_filename = os.path.splitext(input_filename)[0] + "_CLEANED.tsv"

    clean_tsv_data(input_filename, output_filename)
    sys.exit(0)


