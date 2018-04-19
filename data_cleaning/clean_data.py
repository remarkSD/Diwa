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

def clean_tsv_data(input_filenames, output_filename):
    eos_regex = "[\t]+"

    output_file = open(output_filename, "w+", encoding="utf8")
    num_lines_in_output_file = 0

    for input_filename in input_filenames:
        print('\n\nProcessing input TSV: %s...' % input_filename)
        input_file = open(input_filename, "r", encoding="utf8")
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
                    num_lines_in_output_file = num_lines_in_output_file + 1
                entry_number = entry_number + 1

    print('\n\nProcessing complete! See output TSV: %s (%d translations)' %(output_filename, num_lines_in_output_file))

def print_usage():
    print('USAGE: python3 clean_data.py <input1.tsv> <input2.tsv> ... <output.tsv>')
    print('\t<input.tsv> : filename of the input TSV (REQUIRED)')
    print('\t<output.tsv> : filename of the output (REQUIRED)')
    print('\t\tDefault: <input.tsv>_CLEANED.tsv')

if __name__ == "__main__":
    if 2 >= len(sys.argv):
        print_usage()
        sys.exit(1)
    output_filename = sys.argv[len(sys.argv) - 1]
    input_filenames = sys.argv[1:len(sys.argv) - 1]

    clean_tsv_data(input_filenames, output_filename)
    sys.exit(0)


