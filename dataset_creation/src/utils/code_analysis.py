import os
import csv
import logging
import pandas as pd
import re

ccsr_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/input_files/DXCCSR_v2021-2.csv'
swedish_path = '/pvc/Stockholm EPR ICD-10 Corpus_14_nov_2022_removed_personnummer.tsv'
ccsr_dict = {}

def parse_swedish():

    # Open CSV file as a pandas dataframe
    df = pd.read_csv(swedish_path)

    # Split codes column by ',' and flatten the resulting list
    unique_codes = df['codes'].str.split(',').explode().unique().tolist()
    # Remove non-alphanumeric characters from each element
    unique_codes = [re.sub(r'[^a-zA-Z0-9]', '', code) for code in unique_codes]
    # Initialize match and miss counters, and result dictionary
    matches = 0
    misses = 0
    swe_out = {}
    swe_misses = []

    # Check if each cleaned code exists in the ccsr_dict
    for code in unique_codes:
        if code in ccsr_dict:
            swe_out[code] = ccsr_dict[code]
            matches += 1
        elif code[0:2] in ccsr_dict:
            swe_out[code[0:2]] = ccsr_dict[code[0:2]]
            matches += 1
        elif code[0:3] in ccsr_dict:
            swe_out[code[0:3]] = ccsr_dict[code[0:3]]
            matches += 1
        else:
            swe_misses.append(code)
            misses += 1

    # Print the number of matches and misses, and the resulting dictionary
    print(swe_out)
    print(f"Matches: {matches}, Misses: {misses}")
    print(swe_misses)
    unique_values = set(swe_out.values())
    print(unique_values)
    print(f"The number of unique Swedish CCSR codes is: {len(unique_values)}")

    #print(unique_codes)

if __name__ == "__main__":
    with open(ccsr_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ccsr_dict[row["'ICD-10-CM CODE'"].replace("'", '')[0:3]] = row["'Default CCSR CATEGORY DESCRIPTION IP'"]

    #print(ccsr_dict)
    parse_swedish()        





   