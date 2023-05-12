import os
import csv
import logging
import pandas as pd
import re
from matplotlib import pyplot as plt
import plotly.graph_objects as go


ccsr_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/input_files/DXCCSR_v2021-2.csv'
swedish_path = '/pvc/Stockholm EPR ICD-10 Corpus_14_nov_2022_removed_personnummer.tsv'
mimic_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/src/mimic/mimic_tmp.csv'
ahepa_paths = ['/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/mimic_codiesp_filtered_CCS_dev.csv', '/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/achepa_codiesp_filtered_CCS_fold_1_test.csv', '/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/achepa_codiesp_filtered_CCS_fold_1_train.csv']
codie_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/css_codie_labels.txt'
brazilian_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/src/brazilian/brazilian_tmp_4.csv'
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
    # print(swe_out)
    # print(f"Matches: {matches}, Misses: {misses}")
    # print(swe_misses)
    unique_values = set(swe_out.values())
    #print(unique_values)
    print(f"The number of unique Swedish CCSR codes is: {len(unique_values)}")
    return list(unique_values)

def parse_mimic():

    # Open CSV file as a pandas dataframe
    df = pd.read_csv(mimic_path)

    # Split codes column by ',' and flatten the resulting list
    unique_codes = df['ICD10'].str.split(',').explode().unique().tolist()
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
    # print(swe_out)
    # print(f"Matches: {matches}, Misses: {misses}")
    # print(swe_misses)
    unique_values = set(swe_out.values())
    #print(unique_values)
    print(f"The number of unique MIMIC CCSR codes is: {len(unique_values)}")
    return list(unique_values)

def parse_ahepa():

    # create an empty dataframe
    df = pd.DataFrame()

    # loop through the file paths and read each csv file into a dataframe, then concatenate them
    for path in ahepa_paths:
        temp_df = pd.read_csv(path)
        df = pd.concat([df, temp_df], ignore_index=True)

    # use the `eval()` function to convert string representation of a list into a list
    df['labels'] = df['labels'].apply(eval)

    # explode the 'labels' column to create a new row for each element in the list
    exploded_df = df.explode('labels')

    # get the unique values from the 'labels' column
    unique_labels = exploded_df['labels'].unique()
    #print(unique_labels.tolist())
    print(f"The number of unique Ahepa CCSR codes is: {len(unique_labels)}")
    return unique_labels.tolist()

def parse_codie():

    codie_ccsr = []
    with open(codie_path, 'r') as f:
        for row in f:
            codie_ccsr.append(row.strip())
    #print(codie_ccsr)
    print(f"The number of unique Codie CCSR codes is: {len(codie_ccsr)}")
    return codie_ccsr

def parse_brazilian():

    df = pd.read_csv(brazilian_path)

    # use the `eval()` function to convert string representation of a list into a list
    df['labels'] = df['labels'].apply(eval)

    # explode the 'labels' column to create a new row for each element in the list
    exploded_df = df.explode('labels')

    # get the unique values from the 'labels' column
    unique_labels = exploded_df['labels'].unique()
    #print(unique_labels.tolist())
    print(f"The number of unique Brazilian CCSR codes is: {len(unique_labels)}")
    return unique_labels.tolist()


if __name__ == "__main__":
    with open(ccsr_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ccsr_dict[row["'ICD-10-CM CODE'"].replace("'", '')[0:3]] = row["'Default CCSR CATEGORY DESCRIPTION IP'"]

    #print(ccsr_dict)
    #parse_swedish() 
    #parse_mimic() 
    #parse_ahepa() 
    #parse_codie()       
    brazilian = parse_brazilian()
    mimic = parse_mimic()
    swedish = parse_swedish()
    ahepa = parse_ahepa()
    codie = parse_codie()

    union_list = brazilian + mimic + swedish + ahepa + codie
    union_set = set(union_list)


    brazilian = set(brazilian)
    mimic = set(mimic)
    swedish = set(swedish)
    ahepa = set(ahepa)
    codie = set(codie)

    intersection_set = set(brazilian.intersection(mimic, swedish, ahepa, codie))

    print(f"The number of total CCSR codes is: {len(union_set)}")
    print(f"The number of shared CCSR codes (intersection) is: {len(intersection_set)}")







   