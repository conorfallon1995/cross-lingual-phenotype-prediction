import os
import csv
import logging

input_path = '/pvc/UMLSParser/CIDoutputs copy'
#input_path = '/pvc/brazilian/MATCHED copy'
labels_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/css_codie_labels.txt'
#file_path = '/pvc/swedish_mini.tsv'
file_path = '/pvc/Stockholm EPR ICD-10 Corpus_14_nov_2022_removed_personnummer.tsv'
file_name = file_path
out_file_path = '/pvc/test_out_swedish.tsv'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def parse_csv_files(input_path):
    ccsr_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/input_files/DXCCSR_v2021-2.csv'
    ccsr_dict = {}
    with open(ccsr_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ccsr_dict[row["'ICD-10-CM CODE'"].replace("'", '')] = row["'Default CCSR CATEGORY DESCRIPTION IP'"]
    
    # make a dict of three digit truncated ccsr codes
    three_cssr = {key[:3]: value for key, value in ccsr_dict.items()}

    # make a dict of four digit truncated ccsr codes
    four_cssr = {key[:4]: value for key, value in ccsr_dict.items()}

    num_matches = 0
    codie_matches = 0
    cases_counter = 0
    match_locations = []
    codie_match_categories = []
    missed_CIDS = []
    case_found = False
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        fieldnames = reader.fieldnames + ['CSSR']
        rows = []
        for i, row in enumerate(reader):
            icd10_code = row['codes'].strip().replace('.', '').replace("'", '')
            #Want to remove spaces in between
            icd10_code = "".join(icd10_code.split())
            # Must deal with the situation where a list is inputted, not a single ICD10 code
            if icd10_code.count(',') > 0:
                icd10_list = icd10_code.split(",")
                for code in icd10_list:
                    if code in ccsr_dict:
                        num_matches += 1
                        match_locations.append(f"{file_name} at row {i}: {code}")
                        print(f"Match found in {file_name} at row {i} for {code}: {ccsr_dict[code]}")
                        logging.info(f"Match found in {file_name} at row {i} for {code}: {ccsr_dict[code]}")
                        with open(labels_path, 'r') as f:
                            for line in f:
                                if ccsr_category.rstrip() == line.rstrip():
                                    row['CSSR'] = ccsr_category.rstrip()
                                    codie_matches += 1
                                    codie_match_categories.append(ccsr_category)
                                    case_found = True
                                    break
                    elif code in four_cssr:
                        ccsr_category = four_cssr[code]
                        num_matches += 1
                        match_locations.append(f"{file_name} at row {i}: {code}")
                        print(f"Match found in {file_name} at row {i} for {code}: {four_cssr[code]}")
                        logging.info(f"Match found in {file_name} at row {i} for {code}: {four_cssr[code]}")
                        with open(labels_path, 'r') as f:
                            for line in f:
                                if ccsr_category.rstrip() == line.rstrip():
                                    row['CSSR'] = ccsr_category.rstrip()
                                    codie_matches += 1
                                    codie_match_categories.append(ccsr_category)
                                    case_found = True
                                    break
                    elif code[0:3] in three_cssr:
                        ccsr_category = three_cssr[code[0:3]]
                        num_matches += 1
                        match_locations.append(f"{file_name} at row {i}: {code}")
                        print(f"Match found in {file_name} at row {i} for {code}: {three_cssr[code[0:3]]}")
                        logging.info(f"Match found in {file_name} at row {i} for {code}: {three_cssr[code[0:3]]}")
                        with open(labels_path, 'r') as f:
                            for line in f:
                                if ccsr_category.rstrip() == line.rstrip():
                                    row['CSSR'] = ccsr_category.rstrip()
                                    codie_matches += 1
                                    codie_match_categories.append(ccsr_category)
                                    case_found = True
                                    break
                    else:
                        if icd10_code != '':
                            missed_CIDS.append(code)
                        row['CSSR'] = ''
                    rows.append(row)
                with open(out_file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                if icd10_code in ccsr_dict:
                    num_matches += 1
                    match_locations.append(f"{file_name} at row {i}: {icd10_code}")
                    print(f"Match found in {file_name} at row {i} for {icd10_code}: {ccsr_dict[icd10_code]}")
                    logging.info(f"Match found in {file_name} at row {i} for {icd10_code}: {ccsr_dict[icd10_code]}")
                    with open(labels_path, 'r') as f:
                        for line in f:
                            if ccsr_category.rstrip() == line.rstrip():
                                row['CSSR'] = ccsr_category.rstrip()
                                codie_matches += 1
                                codie_match_categories.append(ccsr_category)
                                case_found = True
                                break
                elif icd10_code in four_cssr:
                    ccsr_category = four_cssr[icd10_code]
                    num_matches += 1
                    match_locations.append(f"{file_name} at row {i}: {icd10_code}")
                    print(f"Match found in {file_name} at row {i} for {icd10_code}: {four_cssr[icd10_code]}")
                    logging.info(f"Match found in {file_name} at row {i} for {icd10_code}: {four_cssr[icd10_code]}")
                    with open(labels_path, 'r') as f:
                        for line in f:
                            if ccsr_category.rstrip() == line.rstrip():
                                row['CSSR'] = ccsr_category.rstrip()
                                codie_matches += 1
                                codie_match_categories.append(ccsr_category)
                                case_found = True
                                break
                elif icd10_code[0:3] in three_cssr:
                    ccsr_category = three_cssr[icd10_code[0:3]]
                    num_matches += 1
                    match_locations.append(f"{file_name} at row {i}: {icd10_code}")
                    print(f"Match found in {file_name} at row {i} for {icd10_code}: {three_cssr[icd10_code[0:3]]}")
                    logging.info(f"Match found in {file_name} at row {i} for {icd10_code}: {three_cssr[icd10_code[0:3]]}")
                    with open(labels_path, 'r') as f:
                        for line in f:
                            if ccsr_category.rstrip() == line.rstrip():
                                row['CSSR'] = ccsr_category.rstrip()
                                codie_matches += 1
                                codie_match_categories.append(ccsr_category)
                                case_found = True
                                break
                else:
                    if icd10_code != '':
                        missed_CIDS.append(icd10_code)
                    row['CSSR'] = ''
                rows.append(row)
            with open(out_file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
                writer.writeheader()
                writer.writerows(rows)
    if case_found == True:
        cases_counter += 1
    
                        
    print("Match locations:")
    logging.info("Match locations")
    for location in match_locations:
        print(location)
        logging.info(location)
    for location in codie_match_categories:
        print(location)
    # Get unique elements of the list
    unique_elements = list(set(missed_CIDS))
    # Open file for writing
    with open("swedish_missed.txt", "w") as f:
        # Write each unique element to file, one element per line
        for element in unique_elements:
            f.write(element + "\n")
    print(f"Total matches found: {num_matches}")
    logging.info(f"Total matches found: {num_matches}")    
    print(f"Total codie_labels found: {codie_matches}")
    logging.info(f"Total codie_labels found: {codie_matches}")
    
    unique_items = set(codie_match_categories)
    num_unique = len(unique_items)
    print(f"Number of unique codes out of codie matches: {num_unique}") 
    logging.info(f"Number of unique codes out of codie matches: {num_unique}") 

    print(f"Number of documents which had at least one match: {cases_counter}")
    logging.info(f"Number of documents which had at least one match: {cases_counter}")

    print(f"There were {len(missed_CIDS)} CIDs still unmatched")
    logging.info(f"There were {len(missed_CIDS)} CIDs still unmatched")

    # Total matches found: 338969
    # Total codie_labels found: 338752
    # Number of unique codes out of codie matches: 23
    # Number of documents which had at least one match: 1
    # There were 12761 CIDs still unmatched

    # Total matches found: 341552
    # Total codie_labels found: 341336
    # Number of unique codes out of codie matches: 23
    # Number of documents which had at least one match: 1
    # There were 10178 CIDs still unmatched

    # Latest:
    # Total matches found: 1929
    # Total codie_labels found: 1692
    # Number of unique codes out of codie matches: 141
    # Number of documents which had at least one match: 669 (out of 716)
    # There were 7 CIDs still unmatched (+35%)

if __name__ == '__main__':
    parse_csv_files(input_path)
