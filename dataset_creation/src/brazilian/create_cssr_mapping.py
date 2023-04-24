import os
import csv

input_path = '/pvc/UMLSParser/CIDoutputs copy'
#input_path = '/pvc/brazilian/MATCHED copy'
labels_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/css_codie_labels.txt'

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
    for file_name in os.listdir(input_path):
        if file_name.endswith('.csv'):
            case_found = False
            file_path = os.path.join(input_path, file_name)
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f, delimiter=';')
                fieldnames = reader.fieldnames + ['CSSR']
                rows = []
                for i, row in enumerate(reader):
                    icd10_code = row['ICD10'].strip().replace('.', '').replace("'", '')
                    if icd10_code and icd10_code[0] == '{':
                        print(icd10_code)
                        for i in range(len(icd10_code)):
                            for j in range(i+1, len(icd10_code)+1):
                                # check if the current substring exists in ccsr_dict, with checks
                                if icd10_code[i:j] in ccsr_dict and (j == len(icd10_code) or icd10_code[j] in {',', '}'}):
                                    ccsr_category = ccsr_dict[icd10_code[i:j]]
                                    num_matches += 1
                                    match_locations.append(f"{file_name} at row {i}: {icd10_code}")
                                    print(f"Match found in file {file_name} at row {i+1}: {ccsr_category}")
                                    with open(labels_path, 'r') as f:
                                        for line in f:
                                            if ccsr_category.rstrip() == line.rstrip():
                                                row['CSSR'] = ccsr_category.rstrip()
                                                codie_matches += 1
                                                codie_match_categories.append(ccsr_category)
                                                case_found = True
                                    break
                    elif icd10_code in ccsr_dict:
                        num_matches += 1
                        match_locations.append(f"{file_name} at row {i}: {icd10_code}")
                        print(f"Match found in {file_name} at row {i} for {icd10_code}: {ccsr_dict[icd10_code]}")
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
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                    writer.writeheader()
                    writer.writerows(rows)
            if case_found == True:
                cases_counter += 1
    
                        
    print("Match locations:")
    for location in match_locations:
        print(location)
    for location in codie_match_categories:
        print(location)
    for miss in missed_CIDS:
        print(f"{miss} was missed")
    print(f"Total matches found: {num_matches}")    
    print(f"Total codie_labels found: {codie_matches}")
    
    unique_items = set(codie_match_categories)
    num_unique = len(unique_items)
    print(f"Number of unique codes out of codie matches: {num_unique}") 

    print(f"Number of documents which had at least one match: {cases_counter}")

    print(f"There were {len(missed_CIDS)} CIDs still unmatched")

    # Total matches found: 919
    # Total codie_labels found: 914
    # Number of unique codes out of codie matches: 77
    # Number of documents which had at least one match: 497

    # 17.04.23
    # Total matches found: 1331
    # Total codie_labels found: 1317
    # Number of unique codes out of codie matches: 95
    # Number of documents which had at least one match: 616 (+24%)

    # With shortened key_dicts
    # Total matches found: 1901
    # Total codie_labels found: 1677
    # Number of unique codes out of codie matches: 141
    # Number of documents which had at least one match: 667
    # There were 35 CIDs still unmatched

    # Latest:
    # Total matches found: 1929
    # Total codie_labels found: 1692
    # Number of unique codes out of codie matches: 141
    # Number of documents which had at least one match: 669 (out of 716)
    # There were 7 CIDs still unmatched (+35%)

if __name__ == '__main__':
    parse_csv_files(input_path)
