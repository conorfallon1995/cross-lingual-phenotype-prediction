import sys
sys.path.append('/pvc/cross-lingual-phenotype-prediction')
from dataset_creation.src.utils import mimic_utils
from dataset_creation.src.utils import utils
from dataset_creation.src.utils import build_dataset_spanish_english_experiment, train_test_split_experiment
from dataset_creation.src.utils import codie_utils
import pandas as pd
import os

#CCS_PATH = '/pvc/brazilian/MATCHED copy'
CCS_PATH = '/pvc/UMLSParser/CIDoutputs copy'
TEXT_PATH = '/pvc/brazilian/parsed_xmls/'

if __name__== '__main__':

    SELECTOR = 'PART_2'
    task = 'codie_CCS'
    icd_10_dxccsr_paths = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/input_files/DXCCSR_v2021-2.csv'
    mimic_src_path = '/pvc/connor/mimic-iii-clinical-database-1.4/'
    # created with GEMS (General Equivalence Mapping)
    diagnosis_icd9_icd10_mapper_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/input_files/diagnosis_icd9_icd10.pcl'
    #LABELS OUTPUT PATH
    labels_output_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/{}_labels.pcl'
    train_data_output_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/{}'
    
    
    if SELECTOR in ['PART_1', 'ALL']:
        #load mimic, filter , map icd9 to icd 10 and get relevant sections
        #brazilian_df = mimic_utils.mimic_map_icd9_icd10(diagnosis_icd9_icd10_mapper_path, mimic_src_path, mimic_labels_path=None)
        # Create an empty list to hold the data for the DataFrame
        data = []

        # Iterate through the CCS_PATH directory
        for filename in os.listdir(CCS_PATH):
            # Check if the file is a CSV file
            if filename.endswith(".csv"):
                # Extract the first 4 characters of the filename as the patient ID
                patient_id = filename[:4]
                # Load the CSV file into a pandas DataFrame
                ccs_data = pd.read_csv(os.path.join(CCS_PATH, filename), sep=';')
                # Extract any non-empty strings from the 'CSSR' column and append to 'labels'
                labels = [x for x in ccs_data['CSSR'].tolist() if isinstance(x, str) and x != ""]
                # Make sure there's no duplicates
                labels = list(set(labels))
                # Only add a row to the DataFrame if 'labels' is not empty
                if labels:
                    # Load the text data from the corresponding file in TEXT_PATH
                    with open(os.path.join(TEXT_PATH, f"{patient_id}.txt"), "r") as f:
                        notes = f.read()
                    # Create a dictionary representing a row of data and append to the list
                    data.append({"patient_id": patient_id, "notes": notes, "labels": labels})
                    #data.append({"patient_id": patient_id, "notes": notes, "labels": f'{labels}'})
        # Create the DataFrame from the list of data dictionaries
        df = pd.DataFrame(data, columns=["patient_id", "notes", "labels"])
        df.to_csv('brazilian_tmp_4.csv', index=False)
      
    if SELECTOR in ['PART_2', 'ALL']:

        brazilian_df = pd.read_csv('/pvc/cross-lingual-phenotype-prediction/dataset_creation/src/brazilian/brazilian_tmp_4.csv')

        if task == 'codie_CCS':
            codie_labels = codie_utils.load_codie_labels(labels_output_path)
            #mimic_df_notes = mimic_utils.map_filter_ccs(brazilian_df, codie_labels, icd_10_dxccsr_paths)
            mimic_df_notes = brazilian_df
            dataset_name = 'v3_brazilian_codiesp_filtered_CCS_'
            labels = codie_labels
        '''
        elif task == 'achepa_diagnoses': 
            achepa_labels = achepa_utils.load_achepa_labels(labels_output_path=labels_output_path)
            achepa_diag = achepa_utils.get_diagnosis_icd_mapper(achepa_icd_diagnosis_path)
            mimic_df_notes = mimic_utils.map_icd10_achepa_diag(mimic_df=mimic_df, 
                                                            achepa_diag=achepa_diag, 
                                                            achepa_labels=achepa_labels)
            labels = achepa_labels
            dataset_name = 'mimic_achepa_filtered_diagnoses'
        '''
        # create sorting of codes and get position in array for each code
        label_to_pos, pos_to_label = train_test_split_experiment.label_to_pos_map(labels)
        
        #create numpy matrix for labels patinets x labels
        mimic_labels_array = train_test_split_experiment.brazilian_label_to_tensor(mimic_df_notes, label_to_pos)

        #use stratified sampling to save train/dev/test
        train_test_split_experiment.stratified_sampling_multilearn(mimic_df_notes, 
                                                            mimic_labels_array, 
                                                            train_data_output_path.format(dataset_name))
        #train_test_split_experiment.load_mimic_paper_split(mimic_df_notes, train_data_output_path.format(dataset_name))

