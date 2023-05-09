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
#SWE_PATH = '/pvc/swe100k.tsv'
#SWE_PATH = '/pvc/test_out_swedish.tsv'
SWE_PATH = '/pvc/output.tsv'


if __name__== '__main__':

    # Use PART_3 to get smaller samples such that all datasets of equal size
    SELECTOR = 'PART_2'
    task = 'codie_CCS'
    icd_10_dxccsr_paths = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/input_files/DXCCSR_v2021-2.csv'
    mimic_src_path = '/pvc/connor/mimic-iii-clinical-database-1.4/'
    # created with GEMS (General Equivalence Mapping)
    diagnosis_icd9_icd10_mapper_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/input_files/diagnosis_icd9_icd10.pcl'
    #LABELS OUTPUT PATH
    labels_output_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/{}_labels.pcl'
    train_data_output_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/{}'
    train_data_subsets_output_path = '/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/subsets/{}'
    
    
    if SELECTOR in ['PART_1', 'ALL']:
        #load mimic, filter , map icd9 to icd 10 and get relevant sections
        #brazilian_df = mimic_utils.mimic_map_icd9_icd10(diagnosis_icd9_icd10_mapper_path, mimic_src_path, mimic_labels_path=None)
        # Create an empty list to hold the data for the DataFrame
        data = []
        # Iterate through the CCS_PATH directory
        #df = pd.read_csv(SWE_PATH)
        df = pd.read_csv(SWE_PATH, error_bad_lines=False, warn_bad_lines=True)
        
        for index, row in df.iterrows():
            # Extract the first 4 characters of the filename as the patient ID
            patient_id = row['patientnr']
            # Load the CSV file into a pandas DataFrame
            #ccs_data = pd.read_csv(os.path.join(CCS_PATH, filename), sep=';')
            # Extract any non-empty strings from the 'CSSR' column and append to 'labels'

            #labels = [x for x in row['CSSR'].tolist() if isinstance(x, str) and x != ""]
            
            #
            #NEEDS TO BE FIXED TO DEAL WITH THE CASE WHERE THERE IS MORE THAN ONE CSSR CODE!
            #
            labels = []
            if 'nan' not in str(row['CSSR']):
                labels.append(row['CSSR'])
            # Make sure there's no duplicates
            #labels = list(set(labels))
            # Only add a row to the DataFrame if 'labels' is not empty
            if labels and row['full_note'] != "[UNK]":
                # Load the text data from the corresponding file in TEXT_PATH
                notes = row['full_note']
                # Create a dictionary representing a row of data and append to the list
                data.append({"patient_id": patient_id, "notes": notes, "labels": labels})
                #data.append({"patient_id": patient_id, "notes": notes, "labels": f'{labels}'})
        # Create the DataFrame from the list of data dictionaries
        df = pd.DataFrame(data, columns=["patient_id", "notes", "labels"])

        # # create a boolean mask for rows where the 'labels' column has the value 'hello'
        # mask = df['labels'] == "'[nan]'"
        # # use the mask to index into the DataFrame and select only the rows where the mask is False
        # df = df[~mask]

        # remove rows with '[nan]' in the 'labels' column
        df.to_csv('v2_swedish_tmp_0.csv', index=False)
      
    if SELECTOR in ['PART_2', 'ALL']:

        swedish_df = pd.read_csv('/pvc/cross-lingual-phenotype-prediction/dataset_creation/src/swedish/v2_swedish_tmp_0.csv')

        if task == 'codie_CCS':
            codie_labels = codie_utils.load_codie_labels(labels_output_path)
            #mimic_df_notes = mimic_utils.map_filter_ccs(brazilian_df, codie_labels, icd_10_dxccsr_paths)
            mimic_df_notes = swedish_df
            dataset_name = 'large_swedish_codiesp_filtered_CCS_'
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
        mimic_labels_array = train_test_split_experiment.swedish_label_to_tensor(mimic_df_notes, label_to_pos)

        #use stratified sampling to save train/dev/test
        train_test_split_experiment.stratified_sampling_multilearn(mimic_df_notes, 
                                                            mimic_labels_array, 
                                                            train_data_output_path.format(dataset_name))
        #train_test_split_experiment.load_mimic_paper_split(mimic_df_notes, train_data_output_path.format(dataset_name))

    if SELECTOR in ['PART_3', 'ALL']:

        smallest_set_size = 669
        swedish_df = pd.read_csv('/pvc/cross-lingual-phenotype-prediction/dataset_creation/src/swedish/swedish_tmp_0.csv')
        swedish_df = swedish_df.sample(n=smallest_set_size, random_state=42)

        if task == 'codie_CCS':
            codie_labels = codie_utils.load_codie_labels(labels_output_path)
            #mimic_df_notes = mimic_utils.map_filter_ccs(brazilian_df, codie_labels, icd_10_dxccsr_paths)
            mimic_df_notes = swedish_df
            dataset_name = 'small_swedish_codiesp_filtered_CCS_'
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
        mimic_labels_array = train_test_split_experiment.swedish_label_to_tensor(mimic_df_notes, label_to_pos)

        #use stratified sampling to save train/dev/test
        train_test_split_experiment.stratified_sampling_multilearn_sampling(mimic_df_notes, 
                                                            mimic_labels_array, 
                                                            train_data_subsets_output_path.format(dataset_name))
        #train_test_split_experiment.load_mimic_paper_split(mimic_df_notes, train_data_output_path.format(dataset_name))
