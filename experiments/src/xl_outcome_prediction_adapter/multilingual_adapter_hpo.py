from multilingual_adapter import *

from transformers import AutoConfig, AutoModelWithHeads
from transformers import TrainingArguments, Trainer
from datasets import concatenate_datasets
from transformers import AdapterType, AdapterConfig
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import roc_auc_score as auroc

import argparse
import json
import os

import pytorch_lightning as pl
import ray
import torch.utils.data
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from transformers import BertTokenizerFast
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from hyperopt import hp
import sys

from experiments.src.utils.codiespDataset import codiespDataset
sys.path.append('/pvc/')
from experiments.src.utils import utils
from experiments.src.utils.trainer_callback import EarlyStoppingCallback
import pickle


def tune_adapter(config,
                model_name, 
                task, 
                language,
                data_paths,
                is_first,
                dataset_name,
                task_adapter_path
                ):
                              
        language_codes = {'spanish':'es',
                        'english':'en', 
                        'greek':'el',
                        'portuguese':'mlm',
                        'swedish':'mlm'
                        }

        utils.set_seeds(seed=config['seed'])
        ########################### SETUP ADAPTER ########################### 
        #first_train_language = language
        train_dataset, dev_dataset, _, labels = utils.get_datav2(data_paths,
                                                                dataset_name=dataset_name)

        #task_adapter_name = f'codiesp_diagnosis_v4'
        #task_adapter_name = f'brazilian_codiesp_filtered'
        #task_adapter_name = f'swedish_codiesp_filtered'
        task_adapter_name = f'mimic_codiesp_filtered'
        #task_adapter_name = f'achepa_codiesp_filtered'


        codieSP_adapter = AdapterSetup(task_adapter_path=task_adapter_path,
                                        num_labels=len(labels),
                                        languages=languages, 
                                        task_adapter_name=task_adapter_name, 
                                        is_first=is_first,
                                        model_name=model_name,
                                        config=config)

        ########################### TRAINING FIRST LANGUAGE ########################### 
        adapter_trainer = AdapterTrainer(task_adapter_name, 
                                         model=codieSP_adapter.final_adapter_model)

        adapter_trainer.train_adapter(is_first=is_first,
                                lm_model_code=language_codes[language], 
                                train_dataset=train_dataset,
                                eval_dataset= dev_dataset,
                                config=config,
                                dataset_name=dataset_name, 
                                )
        

if __name__ == "__main__":
        
        # start ray on cluster or debug locally
        cluster = False

        if not cluster:
                print('starting ray cluster locally for debugging')
                ray.init(local_mode=True)
                
        else:
                print('starting ray  with the ray service on the CLUSTER')
                ray.init(address=os.environ["RAY_HEAD_SERVICE_HOST"] + ":6379")
        
        # Is it the first training of the task adapter
        is_first = False

        # model name SLA(single language)
        #mname = 'SLA'
        #mname = 'SLA_large'
        mname = 'MLA' 

        # base model where adapters are intergrated
        model_name = 'xlm-roberta-base'
        #model_name = 'bert-base-multilingual-cased'

        # column name of the text with English Translation
        ''' 'None' if you want the original language elsewise you have to use the name of the column that contains
        the translation '''
        translator_data_selector = None #'Opus_es_en_concat_notes'
        
        # filename to load labels
        filter_set_name = 'ccs_codie'

        # name of the dataset to train with
        #eval_dataset = 'brazilian'
        #eval_dataset = 'codie'
        eval_dataset = 'swedish'
        #eval_dataset = 'achepa'
        #eval_dataset = 'mimic'
        
        '''
         if it is not the first run include the other 
         language adapters name in the list  
         e.g. if it is pretrained with mimic and 
         second training is with CodiEsp 
         languages = ['english', 'spanish']
        '''
        #languages = ['english', 'spanish', 'portuguese', 'swedish', 'greek']
        #languages = ['greek', 'english']
        #languages = ['spanish', 'portuguese', 'greek']
        #languages = ['swedish']
        languages = ['english', 'portuguese', 'spanish', 'swedish']
        
        #language of the current dataset to continue training and evaluation
        #language = 'swedish'
        #language = 'portuguese'
        language = 'swedish'
        
        # just a variable for the naming of the experiments
        mla_order = '_'.join(languages)
        
        # only diagnosis task is implemented
        task = 'diagnosis' 
        
        # is it a test run USED for naming
        test = True

        # resources to execute the hpo
        resources_per_trial = {'cpu': 8, "gpu":1}

        # paths to datasets labels and columnname for translation or not (translation or original)
        data_paths = {'train_data_path_mimic': f"/pvc/data/paper_data/mimic_codiesp_filtered_CCS_fold_1_train.csv",
                'validation_data_path_mimic': f"/pvc/data/paper_data/mimic_codiesp_filtered_CCS_fold_1_dev.csv",
                'test_data_path_mimic': f"/pvc/data/paper_data/mimic_codiesp_filtered_CCS_fold_1_test.csv",
                
                'train_data_path_achepa': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/achepa_codiesp_filtered_CCS_fold_1_train.csv",
                'validation_data_path_achepa': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/achepa_codiesp_filtered_CCS_fold_1_dev.csv",
                'test_data_path_achepa': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/achepa_codiesp_filtered_CCS_fold_1_test.csv",
                # Note that these contain the translations as well
                # 'train_data_path_codie': f"/pvc/data/paper_data/codiesp_CCS_fold_1_train.csv",
                # 'validation_data_path_codie': f"/pvc/data/paper_data/codiesp_CCS_fold_1_dev.csv",
                # 'test_data_path_codie': f"/pvc/data/paper_data/codiesp_CCS_fold_1_test.csv",

                'train_data_path_brazilian': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/v3_brazilian_codiesp_filtered_CCS__fold_1_train.csv",
                'validation_data_path_brazilian': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/v3_brazilian_codiesp_filtered_CCS__fold_1_dev.csv",
                'test_data_path_brazilian': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/v3_brazilian_codiesp_filtered_CCS__fold_1_test.csv",

                'train_data_path_codie': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/codiesp_CCS_train.csv",
                'validation_data_path_codie': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/codiesp_CCS_dev.csv",
                'test_data_path_codie': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/codiesp_CCS_test.csv",
                
                # 'train_data_path_swedish': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/v1_swedish_codiesp_filtered_CCS__fold_1_train.csv",
                # 'validation_data_path_swedish': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/v1_swedish_codiesp_filtered_CCS__fold_1_dev.csv",
                # 'test_data_path_swedish': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/v1_swedish_codiesp_filtered_CCS__fold_1_test.csv",
                
                # The following splits contain about 15k clinical entries
                'train_data_path_swedish': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/large_swedish_codiesp_filtered_CCS__fold_1_train.csv",
                'validation_data_path_swedish': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/large_swedish_codiesp_filtered_CCS__fold_1_dev.csv",
                'test_data_path_swedish': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/large_swedish_codiesp_filtered_CCS__fold_1_test.csv",

                'all_labels_path': f"/pvc/cross-lingual-phenotype-prediction/dataset_creation/output_files/{filter_set_name}_labels.pcl",
                'eval_dataset': eval_dataset,
                'translator_data_selector': translator_data_selector,
                }

        # Paths to best models to continue training
        #task_adapter_mimic_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_mimic_original_SLA/_inner_2c36a0d2_50_first_acc_steps=4,first_attention_dropout=0.1,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.00042751,f_2021-10-20_00-03-23/training_output_en_0.0004275118309968961_0/checkpoint-13914/'
        task_adapter_achepa_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_achepa_original_SLA/_inner_4cc57928_35_first_acc_steps=2,first_attention_dropout=0.1,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.0052487,fi_2021-10-20_14-09-26/training_output_el_0.005248721818032698_0/checkpoint-198'
        task_adapter_mimic_achepa_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_greek_MLA/_inner_b566d67e_34_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-10-20_17-58-50/training_output_el_0_0.0005403420575244382/checkpoint-3781'
        #task_adapter_mimic_codie_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_MLA/_inner_b697bfc6_12_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-10-20_10-52-50/training_output_es_0_0.0011030338137158105/checkpoint-160'
        task_adapter_codie_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_codie_original_SLA/_inner_b34ab760_27_first_acc_steps=2,first_attention_dropout=0.3,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.0076105,fi_2021-10-19_14-34-52/training_output_es_0.007610478516231566_0/checkpoint-205'
        task_adapter_achepa_mimic_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_greek_english_diagnosis_MLA/_inner_3a0ea5a2_41_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-11-16_03-20-21/training_output_en_0_0.0017506470138346506/checkpoint-6176'
        #task_adapter_codie_mimic_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_spanish_english_diagnosis_MLA/_inner_7ceb71c6_34_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2021-11-15_14-43-55/training_output_en_0_0.0008006564657455058/checkpoint-6957'
        
        # My best paths
        task_adapter_mimic_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_mimic_original_SLA_TEST/_inner_fb6dbdf2_1_first_acc_steps=2,first_attention_dropout=0.1,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.0001,first__2023-05-08_12-22-21/training_output_en_0.0001_0/checkpoint-55764'
        task_adapter_codie_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_codie_original_SLA_TEST/_inner_c0c450ee_1_first_acc_steps=2,first_attention_dropout=0.1,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.0001,first__2023-04-24_13-05-57/training_output_es_0.0001_0/checkpoint-1804'
        task_adapter_brazilian_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_portuguese_diagnosis_MLA/_inner_5aa007a8_31_first_acc_steps=2,first_attention_dropout=0.1,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.0089499,fi_2023-04-19_15-44-16/training_output_mlm_0.008949922928507265_0/checkpoint-252'
        task_adapter_swedish_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_swedish_original_SLA_TEST/_inner_fdb01c1e_7_first_acc_steps=8,first_attention_dropout=0.3,first_batch_size=8,first_hidden_dropout=0.3,first_lr=0.0092856,fir_2023-05-03_13-33-26/training_output_mlm_0.009285612214373691_0/checkpoint-9'
        task_adapter_achepa_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_achepa_original_SLA_TEST/_inner_140f543e_14_first_acc_steps=1,first_attention_dropout=0.8,first_batch_size=8,first_hidden_dropout=0.3,first_lr=0.0044789,fi_2023-05-04_14-49-41/training_output_el_0.004478876035354823_0/checkpoint-400'
        task_adapter_swedish_large_sla_path = '/pvc/raytune_ccs_codie/tune_adapter_swedish_large_SLA_large_TEST/_inner_3cebff86_1_first_acc_steps=2,first_attention_dropout=0.1,first_batch_size=8,first_hidden_dropout=0.1,first_lr=0.0001,first__2023-05-09_12-58-47/training_output_mlm_0.0001_0/checkpoint-18941'

        task_adapter_mimic_codie_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_diagnosis_MLA_TEST/_inner_fffa38c0_14_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-03-10_12-01-26/training_output_es_0_0.0072789367740584785/checkpoint-120'
        task_adapter_mimic_brazilian_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_portuguese_diagnosis_MLA_TEST/_inner_021d9a0c_1_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoch_2023-04-24_12-53-28/training_output_mlm_0_1e-05/checkpoint-2632'
        #task_adapter_mimic_swedish_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_diagnosis_MLA_TEST/_inner_fffa38c0_14_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-03-10_12-01-26/training_output_es_0_0.0072789367740584785/checkpoint-120'
        task_adapter_codie_mimic_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_spanish_english_diagnosis_MLA_TEST/_inner_92332dc0_31_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-05-09_22-44-29/training_output_en_0_0.0011514621057752892/checkpoint-5211'
        task_adapter_swedish_large_codie_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_swedish_spanish_diagnosis_MLA_TEST/_inner_ca1dec28_44_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-05-10_14-57-14/training_output_es_0_0.0029066656543682897/checkpoint-660'
        task_adapter_swedish_large_codie_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_swedish_spanish_diagnosis_MLA_TEST/_inner_ca1dec28_44_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-05-10_14-57-14/training_output_es_0_0.0029066656543682897/checkpoint-660'

        task_adapter_mimic_codie_swedish_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_diagnosis_MLA_TEST/_inner_fffa38c0_14_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-03-10_12-01-26/training_output_es_0_0.0072789367740584785/checkpoint-120'
        task_adapter_mimic_codie_brazilian_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_portuguese_diagnosis_MLA_TEST/_inner_a954ed3a_1_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoch_2023-04-24_14-02-34/training_output_es_0_1e-05/checkpoint-1568'
        task_adapter_mimic_brazilian_codie_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_portuguese_spanish_diagnosis_MLA_TEST/_inner_c47ebcb8_26_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-05-04_13-46-19/training_output_es_0_0.00023203736286166917/checkpoint-779'
        task_adapter_codie_swedish_mimic_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_spanish_swedish_english_diagnosis_MLA_TEST/_inner_4dcd2fee_24_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-05-10_14-02-54/training_output_en_0_0.0067896484912566245/checkpoint-5418'

        task_adapter_mimic_codie_brazilian_greek_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_portuguese_greek_diagnosis_MLA_TEST/_inner_618dfffe_42_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-05-04_19-18-43/training_output_el_0_0.0027644557823171927/checkpoint-475'
        task_adapter_mimic_codie_swedish_brazilian_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_swedish_portuguese_diagnosis_MLA_TEST/_inner_8e2972ae_1_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoch_2023-05-04_12-52-41/training_output_mlm_0_1e-05/checkpoint-2296'
        task_adapter_mimic_codie_brazilian_swedish_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_portuguese_swedish_diagnosis_MLA_TEST/_inner_3c36c130_14_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-05-04_13-33-36/training_output_mlm_0_0.006367242027700415/checkpoint-64'
        task_adapter_mimic_brazilian_codie_greek_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_portuguese_spanish_greek_diagnosis_MLA_TEST/_inner_8ccb64be_31_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-05-08_13-52-33/training_output_el_0_0.0003024868430395529/checkpoint-1800'

        task_adapter_mimic_codie_brazilian_swedish_greek_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_english_spanish_portuguese_swedish_greek_diagnosis_MLA_TEST/_inner_8bbc02e8_21_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-05-08_12-55-17/training_output_el_0_0.0003977064698792958/checkpoint-1900'

        task_adapter_codie_brazilian_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_spanish_portuguese_diagnosis_MLA_TEST/_inner_9e30983a_33_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-05-08_11-08-22/training_output_mlm_0_0.0006257019910912267/checkpoint-224'
        task_adapter_codie_swedish_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_spanish_swedish_diagnosis_MLA_TEST/_inner_ddc92ea4_44_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-05-08_12-14-50/training_output_mlm_0_0.008111830863575689/checkpoint-56'

        task_adapter_codie_brazilian_mimic_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_spanish_portuguese_english_diagnosis_MLA_TEST/_inner_18c3274c_30_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-05-09_00-29-53/training_output_en_0_0.000531212189855528/checkpoint-12771'
        
        task_adapter_swedish_mimic_mla_path = '/pvc/raytune_ccs_codie/tune_adapter_swedish_english_diagnosis_MLA_TEST/_inner_fa4c4c76_35_first_acc_steps=0,first_attention_dropout=0,first_batch_size=8,first_hidden_dropout=0,first_lr=0,first_num_epoc_2023-05-09_04-20-10/training_output_en_0_0.00011356730781095573/checkpoint-71254'


        if is_first:
                # first training
                task_adapter_path = None
        else:
                # select path of best model to continue training from 
                #task_adapter_path = task_adapter_mimic_sla_path
                #task_adapter_path = task_adapter_mimic_brazilian_mla_path
                task_adapter_path = task_adapter_mimic_brazilian_codie_mla_path


        '''
                Naming of the experiments
        '''
        if language == 'spanish':
                if translator_data_selector is not None:
                        dataset_name = f"codie_{translator_data_selector}_{task}"
                else:
                        if mname == 'SLA':
                                dataset_name = f"codie_original"
                        elif mname == 'MLA': 
                                dataset_name = f"{mla_order}_{task}"
                        experiment_name = f"{dataset_name}_{mname}"


        elif language == 'english':
                if translator_data_selector is not None:
                        dataset_name = f"mimic_{translator_data_selector}"
                else:
                        if mname == 'SLA':
                                dataset_name = f"mimic_original"
                        elif mname == 'MLA': 
                                dataset_name = f"{mla_order}_{task}"
                        experiment_name = f"{dataset_name}_{mname}"

        elif language == 'greek':
                if translator_data_selector is not None:
                        dataset_name = f"achepa_{translator_data_selector}"
                else:
                        if mname == 'SLA':
                                dataset_name = f"achepa_original"
                        elif mname == 'MLA': 
                                dataset_name = f"{mla_order}_{task}"
                        experiment_name = f"{dataset_name}_{mname}"

        elif language == 'portuguese':
                if translator_data_selector is not None:
                        dataset_name = f"brazilian_{translator_data_selector}"
                else:
                        if mname == 'SLA':
                                dataset_name = f"brazilian_original"
                        elif mname == 'MLA': 
                                dataset_name = f"{mla_order}_{task}"
                        experiment_name = f"{dataset_name}_{mname}"

        elif language == 'swedish':
                if translator_data_selector is not None:
                        dataset_name = f"swedish_{translator_data_selector}"
                else:
                        if mname == 'SLA':
                                dataset_name = f"swedish_original"
                        elif mname == 'SLA_large':
                                dataset_name = f"swedish_large"
                        elif mname == 'MLA': 
                                dataset_name = f"{mla_order}_{task}"
                        experiment_name = f"{dataset_name}_{mname}"

        else: 
                dataset_name = f"{translator_data_selector}"
                experiment_name = f"{dataset_name}_{mname}_{task}"

        experiment_name = f"{experiment_name}" 

        if test:

                experiment_name = experiment_name + "_TEST"           


        
        '''
                settings for hyperparameter tuning
        '''
        if is_first:
                config = {"first_lr": hp.uniform("first_lr", 1e-5, 1e-2),
                        "second_lr": 0,
                        'first_batch_size': 8,
                        'second_batch_size': 0,
                        'per_device_eval_batch_size': 8,
                        "first_acc_steps": hp.choice("first_acc_steps", [1, 2, 4, 8, 16, 32]),
                        "second_acc_steps": 0,
                        "first_warmup_steps": hp.choice("first_warmup_steps", [0, 10, 250, 500, 750]),
                        "second_warmup_steps": 0, 
                        'first_weight_decay': 0,
                        'second_weight_decay': 0,
                        'first_num_epochs':  100, 
                        'second_num_epochs': 1,
                        'seed': 42, 
                        'first_hidden_dropout': hp.choice('first_hidden_dropout', [0.1, 0.3, 0.5, 0.8]),
                        'first_attention_dropout': hp.choice('first_attention_dropout', [0.1, 0.3, 0.5, 0.8]),
                        'second_hidden_dropout': 0,
                        'second_attention_dropout': 0,
                        }
        else:
                config = {"first_lr": 0,
                        "second_lr": hp.uniform("second_lr", 1e-5, 1e-2),
                        'first_batch_size': 8,
                        'second_batch_size': 8,
                        'per_device_eval_batch_size': 8,
                        "first_acc_steps": 0,
                        "second_acc_steps": hp.choice("second_acc_steps", [1, 2, 4, 8, 16]),
                        "first_warmup_steps": 0,
                        "second_warmup_steps": hp.choice("second_warmup_steps", [0, 10, 250, 500, 750]),
                        'first_weight_decay': 0,
                        'second_weight_decay': 0,
                        'first_num_epochs':  100,
                        'second_num_epochs': 100,
                        'seed': 42,
                        'second_hidden_dropout': hp.choice('second_hidden_dropout', [0.1, 0.3, 0.5, 0.8]),
                        'second_attention_dropout': hp.choice('second_attention_dropout', [0.1, 0.3, 0.5, 0.8]),
                        'first_hidden_dropout': 0,
                        'first_attention_dropout': 0,
                        }

        defaults = [{"first_lr": 1e-4, 
                "second_lr": 1e-5, 
                'first_batch_size': 8,
                'second_batch_size': 8,
                'per_device_eval_batch_size': 8,
                "first_acc_steps": 2,
                "second_acc_steps": 2,
                "first_warmup_steps":0,
                "second_warmup_steps":0,
                'first_weight_decay': 0,
                'second_weight_decay': 0,
                'first_num_epochs':  100,
                'seed': 42,
                'second_num_epochs': 100,
                'first_attention_dropout': 0.1,
                'first_hidden_dropout': 0.1,
                'second_attention_dropout': 0.1,
                'second_hidden_dropout': 0.1,
                }]

        utils.set_seeds(seed=config['seed'])


        search = HyperOptSearch(
                            config,
                            metric="eval_val_auc",
                            mode="max",
                            points_to_evaluate=defaults,
                            n_initial_points=30)


        scheduler = AsyncHyperBandScheduler(
                                                brackets=1,
                                                grace_period=2,
                                                reduction_factor=4,
                                                max_t=100
                                        )


        reporter = CLIReporter(
                        parameter_columns=["first_lr",
                                        "first_batch_size",
                                        "first_acc_steps", 
                                        "first_warmup_steps",
                                        "second_lr",
                                        "second_batch_size",
                                        "second_acc_steps", 
                                        "second_warmup_steps"
                                        ],

                        metric_columns=["eval_val_pr_auc",
                                        ]
                            )


        analysis = tune.run(tune.with_parameters(tune_adapter,
                                                model_name=model_name,
                                                task=task, 
                                                language=language,
                                                data_paths=data_paths,
                                                is_first=is_first,
                                                dataset_name=eval_dataset,
                                                task_adapter_path=task_adapter_path
                                                ),    
                        local_dir= f"/pvc/raytune_{filter_set_name}/",
                        resources_per_trial=resources_per_trial,
                        metric="eval_val_auc",
                        mode="max",
                        config=config,
                        num_samples=50,
                        scheduler=scheduler,
                        search_alg=search,
                        progress_reporter=reporter,
                        name=f"tune_adapter_{experiment_name}", 
                        checkpoint_at_end=True,)

        best_config = analysis.get_best_config()
        
        # with open(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}_best_config_{filter_set_name}.pcl",'wb') as f: 
        #         pickle.dump(best_config, f)

        # analysis.best_result_df.to_csv(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}_best_result_{filter_set_name}.csv", index=False)
        # analysis.dataframe().to_csv(f"/pvc/tasks/codie_ccs_based_data/{experiment_name}_best_result_{filter_set_name}.csv", index=False)
        with open(f"/pvc/tasks/brazilian_based_data/{experiment_name}_best_config_{filter_set_name}.pcl",'wb') as f: 
                pickle.dump(best_config, f)

        analysis.best_result_df.to_csv(f"/pvc/tasks/brazilian_based_data/{experiment_name}_best_result_{filter_set_name}.csv", index=False)
        analysis.dataframe().to_csv(f"/pvc/tasks/brazilian_based_data/{experiment_name}_best_result_{filter_set_name}.csv", index=False)
                

        