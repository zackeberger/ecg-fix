#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:26:21 2020

@author: Dani Kiyasseh
"""
import pickle
import os
import torch.nn as nn
import torch
import numpy as np
from itertools import combinations
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
#%%
""" Functions in this script:
    1) flatten_arrays
    2) obtain_contrastive_loss
    3) calculate_auc
    4) change_labels_type
    5) print_metrics
    6) save_metrics
    7) track_metrics
    8) save_config_weights
    9) save_patient_representations
    10) determine_classification_setting
    11) modify_dataset_order_for_multi_task_learning
    12) obtain_saved_weights_name
    13) make_dir
    14) make_saving_directory_contrastive
    15) obtain_information
    16) obtain_criterion
"""
#%%

def flatten_arrays(outputs_list,labels_list,modality_list,indices_list,task_names_list,pids_list):
    outputs_list = np.concatenate(outputs_list)
    labels_list = np.concatenate(labels_list)
    modality_list = np.concatenate(modality_list)
    indices_list = np.concatenate(indices_list)
    task_names_list = np.concatenate(task_names_list)
    pids_list = np.concatenate(pids_list)
    return outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list

import torch
import torch.nn.functional as F

# cache view-pairs by N+device (saves work every batch)
_PAIR_CACHE = {}

def _get_pairs(nviews: int, device: torch.device):
    key = (nviews, device.type, device.index)
    if key not in _PAIR_CACHE:
        idx = torch.arange(nviews, device=device)
        _PAIR_CACHE[key] = torch.combinations(idx, r=2)  # (K,2)
    return _PAIR_CACHE[key]

def obtain_contrastive_loss(latent_embeddings, pids, trial, temperature=0.1, pair_chunk=64):
    """
    Faster NCE-style loss.
    latent_embeddings: (B,H,N)
    pids: Tensor/list (B,)
    """
    device = latent_embeddings.device
    B, H, N = latent_embeddings.shape

    # normalize once (cosine similarity)
    z = F.normalize(latent_embeddings.float(), dim=1)  # (B,H,N)

    # pids -> torch on same device
    if not torch.is_tensor(pids):
        pids = torch.tensor(pids, device=device)
    else:
        pids = pids.to(device, non_blocking=True)

    # precompute same-patient off-diagonal indices ONCE per batch (GPU)
    rows1 = cols1 = rows2 = cols2 = None
    has_extra_pos = False
    if trial in ["CMSC", "CMLC", "CMSMLC"]:
        same = (pids[:, None] == pids[None, :])  # (B,B) bool
        # upper/lower off-diagonal indices
        triu_mask = torch.triu(same, diagonal=1)
        tril_mask = torch.tril(same, diagonal=-1)

        if triu_mask.any():
            rows1, cols1 = triu_mask.nonzero(as_tuple=True)
        if tril_mask.any():
            rows2, cols2 = tril_mask.nonzero(as_tuple=True)

        has_extra_pos = (rows1 is not None) or (rows2 is not None)

    # view-pairs (K,2)
    pairs = _get_pairs(N, device)
    K = pairs.shape[0]
    if K == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # reshape for fast batched bmm
    V = z.permute(2, 0, 1).contiguous()  # (N,B,H)

    targets = torch.arange(B, device=device)

    total_loss = 0.0
    total_pairs = 0

    # chunk view-pairs to control memory for large N (e.g. CMSMLC N=24 -> 276 pairs)
    for start in range(0, K, pair_chunk):
        end = min(start + pair_chunk, K)
        pair_chunk_idx = pairs[start:end]
        kc = pair_chunk_idx.shape[0]

        X = V[pair_chunk_idx[:, 0]]  # (kc,B,H)
        Y = V[pair_chunk_idx[:, 1]]  # (kc,B,H)

        # cross-view logits: (kc,B,B)
        logits = torch.bmm(X, Y.transpose(1, 2)) / temperature

        # -------------------------
        # CMC (fast CE version)
        # -------------------------
        if trial == "CMC":
            loss1 = F.cross_entropy(logits.reshape(-1, B), targets.repeat(kc))
            loss2 = F.cross_entropy(logits.transpose(1, 2).reshape(-1, B), targets.repeat(kc))
            loss_chunk = loss1 + loss2  # matches your two-direction sum
            total_loss += loss_chunk * kc
            total_pairs += kc
            continue

        # -------------------------
        # SimCLR (your denom logic, but logsumexp stable)
        # -------------------------
        if trial == "SimCLR":
            # self similarities (kc,B,B)
            self1 = torch.bmm(X, X.transpose(1, 2)) / temperature
            self2 = torch.bmm(Y, Y.transpose(1, 2)) / temperature

            # remove diagonal from self terms (off-diagonals only)
            inf = torch.tensor(-1e9, device=device, dtype=self1.dtype)
            eye = torch.eye(B, device=device, dtype=torch.bool).unsqueeze(0)  # (1,B,B)
            self1 = self1.masked_fill(eye, inf)
            self2 = self2.masked_fill(eye, inf)

            # numerator = diag of cross-view
            diag_logits = logits.diagonal(dim1=1, dim2=2)  # (kc,B)

            # denom1 = logsumexp( cross-view row + self1 row )
            denom1 = torch.logsumexp(torch.cat([logits, self1], dim=2), dim=2)  # (kc,B)
            denom2 = torch.logsumexp(torch.cat([logits.transpose(1, 2), self2], dim=2), dim=2)  # (kc,B)

            loss1 = -(diag_logits - denom1).mean()
            loss2 = -(diag_logits - denom2).mean()
            loss_chunk = loss1 + loss2

            total_loss += loss_chunk * kc
            total_pairs += kc
            continue

        # -------------------------
        # CMSC / CMLC / CMSMLC
        # -------------------------
        if trial in ["CMSC", "CMLC", "CMSMLC"]:
            # diag losses = same as cross-entropy in both directions, but cheaper via logsumexp
            diag_logits = logits.diagonal(dim1=1, dim2=2)      # (kc,B)
            row_lse = torch.logsumexp(logits, dim=2)           # (kc,B) sum over columns
            col_lse = torch.logsumexp(logits, dim=1)           # (kc,B) sum over rows

            loss_diag1 = -(diag_logits - row_lse).mean()
            loss_diag2 = -(diag_logits - col_lse).mean()

            loss_terms = 2
            loss_sum = loss_diag1 + loss_diag2

            # extra positives: same-patient off-diagonals (if present)
            if rows1 is not None and cols1 is not None:
                pos_logits = logits[:, rows1, cols1]           # (kc, P1)
                pos_den = row_lse[:, rows1]                    # (kc, P1)
                loss_triu = -(pos_logits - pos_den).mean()
                loss_sum = loss_sum + loss_triu
                loss_terms += 1

            if rows2 is not None and cols2 is not None:
                pos_logits = logits[:, rows2, cols2]           # (kc, P2)
                pos_den = col_lse[:, cols2]                    # (kc, P2)
                loss_tril = -(pos_logits - pos_den).mean()
                loss_sum = loss_sum + loss_tril
                loss_terms += 1

            loss_chunk = loss_sum / loss_terms

            total_loss += loss_chunk * kc
            total_pairs += kc
            continue

        # fallback
        raise ValueError(f"Unknown trial: {trial}")

    return total_loss / max(total_pairs, 1)


def calculate_auc(classification,outputs_list,labels_list,save_path_dir):
    ohe = LabelBinarizer()
    labels_ohe = ohe.fit_transform(labels_list)
    if classification is not None:
        if classification != '2-way':
            all_auc = []
            for i in range(labels_ohe.shape[1]):
                auc = roc_auc_score(labels_ohe[:,i],outputs_list[:,i])
                all_auc.append(auc)
            epoch_auroc = np.mean(all_auc)
        elif classification == '2-way':
            if 'physionet2020' in save_path_dir or 'ptbxl' in save_path_dir:
                """ Use This for MultiLabel Process -- Only for Physionet2020 """
                all_auc = []
                for i in range(labels_ohe.shape[1]):
                    auc = roc_auc_score(labels_ohe[:,i],outputs_list[:,i])
                    all_auc.append(auc)
                epoch_auroc = np.mean(all_auc)
            else:
                epoch_auroc = roc_auc_score(labels_list,outputs_list)
    else:
        print('This is not a classification problem!')
    return epoch_auroc

def calculate_acc(outputs_list,labels_list,save_path_dir):
    if 'physionet2020' in save_path_dir or 'ptbxl' in save_path_dir: #multilabel scenario
        """ Convert Preds to Multi-Hot Vector """
        preds_list = np.where(outputs_list>0.5,1,0)
        """ Indices of Hot Vectors of Predictions """
        preds_list = [np.where(multi_hot_vector)[0] for multi_hot_vector in preds_list]
        """ Indices of Hot Vectors of Ground Truth """
        labels_list = [np.where(multi_hot_vector)[0] for multi_hot_vector in labels_list]
        """ What Proportion of Labels Did you Get Right """
        acc = np.array([np.isin(preds,labels).sum() for preds,labels in zip(preds_list,labels_list)]).sum()/(len(np.concatenate(preds_list)))        
    else: #normal single label setting 
        preds_list = torch.argmax(torch.tensor(outputs_list),1)
        ncorrect_preds = (preds_list == torch.tensor(labels_list)).sum().item()
        acc = ncorrect_preds/preds_list.shape[0]
    return acc

def change_labels_type(labels,criterion):
    if isinstance(criterion,nn.BCEWithLogitsLoss):
        labels = labels.type(torch.float)
    elif isinstance(criterion,nn.CrossEntropyLoss):
        labels = labels.type(torch.long)
    return labels

def print_metrics(phase,results_dictionary):
    metric_name_to_label = {'epoch_loss':'loss','epoch_auroc':'auc','epoch_acc':'acc'}
    items_to_print = dict()
    labels = []
    for metric_name,result in results_dictionary.items():
        label = metric_name_to_label[metric_name]
        labels.append('-'.join((phase,label)))
        items_to_print[label] = ['%.4f' % result]
    print(tabulate(items_to_print,labels))

def save_metrics(save_path_dir,prefix,metrics_dict):
    torch.save(metrics_dict,os.path.join(save_path_dir,'%s_metrics_dict' % prefix))

def track_metrics(metrics_dict,results_dictionary,phase,epoch_count):
    for metric_name,results in results_dictionary.items():
        
        if epoch_count == 0 and ('train' in phase or 'test' in phase):
            metrics_dict[metric_name] = dict()
        
        if epoch_count == 0:
            metrics_dict[metric_name][phase] = []
        
        metrics_dict[metric_name][phase].append(results)
    return metrics_dict

def save_config_weights(save_path_dir,best_model_weights,saved_weights_name,phases,trial,downstream_dataset): #which is actually second_dataset
    if trial in ['Linear','Fine-Tuning','Random']:
        saved_weights_name = 'finetuned_weight'
    torch.save(best_model_weights,os.path.join(save_path_dir,saved_weights_name))

def save_patient_representation(save_path_dir,patient_rep_dict,trial):
    if trial not in ['Linear','Fine-Tuning']:
        with open(os.path.join(save_path_dir,'patient_rep'),'wb') as f:
            pickle.dump(patient_rep_dict,f)

def determine_classification_setting(dataset_name,trial):
    #dataset_name = dataset_name[0]
   # if dataset_name in ['mimiciv']:
  #      classification = None 
    if dataset_name == 'physionet':
        classification = '5-way'
    elif dataset_name == 'bidmc':
        classification = '2-way'
    elif dataset_name == 'mimic': #change this accordingly
        classification = '2-way'
    elif dataset_name == 'cipa':
        classification = '7-way'
    elif dataset_name == 'cardiology':
        classification = '12-way'
    elif dataset_name == 'physionet2017':
        classification = '4-way'
    elif dataset_name == 'tetanus':
        classification = '2-way'
    elif dataset_name == 'ptb':
        classification = '2-way'
    elif dataset_name == 'fetal':
        classification = '2-way'
    elif dataset_name == 'physionet2016':
        classification = '2-way'
    elif dataset_name == 'physionet2020':
        classification = '2-way' #because binary multilabel
    elif dataset_name == 'chapman':
        classification = '4-way'
    elif dataset_name == 'chapman_pvc':
        classification = '2-way'
    else: #used for pretraining with contrastive learning
        classification = None
    #print('Original Classification %s' % classification)
    return classification

def modify_dataset_order_for_multi_task_learning(dataset,modalities,leads,class_pairs,fractions):
    dataset = [dataset] #outside of if statement because dataset is original during each iteration
    if not isinstance(fractions,list): #it is already in list format, therefore no need for extra list
        modalities = [modalities]
        leads = [leads]
        class_pairs = [class_pairs]
        fractions = [fractions]
    return dataset,modalities,leads,class_pairs,fractions

def obtain_saved_weights_name(trial,phases):
    if trial not in ['Linear','Fine-Tuning','Random']:
        if 'train' in phases:
            saved_weights = 'pretrained_weight' #name of weights to save 
        elif 'val' in phases and len(phases) == 1 or 'test' in phases and len(phases) == 1:
            saved_weights = 'pretrained_weight' #name of weights to load
    elif trial in ['Linear','Fine-Tuning','Random']:
        if 'train' in phases:
            saved_weights = 'pretrained_weight' #name of weights to load
        elif 'val' in phases and len(phases) == 1 or 'test' in phases and len(phases) == 1:
            saved_weights = 'finetuned_weight' #name of weights to load
    return saved_weights

def obtain_load_path_dir(phases,save_path_dir,trial_to_run,second_dataset,labelled_fraction,leads,max_seed,task,evaluation=False):
    if trial_to_run in ['Linear','Fine-Tuning','Random']:
        labelled_fraction_path = 'training_fraction_%.2f' % labelled_fraction
        leads_path = 'leads_%s' % str(leads[0]) #remember leads is a list of lists 
        if trial_to_run in ['Random']:
            trial_to_run = ''
            if second_dataset in ['chapman','physionet2020']:
                leads_path = 'leads_%s' % str(leads[0]) #only these two datasets have multiple leads
            else:
                leads_path = ''

        if leads[0] == None:
            leads_path = ''

        save_path_dir = os.path.join(save_path_dir,trial_to_run,second_dataset,leads_path,labelled_fraction_path)        
        #print(save_path_dir)
        if 'train' in phases:
            save_path_dir, seed = make_dir(save_path_dir,max_seed,task,trial_to_run,second_pass=True,evaluation=evaluation) #do NOT change second_pass = True b/c this function is only ever used during second pass
        elif 'test' in phases:
            if 'test_metrics_dict' in os.listdir(save_path_dir):
                save_path_dir = 'do not test'
        
        if save_path_dir in ['do not train','do not test']:
            load_path_dir = save_path_dir
        else:
            split_save_path_dir = save_path_dir.split('/')
            seed_index = np.where(['seed' in token for token in split_save_path_dir])[0].item()
            load_path_dir = '/'.join(split_save_path_dir[:seed_index+1]) #you want to exclude everything AFTER the seed path
    else:
        load_path_dir = save_path_dir

    print(load_path_dir)
    print(save_path_dir)
    
    return load_path_dir, save_path_dir

def make_saving_directory_contrastive(task , downstream_dataset,trial_to_load,trial_to_run,seed,max_seed,downstream_task,embedding_dim, held_out_lr, temp):
    base_path = './SecondaryHDD/Contrastive Learning Results' 
    seed_path = 'seed%i' % int(seed)
    dataset_path = downstream_dataset#[0] #dataset used for training
    embedding_path = 'embedding_%i' % embedding_dim #size of embbedding used

    trial_path = trial_to_run
    temprature = f'temp{temp}'
    lr = f'lr{held_out_lr}'

    save_path_dir = os.path.join(base_path,trial_path,dataset_path,embedding_path,seed_path, temprature, lr)
    
    save_path_dir, seed = make_dir(save_path_dir,max_seed,task,trial_to_run,evaluation=False)
    
    return save_path_dir, seed

def make_dir(save_path_dir,max_seed,task,trial_to_run,second_pass=False,evaluation=False): #boolean allows you to overwrite if TRUE 
    """ Recursive Function to Make Sure I do Not Overwrite Previous Seeds """
    split_save_path_dir = save_path_dir.split('/')
    seed_index = np.where(['seed' in token for token in split_save_path_dir])[0].item()
    current_seed = int(split_save_path_dir[seed_index].strip('seed'))
    try:
        if second_pass == False:
            condition = ('obtain_representation' not in task) and (trial_to_run not in ['Linear','Fine-Tuning'])
        elif second_pass == True:
            condition = ('obtain_representation' not in task)
        
        if condition:# and trial_to_run not in ['Linear','Fine-Tuning']: #do not skip if you need to do finetuning
            os.chdir(save_path_dir)

            if 'train_val_metrics_dict' in os.listdir() and evaluation == False:
                if current_seed < max_seed-1:
                    print('Skipping Seed!')
                    new_seed = current_seed + 1
                    seed_path = 'seed%i' % new_seed
                    save_path_dir = save_path_dir.replace('seed%i' % current_seed,seed_path)
                    save_path_dir, seed = make_dir(save_path_dir,max_seed,task,trial_to_run,second_pass=second_pass,evaluation=evaluation)
                else:
                    save_path_dir = 'do not train'
    except:
        os.makedirs(save_path_dir)
    
    if os.path.isdir(save_path_dir) == False: #just in case we miss making the directory somewhere
        os.makedirs(save_path_dir)
    
    if current_seed == max_seed:
        current_seed = 0
    
    return save_path_dir, current_seed

def obtain_information(trial,downstream_dataset,second_dataset,data2leads_dict,data2bs_dict,data2lr_dict,data2classpair_dict):
    if trial in ['Linear','Fine-Tuning','Random']:
        training_dataset = second_dataset
    else:
        training_dataset = downstream_dataset #used for contrastive training 
    leads = data2leads_dict[training_dataset]
    batch_size = data2bs_dict[training_dataset]
    held_out_lr = data2lr_dict[training_dataset]
    class_pair = data2classpair_dict[training_dataset]
    modalities = ['ecg']
    fraction = 1 #1 for chapman, physio2020, and physio2017. Use labelled_fraction for control over fraction of training data used 
    return leads, batch_size, held_out_lr, class_pair, modalities, fraction       

def obtain_criterion(classification):
    if classification == '2-way':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion