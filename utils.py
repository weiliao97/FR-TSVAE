import os
import json 
import numpy as np
import pandas as pd

class AverageMeterSet:
    """Computes average values of metrics"""
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update_dict(self, name_val_dict, n=1):
        for name, val in name_val_dict.items():
            self.update(name, val, n)

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=""):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix="/avg"):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix="/sum"):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix="/count"):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )

def creat_checkpoint_folder(target_path, target_file, data):
    """
    Create a folder to save the checkpoint
    input: target_path,
           target_file,
           data
    output: None
    """
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)

def crop_data_target(database, vital, target_dict, static_dict, mode):

    length = [i.shape[-1] for i in vital]
    all_train_id = list(target_dict[mode].keys())
    stayids = [all_train_id[i] for i, m in enumerate(length) if m > 24]
    sofa_tail = [target_dict[mode][j][24:] / 15 for j in stayids]
    sname = 'static_' + mode
    if database == 'mimic':
        train_filter = [vital[i][:, :-24] for i, m in enumerate(length) if m > 24]
        static_data = [static_dict[sname][static_dict[sname].index.get_level_values('stay_id') == j].values for j in
                   stayids]
    else:
        train_filter = [vital[i][:, :-24] for i, m in enumerate(length) if m >24]
        static_data = [static_dict[sname][static_dict[sname].index.get_level_values('patientunitstayid') == j].values for j in
                   stayids]
    # remove hospital mort flag and los
    # squeese from (1, 25) to (25, )
    static_data = [np.squeeze(np.concatenate((s[:, :2], s[:, 4:]), axis=1)) for s in static_data]
    return train_filter, static_data, sofa_tail, stayids

# def crop_data_target_eicu(vital, target_dict, mode):
#     length = [i.shape[-1] for i in vital]
    
#     all_train_id = list(target_dict[mode].keys())
#     stayids = [all_train_id[i] for i, m in enumerate(length) if m >24]
#     sofa_tail = [target_dict[mode][j][24:]/15 for j in stayids ]
#     return train_filter, sofa_tail, stayids

def filter_sepsis(database, vital, static, sofa, ids): 
    if database == 'mimic':
        id_df = pd.read_csv('/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/mimic_sepsis3.csv')
        sepsis3_id = id_df['stay_id'].values  # 1d array
    else:
        id_df = pd.read_csv('/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/eicu_sepsis3.csv')
        sepsis3_id = id_df['patientunitstayid'].values # 1d array 
    index_dict = dict((value, idx) for idx, value in enumerate(ids))
    ind = [index_dict[x] for x in sepsis3_id if x in index_dict.keys()]
    vital_sepsis = [vital[i] for i in ind]
    static_sepsis = [static[i] for i in ind]
    sofa_sepsis = [sofa[i] for i in ind]
    return vital_sepsis, static_sepsis, sofa_sepsis, [ids[i] for i in ind]

# def filter_sepsis_eicu(vital, sofa, ids):
    
#     index_dict = dict((value, idx) for idx,value in enumerate(ids))
#     ind = [index_dict[x] for x in sepsis3_id if x in index_dict.keys()]
#     vital_sepsis = [vital[i] for i in ind]
#     sofa_sepsis = [sofa[i] for i in ind]
#     return vital_sepsis, sofa_sepsis, [ids[i] for i in ind]

def slice_data(trainval_data, index):
    """
    Slice data based on index 
    """
    return [trainval_data[i] for i in index]