# load data 
import argparse
import os 
import json
import glob 
import copy 
import pickle
import random 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
mse_loss = nn.MSELoss()
import utils
from utils import AverageMeterSet
import prepare_data
import models
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
kf = KFold(n_splits=5, random_state=None, shuffle=False)
from datetime import date
today = date.today()
date = today.strftime("%m%d")
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns 
matplotlib.rcParams["figure.dpi"] = 300
plt.style.use('bmh')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
legend_properties = {'weight':'bold', 'size': 14}
legend_properties_s = {'weight':'bold', 'size': 10}
dir_data = {'satori': '/nobackup/users/weiliao', 'colab':'/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910'}
dir_save = {'satori': '/home/weiliao/FR-TSVAE', 'colab': 'content/drive/My Drive/ColabNotebooks/MIMIC/TCN/VAE'}
# load data
meep_mimic = np.load(dir_data['satori'] + '/MIMIC_compile_0911_2022.npy', \
                allow_pickle=True).item()
train_vital = meep_mimic ['train_head']
dev_vital = meep_mimic ['dev_head']
test_vital = meep_mimic ['test_head']
mimic_static = np.load(dir_data['satori'] + '/MIMIC_static_0922_2022.npy', \
                        allow_pickle=True).item()
mimic_target = np.load(dir_data['satori'] + '/MIMIC_target_0922_2022.npy', \
                        allow_pickle=True).item()
 
class Args:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)

base_dir = '/home/weiliao/FR-TSVAE/checkpoints/'
# wn = '0512_lr1e-4beta.001_res_regrtheta_5_mlp_regr_nonsens_sens21_mask'
# p = base_dir + wn + '/' + 'stage1_sofa_fold_0_epoch33.pt'
wn = '0512_lr1e-4beta.001_res_regrtheta_5_mlp_regr_nonsens_sens1_mask'
p = base_dir + wn + '/' + 'stage1_sofa_fold_0_epoch39.pt'
# wn = '0512_lr1e-4beta.001_res_regrtheta_5_mlp_regr_nonsens_sens0_mask'
# p = base_dir + wn + '/' + 'stage1_sofa_fold_0_epoch37.pt'
mlp_params = [19, 200, 2]


with open(base_dir+wn+'/params.json') as f: 
    params = json.load(f)
params['platform'] = 'satori'
params['device_id'] = 0 
args = Args(params)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_head, train_static, train_sofa, train_id =  utils.crop_data_target('mimic', train_vital, mimic_target, mimic_static, 'train', args.sens_ind)
dev_head, dev_static, dev_sofa, dev_id =  utils.crop_data_target('mimic', dev_vital , mimic_target, mimic_static, 'dev',  args.sens_ind)
test_head, test_static, test_sofa, test_id =  utils.crop_data_target('mimic', test_vital, mimic_target, mimic_static, 'test',  args.sens_ind)

if args.use_sepsis3 == True:
    train_head, train_static, train_sofa, train_id = utils.filter_sepsis('mimic', train_head, train_static, train_sofa, train_id, args.platform)
    dev_head, dev_static, dev_sofa, dev_id = utils.filter_sepsis('mimic', dev_head, dev_static, dev_sofa, dev_id, args.platform)
    test_head, test_static, test_sofa, test_id = utils.filter_sepsis('mimic', test_head, test_static, test_sofa, test_id, args.platform)

# build model
model = models.Ffvae(args)
MLP_model = models.MLP(mlp_params, 20).to(device)
torch.save(MLP_model.state_dict(), '/home/weiliao/FR-TSVAE/start_weights.pt')
optimizer = torch.optim.Adam(MLP_model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

# 10-fold cross validation
trainval_head = train_head + dev_head
trainval_static = train_static + dev_static
trainval_stail = train_sofa + dev_sofa
trainval_ids = train_id + dev_id

# prepare data
torch.autograd.set_detect_anomaly(True)
for c_fold, (train_index, test_index) in enumerate(kf.split(trainval_head)):
    best_loss = 1e4
    patience = 0
    if c_fold >= 1:
        model.load_state_dict(torch.load('/home/weiliao/FR-TSVAE/start_weights.pt'))
    print('Starting Fold %d' % c_fold)
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    train_head, val_head = utils.slice_data(trainval_head, train_index), utils.slice_data(trainval_head, test_index)
    train_static, val_static = utils.slice_data(trainval_static, train_index), utils.slice_data(trainval_static, test_index)
    train_stail, val_stail = utils.slice_data(trainval_stail, train_index), utils.slice_data(trainval_stail, test_index)
    train_id, val_id = utils.slice_data(trainval_ids, train_index), utils.slice_data(trainval_ids, test_index)

    train_dataloader, dev_dataloader, test_dataloader = prepare_data.get_data_loader(args, train_head, val_head,
                                                                                        test_head, 
                                                                                        train_stail, val_stail,
                                                                                        test_sofa,
                                                                                        train_static=train_static,
                                                                                        dev_static=val_static,
                                                                                        test_static=test_static,
                                                                                        train_id=train_id,
                                                                                        dev_id=val_id,
                                                                                        test_id=test_id)
    model.load_state_dict(torch.load(p, map_location='cuda:0'))
    
    for j in range(args.epochs):
        # train 
        model.eval()
        MLP_model.train()
        train_loss = []
        for vitals, static, target, train_ids, key_mask in train_dataloader:
            vitals = vitals.to(device)
            static = static.to(device)
            target = target.to(device)
            key_mask = key_mask.to(device)

            with torch.no_grad():
                # (bs, zdim, T)
                latent_mu, _ = model.encoder(vitals)

            logits, probs = MLP_model(latent_mu[:, model.nonsens_idx, :].transpose(1, 2), mode='discriminator')
            logits_m = torch.stack([logits[i][key_mask[i]==0].mean(dim=0) for i in range(len(logits))])
            loss = loss_fn(logits_m, static.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())

        # eval 
        eval_loss = []
        MLP_model.eval()
        with torch.no_grad():
            for vitals, static, target, train_ids, key_mask in dev_dataloader:
                vitals = vitals.to(device)
                static = static.to(device)
                target = target.to(device)
                key_mask = key_mask.to(device)

                latent_mu, _ = model.encoder(vitals)
                logits, probs = MLP_model(latent_mu[:, model.nonsens_idx, :].transpose(1, 2), mode='discriminator')
                logits_m = torch.stack([logits[i][key_mask[i]==0].mean(dim=0) for i in range(len(logits))])
                loss = loss_fn(logits_m, static.long())
                eval_loss.append(loss.detach().cpu().numpy())
        print('Epoch %d, the train loss is %.4f, the test loss is %.4f'%(j, np.mean(train_loss), np.mean(eval_loss)))
        
        if np.mean(eval_loss) < best_loss: 
              patience = 0 
              best_loss = np.mean(eval_loss)
              best_model_state = copy.deepcopy(MLP_model.state_dict())
        else: 
              patience += 1 
              if patience >= args.patience:
                  print("Epoch %d :"%j, "Early stopped.")
                  torch.save(best_model_state, dir_save[args.platform] + '/checkpoints/' + wn + '/stage3_fold_%d_epoch%d_loss%.5f_1.pt'%(c_fold, j,  np.mean(eval_loss)))
                  break 
#     torch.save(best_model_state, dir_save[args.platform] + '/checkpoints/' + wn + '/stage3_fold_%d_epoch%d_loss%.5f.pt'%(c_fold, j,  np.mean(eval_loss)))