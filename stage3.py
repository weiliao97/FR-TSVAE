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

class Args:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for time series VAE stage3 training")
    parser.add_argument("--device_id", type=int, default=0, help="GPU id")
    parser.add_argument("--platform", type=str, default='satori', choices=['satori', 'colab'], help='Platform to run the code')
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=30, help="Patience epochs for early stopping.")
    # folder under checkpoints
    parser.add_argument("--wn", type=str, default='0512_lr1e-4beta.001_res_regrtheta_5_mlp_regr_nonsens_sens21_mask', help="folder name")
    # mlp channels
    parser.add_argument("--mlp_params", nargs='+', type=int, help="number of channels in the mlp model for sensitive prediction")
    stage3_args = parser.parse_args()

    device = torch.device("cuda:%d"%stage3_args.device_id if torch.cuda.is_available() else "cpu")
    # load data
    meep_mimic = np.load(dir_data[stage3_args.platform] + '/MIMIC_compile_0911_2022.npy', \
                    allow_pickle=True).item()
    train_vital = meep_mimic ['train_head']
    dev_vital = meep_mimic ['dev_head']
    test_vital = meep_mimic ['test_head']
    mimic_static = np.load(dir_data[stage3_args.platform] + '/MIMIC_static_0922_2022.npy', \
                            allow_pickle=True).item()
    mimic_target = np.load(dir_data[stage3_args.platform] + '/MIMIC_target_0922_2022.npy', \
                            allow_pickle=True).item()
    
    base_dir = dir_save[stage3_args.platform] + '/checkpoints/'
    # glob all the fold0 pt files under it
    p = base_dir + stage3_args.wn + '/'
    all_pt = glob.glob(base_dir + stage3_args.wn + '/*fold_0*.pt')
    # make folder /stage3
    if not os.path.exists(base_dir + stage3_args.wn + '/stage3/'):
        os.makedirs(base_dir + stage3_args.wn + '/stage3/')

    with open(base_dir+stage3_args.wn +'/params.json') as f: 
        params = json.load(f)
    params['platform'] = 'satori'
    params['device_id'] = 0 
    args = Args(params)
    

    train_head, train_static, train_sofa, train_id =  utils.crop_data_target('mimic', train_vital, mimic_target, mimic_static, 'train', args.sens_ind)
    dev_head, dev_static, dev_sofa, dev_id =  utils.crop_data_target('mimic', dev_vital , mimic_target, mimic_static, 'dev',  args.sens_ind)
    test_head, test_static, test_sofa, test_id =  utils.crop_data_target('mimic', test_vital, mimic_target, mimic_static, 'test',  args.sens_ind)

    if args.use_sepsis3 == True:
        train_head, train_static, train_sofa, train_id = utils.filter_sepsis('mimic', train_head, train_static, train_sofa, train_id, stage3_args.platform)
        dev_head, dev_static, dev_sofa, dev_id = utils.filter_sepsis('mimic', dev_head, dev_static, dev_sofa, dev_id, stage3_args.platform)
        test_head, test_static, test_sofa, test_id = utils.filter_sepsis('mimic', test_head, test_static, test_sofa, test_id, stage3_args.platform)

    # build model
    model = models.Ffvae(args)
    loss_fn = nn.CrossEntropyLoss()

    # 10-fold cross validation
    trainval_head = train_head + dev_head
    trainval_static = train_static + dev_static
    trainval_stail = train_sofa + dev_sofa
    trainval_ids = train_id + dev_id

    # prepare data
    torch.autograd.set_detect_anomaly(True)
    for p in all_pt: 
        # create sub folder for each pt file
        curr_dir = base_dir + stage3_args.wn + '/stage3/' + p.split('/')[-1].split('.')[0]
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
        MLP_model = models.MLP(stage3_args.mlp_params, args.zdim).to(device)
        # torch.save(MLP_model.state_dict(), '/home/weiliao/FR-TSVAE/start_weights.pt')
        optimizer = torch.optim.Adam(MLP_model.parameters(), lr = args.lr)
        for c_fold, (train_index, test_index) in enumerate(kf.split(trainval_head)):
            best_loss = 1e4
            best_w_loss = 1e4
            saved = False
            patience = 0
            # if c_fold >= 1:
            #     model.load_state_dict(torch.load('/home/weiliao/FR-TSVAE/start_weights.pt'))
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
            # calculate weights for imbalanced data
            weights_per_class = []
            for i in range(3):
                ctype, count= np.unique(np.asarray(val_static)[:, i], return_counts=True)
                total_dev_samples = len(val_static)
                curr = torch.FloatTensor([ total_dev_samples / k / len(ctype) for k in count]).to(device)
                weights_per_class.append(curr)
            
            model.load_state_dict(torch.load(p, map_location='cuda:%d'%stage3_args.device_id))
            print('load model from %s' % p)

            train_loss = pd.DataFrame(columns=['avg_loss', 'avg_w_loss', 'sens0_loss', 'sens1_loss', 'sens21_loss', 'sens0_w_loss', 'sens1_w_loss', 'sens21_w_loss'])
            dev_loss = pd.DataFrame(columns=['avg_loss', 'avg_w_loss', 'sens0_loss', 'sens1_loss', 'sens21_loss', 'sens0_w_loss', 'sens1_w_loss', 'sens21_w_loss'])
            for j in range(stage3_args.epochs):
                # train 
                model.eval()
                MLP_model.train()
                average_meters = AverageMeterSet()
    
                for vitals, static, target, train_ids, key_mask in train_dataloader:
                    vitals = vitals.to(device)
                    static = static.to(device)
                    target = target.to(device)
                    key_mask = key_mask.to(device)

                    with torch.no_grad():
                        # (bs, zdim, T)
                        latent_mu, _ = model.encoder(vitals)

                    # (bs, T, 2/3)
                    logits, probs = MLP_model(latent_mu[:, model.nonsens_idx, :].transpose(1, 2), mode='discriminator')
                    logits_m = torch.stack([logits[i][key_mask[i]==0].mean(dim=0) for i in range(len(logits))])

                    train_losses = [
                        nn.BCEWithLogitsLoss()(_z_logit.to(device), _a_sens.to(device))
                        for _z_logit, _a_sens in zip(
                        logits_m.t(), static.type(torch.FloatTensor).t())]

                    loss = torch.stack(train_losses).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # calcualte weighted clf 
                    train_w_losses = [
                        nn.BCEWithLogitsLoss(pos_weight = weights_per_class[k][1]/weights_per_class[k][0])(_z_logit.to(device), _a_sens.to(device))
                        for k, (_z_logit, _a_sens) in enumerate(zip(
                        logits_m.t(), static.type(torch.FloatTensor).t()))]
                    
                    # creat stats dict: 
                    stats = {}
                    stats['avg_loss'] = loss.item()
                    stats['avg_w_loss'] = torch.stack(train_w_losses).mean().item()
                    stats['sens0_loss'] = train_losses[0].item()
                    stats['sens1_loss'] = train_losses[1].item()
                    stats['sens21_loss'] = train_losses[2].item()
                    stats['sens0_w_loss'] = train_w_losses[0].item()
                    stats['sens1_w_loss'] = train_w_losses[1].item()
                    stats['sens21_w_loss'] = train_w_losses[2].item()
                    average_meters.update_dict(stats)
                
                train_loss.loc[len(train_loss)] = average_meters.averages().values()
                print("EPOCH: ", j, "TRAIN AVGs: ", average_meters.averages())

                # eval 
                MLP_model.eval()
                average_meters = AverageMeterSet()
                with torch.no_grad():
                    for vitals, static, target, train_ids, key_mask in dev_dataloader:
                        vitals = vitals.to(device)
                        static = static.to(device)
                        target = target.to(device)
                        key_mask = key_mask.to(device)

                        latent_mu, _ = model.encoder(vitals)
                        logits, probs = MLP_model(latent_mu[:, model.nonsens_idx, :].transpose(1, 2), mode='discriminator')
                        logits_m = torch.stack([logits[i][key_mask[i]==0].mean(dim=0) for i in range(len(logits))])


                        eval_losses = [
                            nn.BCEWithLogitsLoss()(_z_logit.to(device), _a_sens.to(device))
                            for _z_logit, _a_sens in zip(
                            logits_m.t(), static.type(torch.FloatTensor).t())]

                        eval_loss = torch.stack(train_losses).mean()
                        # calcualte weighted clf 
                        eval_w_losses = [
                            nn.BCEWithLogitsLoss(pos_weight = weights_per_class[k][1]/weights_per_class[k][0])(_z_logit.to(device), _a_sens.to(device))
                            for k, (_z_logit, _a_sens) in enumerate(zip(
                            logits_m.t(), static.type(torch.FloatTensor).t()))]
                        
                        # creat stats dict: 
                        stats = {}
                        stats['avg_loss'] = eval_loss.item()
                        stats['avg_w_loss'] = torch.stack(eval_w_losses).mean().item()
                        stats['sens0_loss'] = eval_losses[0].item()
                        stats['sens1_loss'] = eval_losses[1].item()
                        stats['sens21_loss'] = eval_losses[2].item()
                        stats['sens0_w_loss'] = eval_w_losses[0].item()
                        stats['sens1_w_loss'] = eval_w_losses[1].item()
                        stats['sens21_w_loss'] = eval_w_losses[2].item()
                        average_meters.update_dict(stats)
                    
                    dev_loss.loc[len(dev_loss)] = average_meters.averages().values()
                    print("EPOCH: ", j, "VAL AVGs: ", average_meters.averages())
                
                # val loss saving creteria, avg loss, avg w loss
                if average_meters.averages()['avg_loss/avg'] < best_loss: 
                    patience = 0 
                    best_loss = average_meters.averages()['avg_loss/avg']
                    best_model_state = copy.deepcopy(MLP_model.state_dict())
                else: 
                    patience += 1 
                    if patience >= stage3_args.patience:
                        print("Epoch %d :"%j, "Early stopped.")
                        torch.save(best_model_state, curr_dir + '/stage3_fold_%d_epoch%d_loss%.5f.pt'%(c_fold, j, best_loss))
                        torch.save(best_w_model_state, curr_dir + '/stage3_fold_%d_epoch%d_w_loss%.5f.pt'%(c_fold, j, best_w_loss))
                        # save pd df, show plot, save plot
                        plt.figure()
                        axs = train_loss.plot(figsize=(12, 14), subplots=True)
                        plt.savefig(curr_dir + '/train_loss_fold%d.eps'%c_fold, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
                        plt.figure()
                        axs = dev_loss.plot(figsize=(12, 14), subplots=True)
                        plt.savefig(curr_dir +  '/dev_loss_fold%d.eps'%c_fold, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
                        plt.show()
                        with open(os.path.join(curr_dir, 'train_loss_fold%d.pkl'%c_fold), 'wb') as f:
                            pickle.dump(train_loss, f)
                        with open(os.path.join(curr_dir, 'val_loss_fold%d.pkl'%c_fold), 'wb') as f:
                            pickle.dump(dev_loss, f)
                        saved = True
                        break 

                if average_meters.averages()['avg_w_loss/avg'] < best_w_loss: 
                    best_w_loss = average_meters.averages()['avg_w_loss/avg']
                    best_w_model_state = copy.deepcopy(MLP_model.state_dict())
            # only on fold 0
            if not saved: 
                torch.save(best_model_state, curr_dir + '/stage3_fold_%d_epoch%d_loss%.5f.pt'%(c_fold, j, best_loss))
                torch.save(best_w_model_state, curr_dir + '/stage3_fold_%d_epoch%d_w_loss%.5f.pt'%(c_fold, j, best_w_loss)) 
                # save pd df, show plot, save plot
                plt.figure()
                axs = train_loss.plot(figsize=(12, 14), subplots=True)
                plt.savefig(curr_dir + '/train_loss_fold%d.eps'%c_fold, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
                plt.figure()
                axs = dev_loss.plot(figsize=(12, 14), subplots=True)
                plt.savefig(curr_dir +  '/dev_loss_fold%d.eps'%c_fold, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
                plt.show()
                with open(os.path.join(curr_dir, 'train_loss_fold%d.pkl'%c_fold), 'wb') as f:
                    pickle.dump(train_loss, f)
                with open(os.path.join(curr_dir, 'val_loss_fold%d.pkl'%c_fold), 'wb') as f:
                    pickle.dump(dev_loss, f)
            break
        # start doing test on this pt file and save figures and results


    #     torch.save(best_model_state, dir_save[args.platform] + '/checkpoints/' + wn + '/stage3_fold_%d_epoch%d_loss%.5f.pt'%(c_fold, j,  np.mean(eval_loss)))