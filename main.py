import argparse
import os 
import copy 
import pickle
import timeit 
import numpy as np
import pandas as pd
import torch
import utils
from utils import AverageMeterSet
import prepare_data
import models
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=None, shuffle=False)
from datetime import date
today = date.today()
date = today.strftime("%m%d")
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rcParams["figure.dpi"] = 300
plt.style.use('bmh')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
legend_properties = {'weight':'bold', 'size': 14}
dir_data = {'satori': '/nobackup/users/weiliao', 'colab':'/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910'}
dir_save = {'satori': '/home/weiliao/FR-TSVAE', 'colab': '/content/drive/My Drive/ColabNotebooks/MIMIC/TCN/VAE'}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for time series VAE models")
    parser.add_argument("--device_id", type=int, default=0, help="GPU id")
    parser.add_argument("--platform", type=str, default='colab', choices=['satori', 'colab'], help='Platform to run the code')
    parser.add_argument("--database", type=str, default='mimic', choices=['mimic', 'eicu'], help='Database')
    parser.add_argument("--retrain", action = 'store_true', default= False, help="Whethe retrain")
    parser.add_argument("--retrain_pt", type=str, default = '0525_eicu_lr1e-4beta.001_res_regrtheta_5_mlp_regr_nonsens_sens0/stage1_clfw_fold_0_epoch29.pt')
    
    # data/loss parameters
    parser.add_argument("--use_sepsis3", action = 'store_false', default= True, help="Whethe only use sepsis3 subset")
    parser.add_argument("--bucket_size", type=int, default=300, help="bucket size to group different length of time-series data")
    parser.add_argument("--beta", type=float, default=0.0001, help="coefficent for the elbo loss")
    parser.add_argument("--gamma", type=float, default=0.5, help="coefficent for the total_corr loss")
    parser.add_argument("--alpha", type=float, default=0.5, help="coefficent for the clf loss")
    parser.add_argument("--theta", type=float, default=10, help="coefficent for the sofa loss in stage 1")
    parser.add_argument("--zdim", type=int, default=20, help="dimension of the latent space")
    parser.add_argument("--sens_ind", type=int, default=0, help="index of the sensitive feature")
    parser.add_argument("--scale_elbo", action = 'store_true', help="Whether to scale the ELBO loss")
    # model parameters
    parser.add_argument("--kernel_size", type=int, default=3, help="kernel size")
    parser.add_argument("--drop_out", type=float, default=0.2, help="drop out rate")
    parser.add_argument("--enc_channels",  nargs='+', type=int, help="number of channels in the encoder")
    parser.add_argument("--dec_channels",  nargs='+', type=int, help="number of channels in the decoder")
    parser.add_argument("--num_inputs", type=int, default=200, help="number of features in the inputs")
    # discriminator parameters
    parser.add_argument("--disc_channels",  type=int, default=200, help="number of channels in the discriminator")
    # regressor parameters
    parser.add_argument("--regr_model",  type=str, default='mlp', choices=['mlp', 'tcn'], help='Model choice in sofa prediction')
    parser.add_argument("--regr_channels",  type=int, default=200, help="number of channels in the regressor")
    parser.add_argument("--regr_tcn_channels",  nargs='+', type=int, help="number of channels in the regressor")
    parser.add_argument("--regr_only_nonsens", action = 'store_true', help="Whether only using nonsens latents to predict sofa")
    # training parameters
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--data_batching", type=str, default='same', choices=['same', 'close', 'random'], help='How to batch data')
    parser.add_argument("--bs", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Patience epochs for early stopping.")
    parser.add_argument("--checkpoint", type=str, default='test', help=" name of checkpoint model")
    # ihm 
    parser.add_argument("--thresh", type=int, default=48, help="how many hours of data to use")
    parser.add_argument("--gap", type=int, default=6, help="gap hours between record stop and data used in training")

    args = parser.parse_args()
    device = torch.device("cuda:%d"%args.device_id if torch.cuda.is_available() else "cpu")
    arg_dict = vars(args)
    workname = date + "_" +  args.checkpoint
    utils.creat_checkpoint_folder(dir_save[args.platform] + '/checkpoints/' + workname, 'params.json', arg_dict)

    # load data
    if args.database == 'mimic':
        meep_mimic = np.load(dir_data[args.platform] + '/MIMIC_compile_0911_2022.npy', \
                        allow_pickle=True).item()
        train_vital = meep_mimic ['train_head']
        dev_vital = meep_mimic ['dev_head']
        test_vital = meep_mimic ['test_head']
        mimic_static = np.load(dir_data[args.platform] + '/MIMIC_static_0922_2022.npy', \
                                allow_pickle=True).item()
        mimic_target = np.load(dir_data[args.platform] + '/MIMIC_target_0922_2022.npy', \
                                allow_pickle=True).item()
    else: 
        meep_mimic = np.load(dir_data[args.platform] + '/eICU_compile_0911_2022_2.npy', \
                        allow_pickle=True).item()
        train_vital = meep_mimic ['train_head']
        dev_vital = meep_mimic ['dev_head']
        test_vital = meep_mimic ['test_head']
        mimic_static = np.load(dir_data[args.platform] + '/eICU_static_0922_2022.npy', \
                                allow_pickle=True).item()
        mimic_target = np.load(dir_data[args.platform] + '/eICU_target_0922_2022.npy', \
                                allow_pickle=True).item()
        
    train_head, train_static, train_sofa, train_id =  utils.crop_data_target(args.database, train_vital, mimic_target, mimic_static, 'train', args.sens_ind, args.thresh, args.gap)
    dev_head, dev_static, dev_sofa, dev_id =  utils.crop_data_target(args.database, dev_vital , mimic_target, mimic_static, 'dev',  args.sens_ind,  args.thresh, args.gap)
    test_head, test_static, test_sofa, test_id =  utils.crop_data_target(args.database, test_vital, mimic_target, mimic_static, 'test',  args.sens_ind,  args.thresh, args.gap)

    if args.use_sepsis3 == True:
        train_head, train_static, train_sofa, train_id = utils.filter_sepsis(args.database, train_head, train_static, train_sofa, train_id, args.platform)
        dev_head, dev_static, dev_sofa, dev_id = utils.filter_sepsis(args.database, dev_head, dev_static, dev_sofa, dev_id, args.platform)
        test_head, test_static, test_sofa, test_id = utils.filter_sepsis(args.database, test_head, test_static, test_sofa, test_id, args.platform)



    # 10-fold cross validation
    trainval_head = train_head + dev_head
    trainval_static = train_static + dev_static
    trainval_stail = train_sofa + dev_sofa
    trainval_ids = train_id + dev_id

    # prepare data
    torch.autograd.set_detect_anomaly(True)
    for c_fold, (train_index, test_index) in enumerate(kf.split(trainval_head)):
        # best_loss = 1e4
        # patience = 0

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
        ctype, count= np.unique(np.asarray(val_static), return_counts=True)
        total_dev_samples = len(val_static)
        weights_per_class = torch.FloatTensor([ total_dev_samples / k / len(ctype) for k in count]).to(device)
        
        ctype, count= np.unique(np.asarray(val_stail), return_counts=True)
        total_dev_samples = len(val_stail)
        weights_ihm = torch.FloatTensor([ total_dev_samples / k / len(ctype) for k in count]).to(device)
        # build model
        model = models.Ffvae(args, weights_per_class, weights_ihm)
        if c_fold == 0:
            if args.retrain: 
                model.load_state_dict(torch.load(dir_save[args.platform] + '/checkpoints/' + args.retrain_pt, map_location='cuda:%d'%args.device_id))
            else:
                torch.save(model.state_dict(), dir_save[args.platform] + '/start_weights_%d.pt'%args.device_id)
        else:
            model.load_state_dict(torch.load(dir_save[args.platform] + '/start_weights_%d.pt'%args.device_id))
            
        # df to record loss
        train_loss = pd.DataFrame(columns=['ffvae_cost', 'recon_cost', 'kl_cost', 'corr_term', 'clf_term', 'clf_w_term', 'sofa_term', 'sofaw_term', 'disc_cost', 'sofap_loss'])
        dev_loss = pd.DataFrame(columns=['ffvae_cost', 'recon_cost', 'kl_cost', 'corr_term', 'clf_term', 'clf_w_term', 'sofa_term', 'sofaw_term', 'disc_cost', 'sofap_loss'])
        # test_loss = pd.DataFrame(columns=['ffvae_cost', 'recon_cost', 'kl_cost', 'corr_term', 'clf_term', 'disc_cost', 'sofap_loss'])
        best_loss = 1e5
        best_clf_loss = 1e4 
        best_clf_w_loss = 1e4 
        best_sofa_loss = 1e4
        best_sofaw_loss = 1e4
        best_corr_loss = 1e4
        patience = 0
        start_time = timeit.default_timer()
        saved = False
        for j in range(args.epochs):
            model.train()
            average_meters = AverageMeterSet()

            for vitals, static, target, train_ids, key_mask in train_dataloader:
                vitals = vitals.to(device)
                static = static.to(device)
                target = target.to(device)
                key_mask = key_mask.to(device)

                _, cost_dict = model(vitals, key_mask, target, static, "ffvae_train")

                stats = dict((n, c.item()) for (n, c) in cost_dict.items())
                average_meters.update_dict(stats)
                
            # print and record loss 
            train_loss.loc[len(train_loss)] = average_meters.averages().values()
            print("EPOCH: ", j, "TRAIN AVGs: ", average_meters.averages())

            model.eval()
            average_meters = AverageMeterSet()
            with torch.no_grad():
                for vitals, static, target, train_ids, key_mask in dev_dataloader:
                    vitals = vitals.to(device)
                    static = static.to(device)
                    target = target.to(device)
                    key_mask = key_mask.to(device)

                    _, cost_dict = model(vitals, key_mask, target, static, "test")

                    stats = dict((n, c.item()) for (n, c) in cost_dict.items())
                    average_meters.update_dict(stats)
                
            # print and record loss 
            dev_loss.loc[len(dev_loss)] = average_meters.averages().values()
            print("EPOCH: ", j, "VAL AVGs: ", average_meters.averages())

            if average_meters.averages()['ffvae_cost/avg'] < best_loss:
                patience = 0 
                best_loss = average_meters.averages()['ffvae_cost/avg']
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1 
                if patience >= args.patience:
                    print("Epoch %d :"%j, "Early stopped.")
                    # torch.save(best_model_state, '/home/weiliao/FR-TSVAE/checkpoints/' + workname + '/stage1_epoch%d.pt'%j)
                    break 
            if average_meters.averages()['clf_term/avg'] < best_clf_loss: 
                best_clf_loss = average_meters.averages()['clf_term/avg']
                best_clf_model = copy.deepcopy(model.state_dict())
                
            if average_meters.averages()['clf_w_term/avg'] < best_clf_w_loss: 
                best_clf_w_loss = average_meters.averages()['clf_w_term/avg']
                best_clf_w_model = copy.deepcopy(model.state_dict())
                
            if average_meters.averages()['sofa_term/avg'] < best_sofa_loss: 
                best_sofa_loss = average_meters.averages()['sofa_term/avg']
                best_sofa_model = copy.deepcopy(model.state_dict())

            if average_meters.averages()['sofaw_term/avg'] < best_sofaw_loss: 
                best_sofaw_loss = average_meters.averages()['sofaw_term/avg']
                best_sofaw_model = copy.deepcopy(model.state_dict())
                
            if abs(average_meters.averages()['corr_term/avg']) < best_corr_loss: 
                best_corr_loss = abs(average_meters.averages()['corr_term/avg'])
                best_corr_model =  copy.deepcopy(model.state_dict())
            
            # save if code cannot be completely ran 
            elapsed = timeit.default_timer() - start_time
            # threshold 11:30 
            if elapsed > 82800 and (not saved):
                torch.save(best_model_state, dir_save[args.platform] + '/checkpoints/' + workname + '/stage1_fold_%d_epoch%d.pt'%(c_fold, j))
                torch.save(best_clf_model, dir_save[args.platform] + '/checkpoints/' + workname + '/stage1_clf_fold_%d_epoch%d.pt'%(c_fold, j))
                torch.save(best_clf_w_model, dir_save[args.platform] + '/checkpoints/' + workname + '/stage1_clfw_fold_%d_epoch%d.pt'%(c_fold, j))
                torch.save(best_sofa_model, dir_save[args.platform] + '/checkpoints/' + workname + '/stage1_sofa_fold_%d_epoch%d.pt'%(c_fold, j))
                torch.save(best_sofaw_model, dir_save[args.platform] + '/checkpoints/' + workname + '/stage1_sofaw_fold_%d_epoch%d.pt'%(c_fold, j))
                torch.save(best_corr_model, dir_save[args.platform] + '/checkpoints/' + workname + '/stage1_corr_fold_%d_epoch%d.pt'%(c_fold, j))
                # save pd df, show plot, save plot
                plt.figure()
                axs = train_loss.plot(figsize=(12, 14), subplots=True)
                plt.savefig(dir_save[args.platform] + '/checkpoints/' + workname + '/train_loss_fold%d.eps'%c_fold, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
                plt.figure()
                axs = dev_loss.plot(figsize=(12, 14), subplots=True)
                plt.savefig(dir_save[args.platform] + '/checkpoints/' + workname + '/dev_loss_fold%d.eps'%c_fold, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
                plt.show()
                with open(os.path.join(dir_save[args.platform] + '/checkpoints/' + workname, 'train_loss_fold%d.pkl'%c_fold), 'wb') as f:
                    pickle.dump(train_loss, f)
                with open(os.path.join(dir_save[args.platform] + '/checkpoints/' + workname, 'val_loss_fold%d.pkl'%c_fold), 'wb') as f:
                    pickle.dump(dev_loss, f)
                saved = True
                
        torch.save(best_model_state, dir_save[args.platform] + '/checkpoints/' + workname + '/stage1_fold_%d_epoch%d.pt'%(c_fold, j))
        torch.save(best_clf_model, dir_save[args.platform] + '/checkpoints/' + workname + '/stage1_clf_fold_%d_epoch%d.pt'%(c_fold, j))
        torch.save(best_clf_w_model, dir_save[args.platform] + '/checkpoints/' + workname + '/stage1_clfw_fold_%d_epoch%d.pt'%(c_fold, j))
        torch.save(best_sofa_model, dir_save[args.platform] + '/checkpoints/' + workname + '/stage1_sofa_fold_%d_epoch%d.pt'%(c_fold, j))
        torch.save(best_sofaw_model, dir_save[args.platform] + '/checkpoints/' + workname + '/stage1_sofaw_fold_%d_epoch%d.pt'%(c_fold, j))
        torch.save(best_corr_model, dir_save[args.platform] + '/checkpoints/' + workname + '/stage1_corr_fold_%d_epoch%d.pt'%(c_fold, j))
        saved = True

        # save pd df, show plot, save plot
        plt.figure()
        axs = train_loss.plot(figsize=(12, 14), subplots=True)
        plt.savefig(dir_save[args.platform] + '/checkpoints/' + workname + '/train_loss_fold%d.eps'%c_fold, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
        plt.figure()
        axs = dev_loss.plot(figsize=(12, 14), subplots=True)
        plt.savefig(dir_save[args.platform] + '/checkpoints/' + workname + '/dev_loss_fold%d.eps'%c_fold, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
        plt.show()
        with open(os.path.join(dir_save[args.platform] + '/checkpoints/' + workname, 'train_loss_fold%d.pkl'%c_fold), 'wb') as f:
            pickle.dump(train_loss, f)
        with open(os.path.join(dir_save[args.platform] + '/checkpoints/' + workname, 'val_loss_fold%d.pkl'%c_fold), 'wb') as f:
            pickle.dump(dev_loss, f)

        train_regr_loss = pd.DataFrame(columns=['ffvae_cost', 'recon_cost', 'kl_cost', 'corr_term', 'clf_term', 'clf_w_term', 'sofa_term', 'sofaw_term',  'disc_cost', 'sofap_loss'])
        dev_regr_loss = pd.DataFrame(columns=['ffvae_cost', 'recon_cost', 'kl_cost', 'corr_term', 'clf_term', 'clf_w_term', 'sofa_term', 'sofaw_term', 'disc_cost', 'sofap_loss'])
        best_loss = 1e4
        patience = 0
        # train from best clf model
        model.load_state_dict(torch.load(dir_save[args.platform] + '/checkpoints/%s/stage1_clf_fold_%d_epoch%d.pt'%(workname, c_fold, j)))
        
        # train the regression model
        for j in range(args.epochs): 

            model.train()
            average_meters = AverageMeterSet()

            for vitals, static, target, train_ids, key_mask in train_dataloader:
                vitals = vitals.to(device)
                static = static.to(device)
                target = target.to(device)
                key_mask = key_mask.to(device)

                sofap, cost_dict = model(vitals, key_mask, target, static, "train")

                stats = dict((n, c.item()) for (n, c) in cost_dict.items())
                average_meters.update_dict(stats)
                
            # print and record loss 
            train_regr_loss.loc[len(train_regr_loss)] = average_meters.averages().values()
            print("EPOCH: ", j, "TRAIN AVGs: ", average_meters.averages())

            model.eval()
            average_meters = AverageMeterSet()
            with torch.no_grad():
                for vitals, static, target, train_ids, key_mask in dev_dataloader:
                    vitals = vitals.to(device)
                    static = static.to(device)
                    target = target.to(device)
                    key_mask = key_mask.to(device)

                    _, cost_dict = model(vitals, key_mask, target, static, "test")

                    stats = dict((n, c.item()) for (n, c) in cost_dict.items())
                    average_meters.update_dict(stats)
                
            # print and record loss 
            dev_regr_loss.loc[len(dev_regr_loss)] = average_meters.averages().values()
            print("EPOCH: ", j, "VAL AVGs: ", average_meters.averages())

            if average_meters.averages()['main_cost/avg'] < best_loss:
                patience = 0 
                best_loss = average_meters.averages()['main_cost/avg']
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1 
                if patience >= args.patience:
                    print("Epoch %d :"%j, "Early stopped.")
                    torch.save(best_model_state, dir_save[args.platform] + '/checkpoints/' + workname + '/stage2_fold_%d_epoch%d.pt'%(c_fold, j))
                    break 

        # save pd df, show plot, save plot
        plt.figure()
        axs = train_regr_loss.plot(figsize=(12, 14), subplots=True)
        plt.savefig(dir_save[args.platform] + '/checkpoints/' + workname + '/train_regr_loss_fold%d.eps'%c_fold, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
        plt.figure()
        axs = dev_regr_loss.plot(figsize=(12, 14), subplots=True)
        plt.savefig(dir_save[args.platform] + '/checkpoints/' + workname + '/dev_regr_loss_fold%d.eps'%c_fold, format='eps', bbox_inches = 'tight', pad_inches = 0.1, dpi=1200)
        plt.show()
        with open(os.path.join(dir_save[args.platform] + '/checkpoints/' + workname, 'train_regr_loss_fold%d.pkl'%c_fold), 'wb') as f:
            pickle.dump(train_regr_loss, f)
        with open(os.path.join(dir_save[args.platform] + '/checkpoints/' + workname, 'val_regr_loss_fold%d.pkl'%c_fold), 'wb') as f:
            pickle.dump(dev_regr_loss, f)

