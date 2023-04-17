import argparse
import numpy as np
import torch
import utils
import prepare_data
import models
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=None, shuffle=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for time series VAE models")
    # data/loss parameters
    parser.add_argument("--use_sepsis3", action = 'store_false', default= True, help="Whethe only use sepsis3 subset")
    parser.add_argument("--bucket_size", type=int, default=300, help="bucket size to group different length of time-series data")
    parser.add_argument("--gamma", type=float, default=0.5, help="coefficent for the total_corr loss")
    parser.add_argument("--alpha", type=float, default=0.5, help="coefficent for the clf loss")
    parser.add_argument("--zdim", type=int, default=20, help="dimension of the latent space")
    # model parameters
    parser.add_argument("--kernel_size", type=int, default=3, help="kernel size")
    parser.add_argument("--drop_out", type=float, default=0.2, help="drop out rate")
    parser.add_argument("--enc_channels",  nargs='+', help="number of channels in the encoder")
    parser.add_argument("--dec_channels",  nargs='+', help="number of channels in the decoder")
    parser.add_argument("--num_inputs", type=int, default=200, help="number of features in the inputs")
    # discriminator parameters
    parser.add_argument("--disc_channels",  type=int, default=200, help="number of channels in the discriminator")
    # regressor parameters
    parser.add_argument("--regr_channels",  type=int, default=200, help="number of channels in the regressor")
    # training parameters
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--data_batching", type=str, default='close', choices=['same', 'close', 'random'], help='How to batch data')
    parser.add_argument("--bs", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default='test', help=" name of checkpoint model")

    args = parser.parse_args()

    # load data
    meep_mimic = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_compile_0911_2022.npy', \
                    allow_pickle=True).item()
    train_vital = meep_mimic ['train_head']
    dev_vital = meep_mimic ['dev_head']
    test_vital = meep_mimic ['test_head']
    mimic_static = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_static_0922_2022.npy', \
                            allow_pickle=True).item()
    mimic_target = np.load('/content/drive/MyDrive/ColabNotebooks/MIMIC/Extract/MEEP/Extracted_sep_2022/0910/MIMIC_target_0922_2022.npy', \
                            allow_pickle=True).item()
        
    train_head, train_static, train_sofa, train_id =  utils.crop_data_target(train_vital, mimic_target, mimic_static, 'train')
    dev_head, dev_static, dev_sofa, dev_id =  utils.crop_data_target(dev_vital , mimic_target, mimic_static, 'dev')
    test_head, test_static, test_sofa, test_id =  utils.crop_data_target(test_vital, mimic_target, mimic_static, 'test')

    if args.use_sepsis3 == True:
        train_head, train_static, train_sofa, train_id = utils.filter_sepsis(train_head, train_static, train_sofa, train_id)
        dev_head, dev_static, dev_sofa, dev_id = utils.filter_sepsis(dev_head, dev_static, dev_sofa, dev_id)
        test_head, test_static, test_sofa, test_id = utils.filter_sepsis(test_head, test_static, test_sofa, test_id)

    # build model
    model = models.ffvae(args)

    # 10-fold cross validation
    trainval_head = train_head + dev_head
    trainval_static = train_static + dev_static
    trainval_stail = train_sofa + dev_sofa
    trainval_ids = train_id + dev_id

    # prepare data
    for c_fold, (train_index, test_index) in enumerate(kf.split(trainval_head)):
        best_loss = 1e4
        patience = 0
        if c_fold >= 1:
            model.load_state_dict(torch.load('/content/start_weights.pt'))
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
                                                                                            dev_static=dev_static,
                                                                                            test_static=test_static,
                                                                                            train_id=train_id,
                                                                                            dev_id=val_id,
                                                                                            test_id=test_id)






