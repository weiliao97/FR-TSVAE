import argparse

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
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
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
        
    train_head, train_static, train_sofa, train_id =  crop_data_target(train_vital, mimic_target, mimic_static, 'train')
    dev_head, dev_static, dev_sofa, dev_id =  crop_data_target(dev_vital , mimic_target, mimic_static, 'dev')
    test_head, test_static, test_sofa, test_id =  crop_data_target(test_vital, mimic_target, mimic_static, 'test')

    if args.use_sepsis3 == True:
        train_head, train_static, train_sofa, train_id = filter_sepsis(train_head, train_static, train_sofa, train_id)
        dev_head, dev_static, dev_sofa, dev_id = filter_sepsis(dev_head, dev_static, dev_sofa, dev_id)
        test_head, test_static, test_sofa, test_id = filter_sepsis(test_head, test_static, test_sofa, test_id)

    # build model

    # train model






