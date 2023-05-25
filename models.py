import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
mse_loss = nn.MSELoss()

class MLP(nn.Module):
    def __init__(self, num_neurons, zdim, activ="leakyrelu"):
        """Initialized VAE MLP"""
        super(MLP, self).__init__()
        self.num_neurons = num_neurons
        self.num_layers = len(self.num_neurons) - 1
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                for i in range(self.num_layers)
            ]
        )
        for hidden in self.hiddens:
            torch.nn.init.xavier_uniform_(hidden.weight)
        self.activ = activ
        self.zdim = zdim

    def forward(self, inputs, mode):
        """Computes forward pass for classifier, discriminator"""
        L = inputs # (16, 200)
        for hidden in self.hiddens:
            # (16, 200) --> (16, 2) for discriminator, (16, num_classes) for classifier 
            L = F.leaky_relu(hidden(L))
        if mode == "discriminator":
            logits, probs = L, nn.Softmax(dim=1)(L)
            return logits, probs
        elif mode == "classify":
            return L
        else:
            raise Exception(
                "Wrong mode choose one of discriminator/classifier"
            )
        return

# The following implementation is from
# @article{BaiTCN2018,
# 	author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
# 	title     = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
# 	journal   = {arXiv:1803.01271},
# 	year      = {2018},
# }
# link : https://github.com/locuslab/TCN

class Chomp1d(nn.Module):
    """
    To make sure causal convolution
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels=[256, 256, 256, 256], kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Sequential(
        nn.Linear(num_channels[-1], 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.network(x)
        x = x.contiguous().transpose(1, 2)
        x = self.linear(x)
        return x
    
    
class TemporalConv(nn.Module):
    def __init__(self, num_inputs, num_channels=[256, 128, 64, 40], kernel_size=2, dropout=0.2):
        super(TemporalConv, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        # self.linear = nn.Linear(num_channels[-1]*216, 120)
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(128, 128),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(128, 1)
        # )

    def forward(self, x):
        # (16, 200, 216) --> (16, 40, 216)
        x = self.network(x)
        # (17, 2, 216) --> (17, 432)
        # eg. t = torch.tensor([[[1, 2, 3], [3, 4, 4]], [[5, 6, 5], [7, 8, 4]], [[5, 6, 1], [7, 8, 1]]]) (2, 2, 3)
        # after flatten tensor([[1, 2, 3, 3, 4, 4],[5, 6, 5, 7, 8, 4], [5, 6, 1, 7, 8, 1]])
        # x = torch.flatten(x, start_dim=1, end_dim=2)
        # x = self.linear(x)
        mu_dim = x.shape[1]//2
        return x[:, :mu_dim, :], x[:, mu_dim:, :]

class Add1d(nn.Module):
    """
    To make sure causal convolution
    """
    def __init__(self, pad_size):
        super(Add1d, self).__init__()
        self.pad_size = pad_size

    def forward(self, x):
        return nn.ConstantPad1d((self.pad_size, 0), 0)(x)


class TemporalBlockDec(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlockDec, self).__init__()
        self.add1 = Add1d(padding)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.add2 = Add1d(padding)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.add1, self.conv1, self.relu1, self.dropout1,
                                 self.add2, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvDec(nn.Module):
    def __init__(self, num_inputs, num_channels=[64, 128, 256, 200], kernel_size=2, dropout=0.2):
        super(TemporalConvDec, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        # self.linear = nn.Linear(60, 216*2)
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(128, 128),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(128, 1)
        # )

    def forward(self, x):
        # (16, 60) --> (16, 432)
        # x = self.linear(x)
        # x = torch.reshape(x, (-1, 2, 216))
        # (16, 20, 216) --> (16, 200, 216)
        x = self.network(x)
        # (17, 2, 216) --> (17, 432)
        # eg. t = torch.tensor([[[1, 2, 3], [3, 4, 4]], [[5, 6, 5], [7, 8, 4]], [[5, 6, 1], [7, 8, 1]]]) (2, 2, 3)
        # after flatten tensor([[1, 2, 3, 3, 4, 4],[5, 6, 5, 7, 8, 4], [5, 6, 1, 7, 8, 1]])
        # x = torch.flatten(x, start_dim=1, end_dim=2)
        return x

    
class Ffvae(nn.Module):
    """Initializes FFVAE network: VAE encoder, MLP classifier, MLP discriminator"""
    def __init__(self, args, weights):
        super(Ffvae, self).__init__()

        self.lr = args.lr
        self.beta = args.beta
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.theta = args.theta
        self.zdim = args.zdim
        self.device =  torch.device("cuda:%d"%args.device_id if torch.cuda.is_available() else "cpu")
        self.scale_elbo = args.scale_elbo
        self.batch_size = args.bs
        self.weights = weights

        self.kernel_size = args.kernel_size
        self.drop_out = args.drop_out
        self.enc_channels = args.enc_channels
        self.dec_channels = args.dec_channels
        self.num_inputs = args.num_inputs

        self.disc_channels = args.disc_channels
        self.regr_channels = args.regr_channels if args.regr_model == 'mlp' else args.regr_tcn_channels
        self.regr_model = args.regr_model 
        self.regr_only_nonsens = args.regr_only_nonsens
        
        # VAE encoder

        self.encoder  = TemporalConv(num_inputs=self.num_inputs, num_channels=self.enc_channels, kernel_size=self.kernel_size, dropout=self.drop_out)
        self.decoder = TemporalConvDec(num_inputs=self.zdim, num_channels=self.dec_channels, kernel_size=self.kernel_size, dropout=self.drop_out)

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        # MLP Discriminator
        self.adv_neurons = [args.zdim] + [self.disc_channels] + [2]
        self.discriminator = MLP(self.adv_neurons, args.zdim).to(self.device)
        self.class_neurons = ([args.zdim] + [200] + [1])
            
        if self.regr_only_nonsens:
            self.regr_neurons = ([args.zdim-1] + [200] + [1])
            self.regr = MLP(self.regr_neurons, args.zdim).to(self.device)
        else:
            self.regr_neurons = ([args.zdim] + [200] + [1])
            self.regr = MLP(self.regr_neurons, args.zdim).to(self.device)

        # MLP Classifier
        if args.regr_model == 'mlp':
            
            self.classifier = MLP(self.class_neurons, args.zdim).to(self.device)
            
        elif args.regr_model == 'tcn':
            
            self.classifier = TCN(num_inputs=self.zdim, num_channels=self.regr_channels,
                                        kernel_size=self.kernel_size, dropout=self.drop_out).to(self.device)
        
        else:
            raise NotImplementedError("regr_model not implemented")

        # index for sensitive attribute
        self.n_sens = 1
        self.sens_idx = list(range(self.n_sens))
        self.nonsens_idx = [
            i for i in range(int(self.zdim)) if i not in self.sens_idx
        ]
        # self.count = 0
        
        (
            self.optimizer_ffvae,
            self.optimizer_disc,
            self.optimizer_class,
        ) = self.get_optimizer()

    @staticmethod
    def build_model(args):
        """ Builds FFVAE class """
        model = Ffvae(args)
        return model

    def vae_params(self):
        """Returns VAE parameters required for training VAE"""

        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def discriminator_params(self):
        """Returns discriminator parameters"""
        return list(self.discriminator.parameters())

    def classifier_params(self):
        """Returns classifier parameters"""
        return list(self.classifier.parameters())
    
    def regr_params(self):
        """Returns regr parameters"""
        return list(self.regr.parameters())

    def get_optimizer(self):
        """Returns an optimizer for each network"""
        optimizer_ffvae = torch.optim.Adam(self.vae_params() + self.regr_params(), lr=self.lr)
        optimizer_disc = torch.optim.Adam(self.discriminator_params(), lr=self.lr)
        optimizer_class = torch.optim.Adam(self.classifier_params(),  lr=self.lr)
        return optimizer_ffvae, optimizer_disc, optimizer_class

    def forward(self, inputs, key_mask, labels, attrs, mode="train"):
        """Computes forward pass through encoder ,
            Computes backward pass on the target function"""
        # # Make inputs between 0, 1
        # x = (inputs + 1) / 2

        # encode: get q(z,b|x)
        # (bs, z_dim, T)
        _mu, _logvar = self.encoder(inputs)

        # only non-sensitive dims of latent code modeled as Gaussian
        mu = _mu[:, self.nonsens_idx, :]
        logvar = _logvar[:, self.nonsens_idx, :]
        zb = torch.zeros_like(_mu) 
        # (bs, nonsens_dim, T)
        std = (logvar / 2).exp()
        q_zIx = torch.distributions.Normal(mu, std)

        # the rest are 'b', deterministically modeled as logits of sens attrs a
        # (bs, 1, T)
        b_logits = _mu[:, self.sens_idx, :]

        # draw reparameterized sample and fill in the code
        # (bs, nonsens_dim, T)
        z = q_zIx.rsample()
        # reparametrization
        zb[:, self.sens_idx, :] = b_logits
        zb[:, self.nonsens_idx, :] = z

        # decode: get p(x|z,b)
        # xIz_params = self.decoder(zb, "decode")  # decoder yields distn params not preds
    
        # (bs, z_dim, T) --> (bs, 200, T)
        xIz_params = self.decoder(zb)

        p_xIz = torch.distributions.Normal(loc=xIz_params, scale=1.0)
        # negative recon error per example
        logp_xIz = p_xIz.log_prob(inputs)  

        # Tensor with shape torch.Size([bs]) 
        recon_term = torch.stack([logp_xIz[i, :, key_mask[i] == 0].sum() for i in range(len(logp_xIz))])

        # prior: get p(z)
        p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        # compute analytic KL from q(z|x) to p(z), then ELBO, torch.Size([bs])
        kl = torch.distributions.kl_divergence(q_zIx, p_z).sum((1, 2))

        # vector el
        elbo = recon_term - kl  
        if self.scale_elbo: 
            # scale elbo by time axis
            elbo = elbo / inputs.shape[-1]

        # decode: get p(a|b)
        # b logits shape torch.Size([bs, 1, T]), converts to [bs] based on real length 
        b_squeeze = torch.stack([b_logits[i].squeeze(0)[key_mask[i]==0].mean() for i in range(len(b_logits))])
        clf_losses = [
            nn.BCEWithLogitsLoss()(_b_logit.to(self.device), _a_sens.to(self.device))
            for _b_logit, _a_sens in zip(
            b_squeeze.t(), attrs.type(torch.FloatTensor).t())]
        
        # weighted 
        clf_w_losses = [
            nn.BCEWithLogitsLoss(pos_weight = self.weights[1]/self.weights[0])(_b_logit.to(self.device), _a_sens.to(self.device))
            for _b_logit, _a_sens in zip(
            b_squeeze.t(), attrs.type(torch.FloatTensor).t())]

        # compute loss
        # (bs, T, 2)
        logits_joint, probs_joint = self.discriminator(zb.transpose(1, 2), "discriminator")
        # consider mask 
        # [Tensor with shape torch.Size([T1, 2]), Tensor with shape torch.Size([T2, 2])
        logits_recover = [logits_joint[i][key_mask[i]==0]for i in range(len(logits_joint))]
#         torch.Size([bs])
        total_corr = torch.stack([(l[:, 0] - l[:, 1]).mean() for l in logits_recover])
        
        # add sofa loss classifier has to be tune as well 
        if self.regr_only_nonsens:
            sofa_p_s1 = self.regr(mu.transpose(1, 2), "classify")
        else:
            sofa_p_s1 = self.regr(_mu.transpose(1, 2), "classify")
        sofa_loss = [mse_loss(sofa_p_s1[i][key_mask[i]==0], labels[i][key_mask[i]==0]) \
            for i in range(len(sofa_p_s1))]
#         regr_loss = torch.mean(torch.stack(sofa_loss))

        # random elbo 10^4, totoal_corr 10^-1, clf_losses 10^-2
        ffvae_loss = (
            - self.beta * elbo.mean()
            + self.gamma * total_corr.mean()
            + self.alpha * torch.stack(clf_losses).mean()
            + self.theta * torch.stack(sofa_loss).mean()
        )

        # shuffling minibatch indexes of b0, b1, z
        z_fake = torch.zeros_like(zb)
        z_fake[:, 0, :] = zb[:, 0, :][torch.randperm(zb.shape[0])]
        z_fake[:, 1:, :] = zb[:, 1:, :][torch.randperm(zb.shape[0])]
        z_fake = z_fake.to(self.device).detach()

        # discriminator
#         (bs, T, 2 )
        logits_joint_prime, probs_joint_prime = self.discriminator(
            z_fake.transpose(1, 2), "discriminator"
        )
        logits_prime_recover = [logits_joint_prime[i][key_mask[i]==0]for i in range(len(logits_joint_prime))]
        # 10^-1 torch.Size([])
        disc_loss = (
        0.5
        * (
            torch.stack([F.cross_entropy(logits_recover[i], torch.zeros(logits_recover[i].shape[0], dtype=torch.long, device=self.device))
            for i in range(len(logits_recover))]).mean()
            + 
            torch.stack([F.cross_entropy(logits_prime_recover[i], torch.ones(logits_prime_recover[i].shape[0], dtype=torch.long, device=self.device)) 
            for i in range(len(logits_prime_recover))]).mean()
        ).mean() )

        encoded_x = _mu.detach()

        # IMPORTANT: randomizing sensitive latent
        encoded_x[:, 0, :] = torch.randn_like(encoded_x[:, 0, :])

        # torch.Size([bs, T, 1])
        if self.regr_model == 'mlp':
            sofa_p = self.classifier(encoded_x.transpose(1, 2), "classify")
        elif self.regr_model == 'tcn':
            # encoder x (bs, z_dim, T) 
            sofa_p =  self.classifier(encoded_x)
            
        loss = [mse_loss(sofa_p[i][key_mask[i]==0], labels[i][key_mask[i]==0]) \
            for i in range(len(sofa_p))]
        sofap_loss = torch.mean(torch.stack(loss))

        cost_dict = dict(
            ffvae_cost=ffvae_loss, recon_cost=recon_term.mean(), kl_cost=kl.mean(), corr_term=total_corr.mean(), clf_term = torch.stack(clf_losses).mean(), clf_w_term = torch.stack(clf_w_losses).mean(), sofa_term = torch.stack(sofa_loss).mean(), disc_cost=disc_loss, main_cost=sofap_loss
        )

        # ffvae optimization
        if mode == "ffvae_train":
            self.optimizer_ffvae.zero_grad()
            ffvae_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.vae_params(), 5.0)
            self.optimizer_ffvae.step()

            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator_params(), 5.0)
            self.optimizer_disc.step()

        # classifier optimization
        elif mode == "train":
            self.optimizer_class.zero_grad()
            sofap_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier_params(), 5.0)
            self.optimizer_class.step()

        return sofa_p, cost_dict