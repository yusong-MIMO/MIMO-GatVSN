import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
CE = torch.nn.BCELoss(reduction='sum')


class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size = 4):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        self.channel = channels

        # self.fc1_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc2_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc1_depth1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc2_depth1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # 
        # self.fc1_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc2_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc1_depth2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc2_depth2 = nn.Linear(channels * 1 * 22 * 22, latent_size)

        # For a 72x72 input, the output feature map size is 18x18
        feature_size = channels * 1 * 18 * 18
        self.fc1_rgb = nn.Linear(feature_size, latent_size)
        self.fc2_rgb = nn.Linear(feature_size, latent_size)
        self.fc1_depth = nn.Linear(feature_size, latent_size)
        self.fc2_depth = nn.Linear(feature_size, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.layer3(self.leakyrelu(self.layer1(rgb_feat)))
        depth_feat = self.layer4(self.leakyrelu(self.layer2(depth_feat)))
        # print(rgb_feat.size())
        # print(depth_feat.size())
        # if rgb_feat.shape[2] == 16:
        #     rgb_feat = rgb_feat.view(-1, self.channel * 1 * 16 * 16)
        #     depth_feat = depth_feat.view(-1, self.channel * 1 * 16 * 16)
        #
        #     mu_rgb = self.fc1_rgb1(rgb_feat)
        #     logvar_rgb = self.fc2_rgb1(rgb_feat)
        #     mu_depth = self.fc1_depth1(depth_feat)
        #     logvar_depth = self.fc2_depth1(depth_feat)
        # elif rgb_feat.shape[2] == 22:
        #     rgb_feat = rgb_feat.view(-1, self.channel * 1 * 22 * 22)
        #     depth_feat = depth_feat.view(-1, self.channel * 1 * 22 * 22)
        #     mu_rgb = self.fc1_rgb2(rgb_feat)
        #     logvar_rgb = self.fc2_rgb2(rgb_feat)
        #     mu_depth = self.fc1_depth2(depth_feat)
        #     logvar_depth = self.fc2_depth2(depth_feat)
        # else:
        rgb_feat = rgb_feat.view(-1, self.channel * 1 * 18 * 18)
        depth_feat = depth_feat.view(-1, self.channel * 1 * 18 * 18)
        mu_rgb = self.fc1_rgb(rgb_feat)
        logvar_rgb = self.fc2_rgb(rgb_feat)
        mu_depth = self.fc1_depth(depth_feat)
        logvar_depth = self.fc2_depth(depth_feat)

        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm,z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth+ce_depth_rgb-bi_di_kld
        # latent_loss = torch.abs(cos_sim(z_rgb,z_depth)).sum()

        return latent_loss