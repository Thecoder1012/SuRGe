import torch.nn as nn
from torchvision.models import vgg19, resnet50
import config
import torch
import torch.nn.functional as F
import ot
import scipy as sp
import numpy as np
import warnings
import torch
from torch import nn
import torch.nn.functional as F
import scipy as sp
import ot
import torch
from ot.gromov import gromov_wasserstein2, gromov_wasserstein
# Ignore all warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gw_l_old_grad(image1_batch, image2_batch):
    batch_size = image2_batch.shape[0]
    # c1 /= c1.max()
    # c2 /= c2.max()
    image1_batch_f = torch.flatten(image1_batch, start_dim = 1)
    image2_batch_f = torch.flatten(image2_batch, start_dim = 1)

    #torch impl
    c1_torch = torch.cdist(image1_batch_f, image1_batch_f)
    c2_torch = torch.cdist(image2_batch_f, image2_batch_f)

    c1_torch /= c1_torch.max()
    c2_torch /= c2_torch.max()

    p = ot.unif(batch_size)
    q = ot.unif(batch_size)

    #torch impl
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    p_torch = torch.tensor(p).requires_grad_(True).to(device)
    q_torch = torch.tensor(q).to(device)
    c1 = c1_torch.cpu().detach().numpy()
    c2 = c2_torch.cpu().detach().numpy()

    gw0, log0 = ot.gromov.gromov_wasserstein(c1, c2, p, q, 'square_loss', epsilon = 5e-4, log = True)
    # gw_loss = gromov_wasserstein2(c1_torch, c2_torch, p_torch, q_torch)
    # gw, log = ot.gromov.entropic_gromov_wasserstein(c1, c2, p, q, 'square_loss', epsilon = 5e-4, log=True)
    # print(log0)
    return torch.tensor(log0['gw_dist'], requires_grad = True)
    # print(gw_loss)
    # return gw_loss

def jenson_shannon_divergence(net_1_logits, net_2_logits):
    # fake_dist = img_to_dist(net_1_logits)
    # high_res_dist = img_to_dist(net_2_logits)
    net_1_probs = F.softmax(net_1_logits, dim=0)
    net_2_probs = F.softmax(net_2_logits, dim=0)
    
    total_m = 0.5 * (net_1_probs + net_2_probs)
    
    # loss1 = 0.0
    # loss2 = 0.0
    loss1 = F.kl_div(F.log_softmax(net_1_logits, dim=0), total_m, reduction="batchmean") 
    loss2 = F.kl_div(F.log_softmax(net_2_logits, dim=0), total_m, reduction="batchmean") 
    return (0.5 * (loss1+loss2))

class dwloss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.softmax_temp = nn.Parameter(torch.tensor([0.1]))
        self.bce = nn.BCEWithLogitsLoss()
        self.softmax_temp = 0.1
        

    def forward(self, lr, sr, hr, disc_fake):
        # sigma, delta, kappa = F.softmax(self.weights, dim = 0)
        # tuner = self.softmax_temp
        # tuner = 0.1
       
        tuner = self.softmax_temp
        # gw1 = gw_l_n(lr, sr)
        gw = gw_l_old_grad(lr, sr)
        gw = gw.to(device)
        # gw = torch.tensor(gw1, requires_grad = True)
        js = jenson_shannon_divergence(sr, hr)
        adv = self.bce(disc_fake, torch.ones_like(disc_fake))
        # print(gw)
        # print(js)
        # print(adv)
        # loss = (sigma * gw) + (delta * js) + (kappa * adv)
        concat_loss = torch.cat([gw.unsqueeze(0), js.unsqueeze(0), adv.unsqueeze(0)], dim = 0)
        weights = F.softmax(concat_loss / tuner, dim = 0)
        loss = torch.sum(concat_loss * weights)

        return loss
