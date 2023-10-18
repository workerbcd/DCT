import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import math

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
def pair_mining(feats,proxy,labels,domains):
    N = feats.shape[0]
    feat_matrix = torch.exp(feats @ feats.t())
    featproxy_matrix = torch.exp(proxy @ feats.t())
    is_domain_pos = domains.expand(N, N).eq(domains.expand(N, N).t())
    is_domain_neg = domains.expand(N, N).ne(domains.expand(N, N).t())
    is_label_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_label_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    domainpos_labelneg = is_domain_pos & is_label_neg
    domainneg_labelpos = is_domain_neg & is_label_pos

    dist_ff, dist_pf = [], []
    dist_ff_p,dist_pf_p = [], []
    for i in range(N):
        dist_ff.append(torch.sum(feat_matrix[i][domainpos_labelneg[i]].contiguous())+torch.zeros(1).cuda())
        dist_pf.append(torch.sum(featproxy_matrix[i][is_label_neg[i]].contiguous())+torch.zeros(1).cuda())
        dist_ff_p.append(torch.sum(feat_matrix[i][domainneg_labelpos[i]]).contiguous()+torch.zeros(1).cuda())
        dist_pf_p.append(torch.sum(featproxy_matrix[i][is_label_pos[i]]).contiguous().unsqueeze(0))
    dist_ff = torch.cat(dist_ff)
    dist_pf = torch.cat(dist_pf)
    dist_ff_p = torch.cat(dist_ff_p)
    for d in dist_pf_p:
        if 0 in d.shape:
            print("d.shape")
    dist_pf_p = torch.cat(dist_pf_p,dim=0)
    return dist_ff,dist_pf, dist_pf_p+dist_ff_p

class PCLLoss(object):
    def __init__(self,scale=1):
        self.scale = scale
    def __call__(self, feats, proxy,labels,domainlabels):
        feats = normalize(feats)
        proxy = normalize(proxy)
        sum_ff, sum_pf, pos = pair_mining(feats,proxy,labels,domainlabels)
        loss = (-torch.log(pos/(sum_pf+sum_ff+pos))).mean(0).sum()
        return loss


class ProxyPLoss(nn.Module):
    '''
    pass
    '''

    def __init__(self, num_classes, scale):
        super(ProxyPLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.LongTensor([i for i in range(num_classes)]).cuda()
        self.scale = scale

    def forward(self, feature, target, proxy):
        feature = F.normalize(feature, p=2, dim=1)
        pred = F.linear(feature, F.normalize(proxy, p=2, dim=1))  # (N, C)

        # label = (self.label.unsqueeze(1) == target.unsqueeze(0))
        # pred = torch.masked_select(pred.transpose(1, 0), label)
        # pred = pred.unsqueeze(1)

        feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)

        # index_label = torch.LongTensor([i for i in range(feature.shape[0])])  # generate index label
        # index_matrix = index_label.unsqueeze(1) == index_label.unsqueeze(0)  # get index matrix

        feature = feature * ~label_matrix  # get negative matrix
        feature = feature.masked_fill(feature < 1e-6, -np.inf)

        logits = torch.cat([pred, feature], dim=1)  # (N, C+N)
        # print(pred.shape, feature.shape)
        # label = torch.zeros(logits.size(0), dtype=torch.long).cuda()
        loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), target)

        return loss

class ProxyDCLoss(nn.Module):
    '''
    pass
    '''

    def __init__(self, num_classes,domain_classes,scale):
        super(ProxyDCLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.LongTensor([i for i in range(num_classes)]).cuda()
        self.domains = torch.LongTensor([i for i in range(domain_classes)]).cuda()
        self.scale = scale

    def forward(self, feature, target, domain_label, proxy,d_proxy):
        # feature = F.normalize(feature, p=2, dim=1)
        # pred = F.linear(feature, F.normalize(proxy, p=2, dim=1))  # (N, C)
        # d_pred = F.linear(feature,F.normalize(d_proxy,p=2,dim=1)) # (N,D)
        #
        # label = (self.label.unsqueeze(1) == target.unsqueeze(0))
        # pred = torch.masked_select(pred.transpose(1, 0), label)
        # pred = pred.unsqueeze(1)
        #
        # feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
        # label_matrix = target.unsqueeze(1) == target.unsqueeze(0)
        # domain_matric = domain_label.unsqueeze(1) == domain_label.unsqueeze(0)
        #
        # # index_label = torch.LongTensor([i for i in range(feature.shape[0])])  # generate index label
        # # index_matrix = index_label.unsqueeze(1) == index_label.unsqueeze(0)  # get index matrix
        #
        # d_feature = feature * ~domain_matric
        # d_feature = d_feature.masked_fill(d_feature<1e-6, -np.inf)
        # feature = feature * ~label_matrix  # get negative matrix
        # feature = feature.masked_fill(feature < 1e-6, -np.inf)
        #
        # logits = torch.cat([pred, feature], dim=1)  # (N, C+N)
        # d_logits = torch.cat([d_pred,d_feature],dim=1)
        # # print(pred.shape, feature.shape)
        # # label = torch.zeros(logits.size(0), dtype=torch.long).cuda()
        # loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), target)
        # d_loss = F.nll_loss(F.log_softmax(self.scale* d_logits, dim =1), domain_label)
        # return loss / d_loss.clamp(1e-12)
        feature = F.normalize(feature, p=2, dim=1)
        pred = F.linear(feature, F.normalize(proxy, p=2, dim=1))  # (N, C)

        # label = (self.label.unsqueeze(1) == target.unsqueeze(0))
        # pred = torch.masked_select(pred.transpose(1, 0), label)
        # pred = pred.unsqueeze(1)

        feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)
        domain_matric = domain_label.unsqueeze(1) == domain_label.unsqueeze(0)
        domainpos_labelneg = (~label_matrix) & domain_matric
        # domainneg_labelpos = (~domain_matric) & label_matrix
        # index_label = torch.LongTensor([i for i in range(feature.shape[0])])  # generate index label
        # index_matrix = index_label.unsqueeze(1) == index_label.unsqueeze(0)  # get index matrix


        feature_neg = feature * domainpos_labelneg  # get negative matrix
        # feature_pos = feature * domainneg_labelpos
        feature_neg = feature_neg.masked_fill(feature_neg < 1e-6, -np.inf)
        # feature_pos = feature_pos.masked_fill(feature_pos < 1e-6, -np.inf)



        logits = torch.cat([pred, feature_neg], dim=1)  # (N, N+1+N)
        # logits_u = torch.cat([feature_pos,pred],dim=1)
        # logits = torch.sum(torch.exp(logits_u),dim=1) / torch.sum(torch.exp(logits_d),dim=1)
        # logits = torch.log(logits)
        # logits = F.log_softmax(self.scale*logits,dim=1)
        # N = logits.size(0)

        # print(pred.shape, feature.shape)
        # label = torch.zeros(logits.size(0), dtype=torch.long).cuda()
        loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), target)
        # loss = F.nll_loss(logits,label)
        return loss