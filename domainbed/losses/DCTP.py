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

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class DomainTripletLoss_Prototype(object):
    def __init__(self, prototype,margin=None, hard_factor=0.0):
        self.margin = margin
        self.classnum, self.domainum, featsize= prototype.shape
        # self.prototype = nn.Parameter(prototype.cuda()) # shape class x doamin x featsize
        self.prototype = nn.Parameter(torch.rand(self.classnum,featsize).cuda())
        nn.init.kaiming_uniform_(self.prototype, a=math.sqrt(5))
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
        self.softmargin = nn.SoftMarginLoss()
    def initproto(self,dataloader,network):
        network.eval()
        featscount = np.array([0] * (self.num_classes * self.num_domains), dtype=np.float32)
        with torch.no_grad():
            for loader in dataloader:
                for xyz in loader:
                    try:
                        x, y = xyz
                        dmy = None
                    except:
                        x, y, dmy = xyz
                    x.cuda()
                    pos = y * self.num_domains + dmy
                    pos_ = pos
                    pos = pos.cpu().detach().numpy()
                    pos = Counter(pos)
                    c = pos.items()
                    c0 = [t[0] for t in c]
                    c1 = [t[1] for t in c]
                    featscount[c0] += c1
                    p = self.get_feats(x)
                    self.prototypes.index_add(0, pos_, p)
        featscount = torch.tensor(featscount).view(-1, 1)
        self.prototypes = self.prototypes / featscount
        self.prototypes = self.prototypes.view(self.num_classes, self.num_domains, -1)
        del featscount
        print("---------------prototype prepared-------------")

    def __call__(self, global_feat, labels=None, camlabel=None, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
            self.prototype = normalize(self.prototype,axis=-1)
# ##----------------------domain-class implementation-------------------
#         domainsum_proto = self.prototype.sum(dim=1,keepdim=True)
#         classsum_proto = self.prototype.sum(dim=0,keepdim=True)
#         class_feat = [F.pairwise_distance(global_feat, domainsum_proto[i, 0, :]/ self.domainum, p=1, keepdim=True) \
#                       for i in range(self.classnum)]
#         class_feat = torch.cat(class_feat, dim=1)
#         if labels==None:
#             return 1.0/class_feat
#         domainsum_labelpos_proto = domainsum_proto.data[labels.long(),0,:]
#         domainpos_classsum_proto = classsum_proto.data[0,camlabel.long(),:]
#         # domainpos_labelpos_proto = self.prototype.data[labels.long(),camlabel.long(),:]
#         # domainpos_labelneg_protomean = (domainpos_classsum_proto-domainpos_labelpos_proto)/(self.classnum-1)
#         # domainneg_classpos_protomean = (domainsum_labelpos_proto - domainpos_labelpos_proto)/(self.domainum-1)
#         domainpos_labelneg_protomean = domainpos_classsum_proto / self.classnum
#         domainneg_classpos_protomean = domainsum_labelpos_proto / self.domainum
#
#         dist_ap = F.pairwise_distance(global_feat,domainneg_classpos_protomean,p=1)
#         dist_an = F.pairwise_distance(global_feat,domainpos_labelneg_protomean,p=1)
#         domainprotomean = domainsum_proto/self.domainum
#         classprotomean = classsum_proto/self.classnum
#         domainprotomean = domainprotomean.view(self.classnum,1,-1).expand(self.prototype.size())
#         classprotomean = classprotomean.view(1,self.domainum,-1).expand(self.prototype.size())
#         proto_pos = F.pairwise_distance(self.prototype,domainprotomean,p=1).view(self.domainum*self.classnum)
#         proto_neg = F.pairwise_distance(self.prototype,classprotomean,p=1).view(self.domainum*self.classnum)
#
#         y = dist_an.new().resize_as_(dist_an).fill_(1)
#         protoy = proto_neg.new().resize_as_(proto_neg).fill_(1)
#         if self.margin is not None:
#             loss = self.ranking_loss(dist_an, dist_ap, y)
#         else:
#             loss = self.ranking_loss(dist_an - dist_ap, y)
#         loss_proto = self.softmargin(proto_neg-proto_pos,protoy)
#         return 1.0/class_feat, loss, loss_proto
###-------------class implementation----------------------
        # posproto = self.prototypes[labels.long(),:]
        # class_feat = [F.pairwise_distance(global_feat, self.prototype[i,:], p=2, keepdim=True) \
        #               for i in range(self.classnum)]
        # class_feat = torch.cat(class_feat, dim=1)
        class_feat = torch.matmul(global_feat, self.prototype.t())
        return class_feat
