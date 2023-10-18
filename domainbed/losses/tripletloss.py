import torch
from torch import nn
import domainbed.networks as networks


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


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    #
    # # `dist_ap` means distance(anchor, positive)
    # # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    # dist_ap, relative_p_inds = torch.max(
    #     dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    # dist_an, relative_n_inds = torch.min(
        # dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_aps,dist_ans = [],[]
    for i in range(N):
        dist_ap, _ = torch.max(dist_mat[i][is_pos[i]].contiguous(), 0, keepdim=True)
        dist_an, _ = torch.max(dist_mat[i][is_neg[i]].contiguous(), 0, keepdim=True)
        dist_aps.append(dist_ap)
        dist_ans.append(dist_an)
    # shape [N]
    dist_aps = torch.cat(dist_aps).clamp(min=1e-12)
    dist_ans = torch.cat(dist_ans).clamp(min=1e-12)

    return dist_aps, dist_ans

def domain_hard_sample_mining(dist_mat, labels,domains,leri=None):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_domain_pos = domains.expand(N,N).eq(domains.expand(N, N).t())
    is_domain_neg = domains.expand(N,N).ne(domains.expand(N,N).t())
    is_2 = domains.expand(N,N).eq(torch.ones(N,N).cuda())
    is_label_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_label_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    domainpos_labelneg = is_label_neg# & is_domain_pos
    domainneg_labelpos = is_label_pos #& is_domain_neg
    if leri:
        is_domain_pos = is_domain_pos | is_2
        is_domain_neg = is_domain_neg | is_2
        domainpos_labelneg = is_label_neg & is_domain_pos
        domainneg_labelpos = is_label_pos & is_domain_neg


    dist_dpln,dist_dnlp =[],[]
    for i in range(N):
        if dist_mat[i][domainneg_labelpos[i]].shape[0]!=0:
        # if leri!="neg" and dist_mat[i][domainneg_labelpos[i]].shape[0]!=0:
            dist_dnlp.append(torch.max(dist_mat[i][domainneg_labelpos[i]].contiguous(), 0, keepdim=True)[0])
        else:
            dist_dnlp.append(torch.zeros(1).cuda())
        if dist_mat[i][domainpos_labelneg[i]].shape[0]!=0:
            dist_dpln.append(torch.min(dist_mat[i][domainpos_labelneg[i]].contiguous(), 0, keepdim=True)[0])
        else:
            dist_dpln.append(torch.zeros(1).cuda())

    dist_dnlp = torch.cat(dist_dnlp).clamp(min=1e-12)
    dist_dpln = torch.cat(dist_dpln).clamp(min=1e-12)

    return dist_dnlp,dist_dpln



class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels,cam,normalize_feature=False):
        # if normalize_feature:
        #     global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class DomainTripletLoss(object):
    def __init__(self,hparams,margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        self.bottleneck = nn.BatchNorm1d(hparams["out_dim"]).cuda()
        self.hp = hparams
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(networks.weights_init_kaiming)
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, camlabel,normalize_feature=False,use_bn=True):
        if use_bn:
            global_feat = self.bottleneck(global_feat)
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = domain_hard_sample_mining(dist_mat, labels,camlabel,leri=self.hp["test_leri"])

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None :
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_mat

