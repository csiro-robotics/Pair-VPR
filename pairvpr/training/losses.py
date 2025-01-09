# Copyright 2024-present CSIRO.
# Licensed under CSIRO BSD-3-Clause-Clear (Non-Commercial)
# Acknowledgements: https://github.com/amaralibey/MixVPR for the get_loss and get_miner functions.


import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity


def get_loss(loss_name):
    if loss_name == 'SupConLoss': return losses.SupConLoss(temperature=0.07)
    if loss_name == 'CircleLoss': return losses.CircleLoss(m=0.4, gamma=80) #these are params for image retrieval
    if loss_name == 'MultiSimilarityLoss': return losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
    if loss_name == 'ContrastiveLoss': return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if loss_name == 'Lifted': return losses.GeneralizedLiftedStructureLoss(neg_margin=0, pos_margin=1, distance=DotProductSimilarity())
    if loss_name == 'FastAPLoss': return losses.FastAPLoss(num_bins=30)
    if loss_name == 'NTXentLoss': return losses.NTXentLoss(temperature=0.07) #The MoCo paper uses 0.07, while SimCLR uses 0.5.
    if loss_name == 'TripletMarginLoss': return losses.TripletMarginLoss(margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor='all') #or an int, for example 100
    if loss_name == 'CentroidTripletLoss': return losses.CentroidTripletLoss(margin=0.05,
                                                                            swap=False,
                                                                            smooth_loss=False,
                                                                            triplets_per_anchor="all",)
    raise NotImplementedError(f'Sorry, <{loss_name}> loss function is not implemented.')


def get_miner(miner_name, margin=0.1):
    if miner_name == 'TripletMarginMiner' : return miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard") # all, hard, semihard, easy
    if miner_name == 'MultiSimilarityMiner' : return miners.MultiSimilarityMiner(epsilon=margin, distance=CosineSimilarity())
    if miner_name == 'PairMarginMiner' : return miners.PairMarginMiner(pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity())
    if miner_name == 'BatchHardMiner' : return miners.BatchHardMiner()
    return None



class MaskedMSELoss(torch.nn.Module):
    # Used during Stage One training
    def __init__(self, norm_pix_loss=True, masked=True):
        """
        norm_pix_loss: normalize each patch by their pixel mean and variance
        masked: compute loss over the masked patches only 
        """
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.masked = masked 
        
    def forward(self, pred, mask, target, dsetsource=None):
        """
        dsetsource: refers to the source dataset of the current loss item. Allows for per-dataset loss tracking.
        """
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        if self.masked:
            loss = loss * mask
            if dsetsource is not None:
                loss0 = (loss[dsetsource == 0]).sum() / mask[dsetsource == 0].sum()
                loss1 = (loss[dsetsource == 1]).sum() / mask[dsetsource == 1].sum()
                loss2 = (loss[dsetsource == 2]).sum() / mask[dsetsource == 2].sum()
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        if dsetsource is not None: # detached as currently used for logging only
            return loss, loss0.detach(), loss1.detach(), loss2.detach()
        else:
            return loss


class VPRLoss():
    def __init__(self):
        miner_name = 'MultiSimilarityMiner'
        loss_name = 'MultiSimilarityLoss'
        miner_margin = 0.1

        self.miner = get_miner(miner_name, miner_margin) # global miner
        self.localminer = get_miner('BatchHardMiner') # returns hardest positive and hardest negative
        # want local miner to minimise the number of pairs required, else memory required will explode
        self.loss_fn = get_loss(loss_name)

    def mining(self, descriptors, labels):
        return self.miner(descriptors, labels), self.localminer(descriptors, labels)

    def loss_function_global(self, descriptors, labels, miner_outputs):
        loss = self.loss_fn(descriptors, labels, miner_outputs)
        # calculate the % of trivial pairs/triplets
        # which do not contribute in the loss value
        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined / nb_samples)
        return loss, batch_acc

    def loss_function_legacy(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)

            # calculate the % of trivial pairs/triplets
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)

        else:  # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                # somes losses do the online mining inside (they don't need a miner objet),
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class,
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        return loss, batch_acc