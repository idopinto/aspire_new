"""
Functions for computing distances between documents with fine grained representations.
"""
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional
import geomloss

class AllPairMaskedWasserstein:
    def __init__(self, model_hparams):
        self.geoml_blur = model_hparams.get('geoml_blur', 0.05)
        self.geoml_scaling = model_hparams.get('geoml_scaling', 0.9)
        self.geoml_reach = model_hparams.get('geoml_reach', None)
        self.sent_sm_temp = model_hparams.get('sent_sm_temp', 1.0)

    def compute_distance(self, query, cand, return_pair_sims=False):
        """
        Given a set of query and candidate reps compute the wasserstein distance between
        the query and candidates.
        :param query: namedtuple(
            embed: batch_size x encoding_dim x q_max_sents;
            abs_lens: list(int); number of sentences in every batch element.)
        :param cand: namedtuple(
            embed: batch_size x encoding_dim x q_max_sents;
            abs_lens: list(int); number of sentences in every batch element.)
        :return:
            batch_sims: ef_batch_size; pooled pairwise _distances_ between
                input reps. (distances are just negated similarities here)
        """
        query_reps, query_abs_lens = query.embed, query.abs_lens
        cand_reps, cand_abs_lens = cand.embed, cand.abs_lens
        qef_batch_size, _, qmax_sents = query_reps.size()
        cef_batch_size, encoding_dim, cmax_sents = cand_reps.size()
        pad_mask = np.ones((qef_batch_size, qmax_sents, cmax_sents))*-10e8
        for i in range(qef_batch_size):
            ql, cl = query_abs_lens[i], cand_abs_lens[i]
            pad_mask[i, :ql, :cl] = 0.0
        pad_mask = Variable(torch.FloatTensor(pad_mask))
        if torch.cuda.is_available():
            pad_mask = pad_mask.cuda()
        assert (qef_batch_size == cef_batch_size)
        # (effective) batch_size x qmax_sents x cmax_sents
        # inputs are: batch_size x encoding_dim x c/qmax_sents so permute them.
        neg_pair_dists = -1*torch.cdist(query_reps.permute(0, 2, 1).contiguous(),
                                        cand_reps.permute(0, 2, 1).contiguous())
        if len(neg_pair_dists.size()) == 2:
            neg_pair_dists = neg_pair_dists.unsqueeze(0)
        assert (neg_pair_dists.size(1) == qmax_sents)
        assert (neg_pair_dists.size(2) == cmax_sents)
        # Add very large negative values in the pad positions which will be zero.
        neg_pair_dists = neg_pair_dists + pad_mask
        q_max_sent_sims, _ = torch.max(neg_pair_dists, dim=2)
        c_max_sent_sims, _ = torch.max(neg_pair_dists, dim=1)
        query_distr = functional.log_softmax(q_max_sent_sims/self.sent_sm_temp, dim=1).exp()
        cand_distr = functional.log_softmax(c_max_sent_sims/self.sent_sm_temp, dim=1).exp()
        if return_pair_sims:
            # This is only used at test time -- change the way the pad mask is changed in place
            # if you want to use at train time too.
            pad_mask[pad_mask == 0] = 1.0
            pad_mask[pad_mask == -10e8] = 0.0
            neg_pair_dists = neg_pair_dists * pad_mask
            # p=1 is the L2 distance oddly enough.
            ot_solver = geomloss.SamplesLoss("sinkhorn", p=1, blur=self.geoml_blur, reach=self.geoml_reach,
                                             scaling=self.geoml_scaling, debias=False, potentials=True)
            # Input reps to solver need to be: batch_size x c/qmax_sents x encoding_dim
            q_pot, c_pot = ot_solver(query_distr, query_reps.permute(0, 2, 1).contiguous(),
                                     cand_distr, cand_reps.permute(0, 2, 1).contiguous())
            # Implement the expression to compute the plan from the potentials:
            # https://www.kernel-operations.io/geomloss/_auto_examples/optimal_transport/
            # plot_optimal_transport_labels.html?highlight=plan#regularized-optimal-transport
            outersum = q_pot.unsqueeze(dim=2).expand(-1, -1, cmax_sents) + \
                       c_pot.unsqueeze(dim=2).expand(-1, -1, qmax_sents).permute(0, 2, 1)
            # Zero out the pad values because they seem to cause nans to occur.
            outersum = outersum * pad_mask
            exps = torch.exp(torch.div(outersum+neg_pair_dists, self.geoml_blur))
            outerprod = torch.einsum('bi,bj->bij', query_distr, cand_distr)
            transport_plan = exps*outerprod
            pair_sims = neg_pair_dists
            masked_sims = transport_plan*pair_sims
            wasserstein_dists = torch.sum(torch.sum(masked_sims, dim=1), dim=1)
            return wasserstein_dists, [query_distr, cand_distr, pair_sims, transport_plan, masked_sims]
        else:
            ot_solver_distance = geomloss.SamplesLoss("sinkhorn", p=1, blur=self.geoml_blur, reach=self.geoml_reach,
                                                      scaling=self.geoml_scaling, debias=False, potentials=False)
            wasserstein_dists = ot_solver_distance(query_distr, query_reps.permute(0, 2, 1).contiguous(),
                                                   cand_distr, cand_reps.permute(0, 2, 1).contiguous())
            return wasserstein_dists

def allpair_masked_dist_l2max(query, cand, return_pair_sims=False):
    """
    Given a set of query and candidate reps compute dot product similarity
    between all the query facet reps and all the candidate facet reps,
    then aggregate similarity with a log sum exp.
    :param query: namedtuple(
        embed: batch_size x encoding_dim x q_max_sents;
        abs_lens: list(int); number of sentences in every batch element.)
    :param cand: namedtuple(
        embed: batch_size x encoding_dim x q_max_sents;
        abs_lens: list(int); number of sentences in every batch element.)
    :return:
        batch_sims: ef_batch_size; pooled pairwise _distances_ between
            input reps. (distances are just negated similarities here)
    """
    query_reps, query_abs_lens = query.embed, query.abs_lens
    cand_reps, cand_abs_lens = cand.embed, cand.abs_lens
    qef_batch_size, _, qmax_sents = query_reps.size()
    cef_batch_size, encoding_dim, cmax_sents = cand_reps.size()
    pad_mask = np.ones((qef_batch_size, qmax_sents, cmax_sents)) * -10e8
    for i in range(qef_batch_size):
        ql, cl = query_abs_lens[i], cand_abs_lens[i]
        pad_mask[i, :ql, :cl] = 0.0
    pad_mask = Variable(torch.FloatTensor(pad_mask))
    if torch.cuda.is_available():
        pad_mask = pad_mask.cuda()
    assert (qef_batch_size == cef_batch_size)
    # (effective) batch_size x qmax_sents x cmax_sents
    # inputs are: batch_size x encoding_dim x c/qmax_sents so permute them.
    neg_pair_dists = -1 * torch.cdist(query_reps.permute(0, 2, 1), cand_reps.permute(0, 2, 1))
    if len(neg_pair_dists.size()) == 2:
        neg_pair_dists = neg_pair_dists.unsqueeze(0)
    assert (neg_pair_dists.size(1) == qmax_sents)
    assert (neg_pair_dists.size(2) == cmax_sents)
    # Add very large negative values in the pad positions which will be zero.
    neg_pair_dists = neg_pair_dists + pad_mask
    # Max across all the pairwise distances
    # - because these are negative distances the smallest distance will be picked.
    batch_dists, indices = torch.max(neg_pair_dists.view(qef_batch_size, qmax_sents * cmax_sents), dim=1)
    # At test time return similarities which can be used for ranking.
    # Negation of L2 distance isnt a similarity.
    if return_pair_sims:
        # L2 distance to similarity: https://stats.stackexchange.com/a/53070/55807
        batch_sims = batch_dists
        pair_sims = neg_pair_dists
        return batch_sims, pair_sims
    # Return a positive distance - the smallest distance is minimized even further.
    else:
        return -1 * batch_dists


def allpair_masked_dist_l2sup(query, cand):
    """
    Given a set of query and candidate reps compute l2 distance
    between all the reps and all the candidate reps and return similarity
    of (pre) aligned pair of sentences.
    :param query: namedtuple(
        embed: batch_size x encoding_dim x q_max_sents;
        abs_lens: list(int); number of sentences in every batch element.)
    :param cand: namedtuple(
        embed: batch_size x encoding_dim x q_max_sents;
        abs_lens: list(int); number of sentences in every batch element.
        align_idxs: list(int, int); alignment from query to cand)
    :return:
        batch_sims: ef_batch_size; pooled pairwise _distances_ between
            input reps. (distances are just negated similarities here)
    """
    query_reps, query_abs_lens = query.embed, query.abs_lens
    cand_reps, cand_abs_lens, cand_align_idxs = cand.embed, cand.abs_lens, cand.align_idxs
    qef_batch_size, _, qmax_sents = query_reps.size()
    cef_batch_size, encoding_dim, cmax_sents = cand_reps.size()
    # pad_mask = np.ones((qef_batch_size, qmax_sents, cmax_sents))*-10e8
    for i in range(qef_batch_size):
        ql, cl = query_abs_lens[i], cand_abs_lens[i]
        # pad_mask[i, :ql, :cl] = 0.0
        # If the index is beyond what is present in the q or c cause of truncation then clip it.
        cand_align_idxs[i][0] = min(cand_align_idxs[i][0], ql - 1)
        cand_align_idxs[i][1] = min(cand_align_idxs[i][1], cl - 1)
    # pad_mask = Variable(torch.FloatTensor(pad_mask))
    cand_align_idxs = Variable(torch.LongTensor(cand_align_idxs))
    if torch.cuda.is_available():
        # pad_mask = pad_mask.cuda()
        cand_align_idxs = cand_align_idxs.cuda()
    assert (qef_batch_size == cef_batch_size)
    # (effective) batch_size x qmax_sents x cmax_sents
    # inputs are: batch_size x encoding_dim x c/qmax_sents so permute them.
    pair_sims = -1 * torch.cdist(query_reps.permute(0, 2, 1), cand_reps.permute(0, 2, 1))
    if len(pair_sims.size()) == 2:
        pair_sims = pair_sims.unsqueeze(0)
    assert (pair_sims.size(1) == qmax_sents)
    assert (pair_sims.size(2) == cmax_sents)
    # Add very large negative values in the pad positions which will be zero.
    # pair_sims = pair_sims + pad_mask
    # Read of distances to minimize
    batch_sims = pair_sims[torch.arange(qef_batch_size), cand_align_idxs[torch.arange(qef_batch_size), 0],
    cand_align_idxs[torch.arange(qef_batch_size), 1]]
    # Return a distance instead of a similarity - so the smallest distance is minimized even further.
    return -1 * batch_sims


def allpair_masked_dist_l2sup_weighted(query, cand):
    """
    Given a set of query and candidate reps compute l2 distance
    between all the reps and all the candidate reps and return similarity
    of (pre) aligned pair of sentences.
    - Also weight the distances by the number of values in the cross-doc sim matrix.
    - This is for use in multi tasking with the OT loss.
    :param query: namedtuple(
        embed: batch_size x encoding_dim x q_max_sents;
        abs_lens: list(int); number of sentences in every batch element.)
    :param cand: namedtuple(
        embed: batch_size x encoding_dim x q_max_sents;
        abs_lens: list(int); number of sentences in every batch element.
        align_idxs: list(int, int); alignment from query to cand)
    :return:
        batch_sims: ef_batch_size; pooled pairwise _distances_ between
            input reps. (distances are just negated similarities here)
    """
    query_reps, query_abs_lens = query.embed, query.abs_lens
    cand_reps, cand_abs_lens, cand_align_idxs = cand.embed, cand.abs_lens, cand.align_idxs
    qef_batch_size, _, qmax_sents = query_reps.size()
    cef_batch_size, encoding_dim, cmax_sents = cand_reps.size()
    # pad_mask = np.ones((qef_batch_size, qmax_sents, cmax_sents))*-10e8
    cd_sizes = []
    for i in range(qef_batch_size):
        ql, cl = query_abs_lens[i], cand_abs_lens[i]
        cd_sizes.append(ql * cl)
        # pad_mask[i, :ql, :cl] = 0.0
        # If the index is beyond what is present in the q or c cause of truncation then clip it.
        cand_align_idxs[i][0] = min(cand_align_idxs[i][0], ql - 1)
        cand_align_idxs[i][1] = min(cand_align_idxs[i][1], cl - 1)
    # pad_mask = Variable(torch.FloatTensor(pad_mask))
    cand_align_idxs = Variable(torch.LongTensor(cand_align_idxs))
    cd_sizes = Variable(torch.FloatTensor(cd_sizes))
    if torch.cuda.is_available():
        # pad_mask = pad_mask.cuda()
        cand_align_idxs = cand_align_idxs.cuda()
        cd_sizes = cd_sizes.cuda()
    assert (qef_batch_size == cef_batch_size)
    # (effective) batch_size x qmax_sents x cmax_sents
    # inputs are: batch_size x encoding_dim x c/qmax_sents so permute them.
    pair_sims = -1 * torch.cdist(query_reps.permute(0, 2, 1), cand_reps.permute(0, 2, 1))
    if len(pair_sims.size()) == 2:
        pair_sims = pair_sims.unsqueeze(0)
    assert (pair_sims.size(1) == qmax_sents)
    assert (pair_sims.size(2) == cmax_sents)
    # Add very large negative values in the pad positions which will be zero.
    # pair_sims = pair_sims + pad_mask
    # Read of distances to minimize
    batch_sims = pair_sims[torch.arange(qef_batch_size), cand_align_idxs[torch.arange(qef_batch_size), 0],
    cand_align_idxs[torch.arange(qef_batch_size), 1]]
    # divide by the number of elements in the cross-doc matrix.
    batch_sims = batch_sims / cd_sizes
    # Return a distance instead of a similarity - so the smallest distance is minimized even further.
    return -1 * batch_sims