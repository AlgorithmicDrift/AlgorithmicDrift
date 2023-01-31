import copy

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType

import numpy as np
import scipy.sparse as sp
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class BPRLoss(nn.Module):

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class LightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self,
                 config,
                 dataset,
                 mean_slant_users,
                 items_labels,
                 lambda_factor,
                 num_users,
                 num_items,
                 reweighting_mitigation=False):
        super(LightGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        # load parameters info
        # int type:the embedding size of lightGCN
        self.latent_dim = config['embedding_size']
        # int type:the layer num of lightGCN
        self.n_layers = config['n_layers']
        # float32 type: the weight decay for l2 normalization
        self.reg_weight = config['reg_weight']
        self.require_pow = config['require_pow']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.mean_slant_users = mean_slant_users
        self.items_labels = items_labels
        self.reweighting_mitigation = reweighting_mitigation
        self.lambda_factor = np.float64(lambda_factor.split("_")[1])
        self.num_users = num_users
        self.num_items = num_items

        self.build_count_weight()

    def get_norm_adj_mat(self, new_interaction_matrix=None):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix

        int_matrix = self.interaction_matrix
        if new_interaction_matrix is not None:
            int_matrix = new_interaction_matrix

        A = sp.dok_matrix(
            (self.n_users +
             self.n_items,
             self.n_users +
             self.n_items),
            dtype=np.float32)
        inter_M = int_matrix
        inter_M_t = int_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(
            dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(
                self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def forward_gen_graph(self, users, items, values, user_count_start):
        all_embeddings = self.get_ego_embeddings().cuda()
        embeddings_list = [all_embeddings]

        new_interaction_matrix = torch.zeros(1).to(self.device).repeat(
            users.shape[0] + 1, self.n_items)  # self.get_rating_matrix(users)

        reindexed_users = users - user_count_start - 1

        col_indices = items[reindexed_users].flatten()
        row_indices = torch.arange(
            users.shape[0]).to(
            self.device).repeat_interleave(
            items.shape[1],
            dim=0)

        new_interaction_matrix.index_put_(
            (row_indices + 1, col_indices), values.flatten().cuda())

        new_interaction_matrix = sp.coo_matrix(
            new_interaction_matrix.cpu().numpy())

        new_norm_adj_matrix = self.get_norm_adj_mat(
            new_interaction_matrix).cuda()

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(
                new_norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings

    def build_count_weight(self):
        self.count_weight = np.ones((self.num_users + 1, self.num_items + 1))

        if self.reweighting_mitigation:
            alpha_users = 1 - self.mean_slant_users

            for i in range(self.num_users + 1):
                for j in range(self.num_items + 1):
                    self.count_weight[i][j] = (1 - (alpha_users[i] * self.items_labels[j])) * self.lambda_factor

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]


        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        harmful_weights = torch.tensor(
            self.count_weight).type('torch.FloatTensor').to(self.device)

        # print(harmful_weights[user][:, pos_item].shape)
        # print(pos_scores.shape)
        # pos_scores *= harmful_weights[user, pos_item]
        neg_scores *= harmful_weights[user, neg_item]

        # [0, 1]
        # [5, 3]
        #
        # [0 0 1 0.4 0.5 0]
        # [0 0 1 0.4 0.5 0]
        #
        # (0.2) --> user 0, item 5
        # (0.7) --> user 1, item 3

        # print(user_all_embeddings.shape)
        # # print(len(neg_item))
        # print("\n")

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate EMB Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow)

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict_for_graphs(self, interaction, user_count_start):
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        values = interaction['item_value']

        users = users.detach().cpu().numpy()

        user_all_embeddings, item_all_embeddings = self.forward_gen_graph(
            users, items, values, user_count_start)

        u_embeddings = user_all_embeddings[users]
        # i_embeddings = item_all_embeddings[items]
        scores = torch.matmul(
            u_embeddings,
            item_all_embeddings.transpose(
                0,
                1))

        return scores.view(-1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(
            u_embeddings,
            self.restore_item_e.transpose(
                0,
                1))

        return scores.view(-1)
