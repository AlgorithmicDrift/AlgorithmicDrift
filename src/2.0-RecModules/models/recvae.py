import sys
import os

# add 2.0-RecModules folder in sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType



def swish(x):
    r"""Swish activation function:

    .. math::
        \text{Swish}(x) = \frac{x}{1 + \exp(-x)}
    """
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):

    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights):
        super(CompositePrior, self).__init__()

        self.mixture_weights = mixture_weights

        self.mu_prior = nn.Parameter(
            torch.Tensor(
                1,
                latent_dim),
            requires_grad=False)
        self.mu_prior.data.fill_(0)

        self.logvar_prior = nn.Parameter(
            torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)

        self.logvar_uniform_prior = nn.Parameter(
            torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)

        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)

        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)

        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [
            g.add(
                np.log(w)) for g, w in zip(
                gaussians, self.mixture_weights)]

        density_per_gaussian = torch.stack(gaussians, dim=-1)

        return torch.logsumexp(density_per_gaussian, dim=-1)


class Encoder(nn.Module):

    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dropout_prob):
        x = F.normalize(x)
        x = F.dropout(x, dropout_prob, training=self.training)

        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)


class RecVAE(GeneralRecommender):
    r"""Collaborative Denoising Auto-Encoder (RecVAE) is a recommendation model
    for top-N recommendation with implicit feedback.

    We implement the model following the original author
    """
    input_type = InputType.PAIRWISE

    def __init__(
            self,
            config,
            dataset,
            mean_slant_users,
            items_labels,
            lambda_factor,
            num_users,
            num_items,
            reweighting_mitigation=False):
        super(RecVAE, self).__init__(config, dataset)

        self.hidden_dim = config["hidden_dimension"]
        self.latent_dim = config['latent_dimension']
        self.dropout_prob = config['dropout_prob']
        self.beta = config['beta']
        self.mixture_weights = config['mixture_weights']
        self.gamma = config['gamma']

        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)

        self.encoder = Encoder(self.hidden_dim, self.latent_dim, self.n_items)
        self.prior = CompositePrior(
            self.hidden_dim,
            self.latent_dim,
            self.n_items,
            self.mixture_weights)
        self.decoder = nn.Linear(self.latent_dim, self.n_items)

        # parameters initialization
        self.apply(xavier_normal_initialization)

        # reweighting_mitigation parameters

        self.mean_slant_users = mean_slant_users
        self.items_labels = items_labels
        self.reweighting_mitigation = reweighting_mitigation
        self.lambda_factor = np.float64(lambda_factor.split("_")[1])
        self.num_users = num_users
        self.num_items = num_items

        self.build_count_weight()


    def get_rating_matrix(self, user):
        r"""Get a batch of user's feature with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The user's feature of a batch of user, shape: [batch_size, n_items]
        """
        # Following lines construct tensor of shape [B,n_items] using the
        # tensor of shape [B,H]
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(
            user.shape[0]).to(
            self.device).repeat_interleave(
            self.history_item_id.shape[1],
            dim=0)

        rating_matrix = torch.zeros(1).to(
            self.device).repeat(
            user.shape[0], self.n_items)
        rating_matrix.index_put_(
            (row_indices, col_indices), self.history_item_value[user].flatten())
        return rating_matrix

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + epsilon * std
        else:
            return mu

    def forward(self, rating_matrix, dropout_prob):
        mu, logvar = self.encoder(rating_matrix, dropout_prob=dropout_prob)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        return x_pred, mu, logvar, z

    def build_count_weight(self):
        self.count_weight = np.ones((self.num_users + 1, self.num_items + 1))

        if self.reweighting_mitigation:
            alpha_users = 1 - self.mean_slant_users

            for i in range(self.num_users + 1):
                for j in range(self.num_items + 1):
                    self.count_weight[i][j] = (1 - (alpha_users[i] * self.items_labels[j])) * self.lambda_factor



    def calculate_loss(self, interaction, encoder_flag):
        user = interaction[self.USER_ID]
        rating_matrix = self.get_rating_matrix(user)

        if encoder_flag:
            dropout_prob = self.dropout_prob
        else:
            dropout_prob = 0
        x_pred, mu, logvar, z = self.forward(rating_matrix, dropout_prob)

        if self.gamma:
            norm = rating_matrix.sum(dim=-1)
            kl_weight = self.gamma * norm
        else:
            kl_weight = self.beta

        harmful_weights = torch.tensor(
            self.count_weight).type('torch.FloatTensor').to('cuda')

        mll = ((F.log_softmax(x_pred, dim=-1) *
                rating_matrix) * harmful_weights).sum(dim=-1).mean()

        kld = (log_norm_pdf(z, mu, logvar) - self.prior(rating_matrix, z)
               ).sum(dim=-1).mul(kl_weight).mean()

        negative_elbo = -(mll - kld)

        # print("Encoder flag:", encoder_flag, "Mll:", mll.item(), "Kld:", kld.item())

        return negative_elbo

    def predict(self, interaction):
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]

        rating_matrix = self.get_rating_matrix(users)

        scores, _, _, _ = self.forward(rating_matrix, self.dropout_prob)

        return scores[[torch.arange(len(items)).to(self.device), items]]

    @torch.inference_mode()
    def predict_for_graphs(self, interaction, user_count_start):
        users = interaction[self.USER_ID].to(self.device)
        items = interaction[self.ITEM_ID].to(self.device)
        values = interaction['item_value'].to(self.device)

        users = users.detach().cpu().numpy()

        rating_matrix = torch.zeros(1).to(
            self.device).repeat(
            users.shape[0],
            self.n_items)  # self.get_rating_matrix(users)

        users = users - user_count_start - 1

        if self.device == 'cuda':
            col_indices = items[users].flatten().cuda()
        else:
            col_indices = items[users].flatten()
        row_indices = torch.arange(
            users.shape[0]).to(
            self.device).repeat_interleave(
            items.shape[1],
            dim=0)

        if self.device == 'cuda':
            rating_matrix.index_put_(
                (row_indices, col_indices), values.flatten().cuda())
        else:
            rating_matrix.index_put_(
                (row_indices, col_indices), values.flatten())

        scores, _, _, z = self.forward(rating_matrix, self.dropout_prob)

        # scores[[torch.arange(len(items)).to(self.device), items]]
        return scores.view(-1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _, _ = self.forward(rating_matrix, self.dropout_prob)

        return scores.view(-1)

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(
            deepcopy(self.encoder.state_dict()))
