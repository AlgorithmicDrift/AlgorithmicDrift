
import numpy as np
import scipy.sparse as sp
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType
from scipy.spatial.distance import squareform, pdist

class ComputeSimilarity:

    def __init__(self, dataMatrix, topk=100, shrink=0, normalize=True):
        r"""Computes the cosine similarity of dataMatrix

        If it is computed on :math:`URM=|users| \times |items|`, pass the URM.

        If it is computed on :math:`ICM=|items| \times |features|`, pass the ICM transposed.

        Args:
            dataMatrix (scipy.sparse.csr_matrix): The sparse data matrix.
            topk (int) : The k value in KNN.
            shrink (int) :  hyper-parameter in calculate cosine distance.
            normalize (bool):   If True divide the dot product by the product of the norms.
        """

        super(ComputeSimilarity, self).__init__()

        self.shrink = shrink
        self.normalize = normalize

        self.n_rows, self.n_columns = dataMatrix.shape

        self.TopK = min(topk, self.n_columns)

        self.dataMatrix = dataMatrix.copy()

    def compute_similarity_2(self, transpose=False):
        """
        Method to compute a similarity matrix from original df_matrix
        :param transpose: If True, calculate the similarity in a transpose matrix
        :type transpose: bool, default False
        """

        # Calculate distance matrix
        if transpose:
            similarity_matrix = np.float32(squareform(pdist(self.dataMatrix.T.toarray(), "cosine")))
        else:
            similarity_matrix = np.float32(squareform(pdist(self.dataMatrix.toarray(), "cosine")))

        # Remove NaNs
        similarity_matrix[np.isnan(similarity_matrix)] = 1.0
        # transform distances in similarities. Values in matrix range from 0-1
        similarity_matrix = (similarity_matrix.max() - similarity_matrix) / similarity_matrix.max()

        # sorted similarity matrix by rows in descending order
        sorted_similarity_matrix = np.sort(-similarity_matrix, axis=1)*-1
        sorted_similarity_matrix[:, (self.TopK + 1):] = 0.

        sorted_similarity_matrix = sp.csr_matrix(sorted_similarity_matrix).tocsc()

        return sorted_similarity_matrix

    def compute_similarity(self, method, users=None, block_size=1001):
        r"""Compute the similarity for the given dataset

        Args:
            method (str) : Caculate the similarity of users if method is 'user', otherwise, calculate the similarity of items.
            block_size (int): divide matrix to :math:`n\_rows \div block\_size` to calculate cosine_distance if method is 'user',
                 otherwise, divide matrix to :math:`n\_columns \div block\_size`.

        Returns:

            list: The similar nodes, if method is 'user', the shape is [number of users, neigh_num],
            else, the shape is [number of items, neigh_num].
            scipy.sparse.csr_matrix: sparse matrix W, if method is 'user', the shape is [self.n_rows, self.n_rows],
            else, the shape is [self.n_columns, self.n_columns].
        """

        values = []
        rows = []
        cols = []
        neigh = []

        self.dataMatrix = self.dataMatrix.astype(np.float32)

        # Compute sum of squared values to be used in normalization
        if method == 'user':
            sumOfSquared = np.array(
                self.dataMatrix.power(2).sum(
                    axis=1)).ravel()
            if users is None:
                end_local = self.n_rows
            else:
                end_local = max(users) + 1

        elif method == 'item':
            sumOfSquared = np.array(
                self.dataMatrix.power(2).sum(
                    axis=0)).ravel()
            end_local = self.n_columns
        else:
            raise NotImplementedError(
                "Make sure 'method' in ['user', 'item']!")
        sumOfSquared = np.sqrt(sumOfSquared)

        if users is None:
            start_block = 0
        else:
            start_block = min(users)

        # Compute all similarities using vectorization
        while start_block < end_local:

            end_block = min(start_block + block_size, end_local)
            this_block_size = end_block - start_block

            # All data points for a given user or item
            if method == 'user':
                data = self.dataMatrix[start_block:end_block, :]
            else:
                data = self.dataMatrix[:, start_block:end_block]
            data = data.toarray()

            # Compute similarities
            if method == 'user':
                this_block_weights = self.dataMatrix.dot(data.T)
            else:
                this_block_weights = self.dataMatrix.T.dot(data)

            # for i in range(len(this_block_weights)):
            #     if i > 0 and len(np.nonzero(this_block_weights[i])[0]) == 0:
            #         print(i, "has all zero")

            for index_in_block in range(this_block_size):
                this_line_weights = this_block_weights[:, index_in_block]

                Index = index_in_block + start_block
                #print("Index", Index)
                this_line_weights[Index] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:
                    denominator = sumOfSquared[Index] * \
                        sumOfSquared + self.shrink + 1e-6
                    this_line_weights = np.multiply(
                        this_line_weights, 1 / denominator)

                elif self.shrink != 0:
                    this_line_weights = this_line_weights / self.shrink

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of users or items
                # - Partition the data to extract the set of relevant users or items
                # - Sort only the relevant users or items
                # - Get the original index
                relevant_partition = (-this_line_weights).argpartition(
                    self.TopK - 1)[0:self.TopK]
                relevant_partition_sorting = np.argsort(
                    -this_line_weights[relevant_partition])
                top_k_idx = relevant_partition[relevant_partition_sorting]
                neigh.append(top_k_idx)

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_line_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_line_weights[top_k_idx][notZerosMask])

                if method == 'user':
                    rows.extend(np.ones(numNotZeros) * Index)
                    cols.extend(top_k_idx[notZerosMask])
                else:
                    rows.extend(top_k_idx[notZerosMask])
                    cols.extend(np.ones(numNotZeros) * Index)

            start_block += block_size

        # End while
        if method == 'user':
            W_sparse = sp.csr_matrix(
                (values, (rows, cols)), shape=(
                    self.n_rows, self.n_rows), dtype=np.float32)
        else:
            W_sparse = sp.csr_matrix(
                (values, (rows, cols)), shape=(
                    self.n_columns, self.n_columns), dtype=np.float32)
        return neigh, W_sparse.tocsc()


class UserKNN(GeneralRecommender):
    r"""UserKNN is a basic model that compute item similarity with the interaction matrix.

    """
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset, users):
        super(UserKNN, self).__init__(config, dataset)

        # load parameters info
        self.k = config['k']

        print("k", self.k)

        self.shrink = config['shrink'] if 'shrink' in config else 0.0
        self.method = config['method'] if 'method' in config else 'item'

        self.interaction_matrix = dataset.inter_matrix(
            form='csr').astype(np.float32)
        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]
        _, self.w = ComputeSimilarity(
            self.interaction_matrix, topk=self.k, shrink=self.shrink).\
            compute_similarity(self.method)#, users=users)

        if self.method == 'user':
            # self.pred_mat = self.interaction_matrix.T.dot(self.w).tolil().T
            self.pred_mat = self.w.dot(self.interaction_matrix).tolil()

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ['w', 'pred_mat']

    def forward(self, user, item):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user = user.cpu().numpy().astype(int)
        item = item.cpu().numpy().astype(int)

        result = []

        for index in range(len(user)):
            uid = user[index]
            iid = item[index]
            score = self.pred_mat[uid, iid]
            result.append(score)
        result = torch.from_numpy(np.array(result)).to(self.device)
        return result

    def predict_for_graphs(self, interaction, user_count_start=0):
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]

        new_interaction_matrix = self.interaction_matrix.copy()

        for i in range(len(users)):
            new_interaction_matrix[users[i], items[i]] = 1.

        _, self.w = ComputeSimilarity(
            new_interaction_matrix, topk=self.k, shrink=self.shrink).compute_similarity(
            self.method)#, users=users.numpy())

        # new_pred_mat = new_interaction_matrix.T.dot(self.w).tolil().T
        new_pred_mat = self.w.dot(new_interaction_matrix).tolil()

        results = new_pred_mat[users, :].toarray().flatten()

        results = torch.from_numpy(np.array(results)).to(self.device)

        return results

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user = user.cpu().numpy()

        score = self.pred_mat[user, :].toarray().flatten()
        result = torch.from_numpy(score).to(self.device)

        return result
