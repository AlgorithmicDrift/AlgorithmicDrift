import torch


class Random:

    def __init__(self, num_users, num_items):

        self.num_users = num_users
        self.num_items = num_items

    def predict_for_graphs(self, interaction=None, user_count_start=None):

        uniform_scores = torch.ones(self.num_users * (self.num_items + 1))
        return uniform_scores

    def full_sort_predict(self, interaction=None):
        return self.predict_for_graphs(interaction=interaction)


