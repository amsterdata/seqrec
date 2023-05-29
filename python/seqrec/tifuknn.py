import sys
import numpy as np
from scipy.sparse import csr_matrix
import math
from sklearn.neighbors import NearestNeighbors
import datetime
from seqrec import Index, tifu_reps

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class TIFUKNN:
    def __init__(self, train_baskets, k=300, kplus=0, m=7, rb=1, rg=0.6, alpha=0.7):

        self.k = k
        self.kplus = kplus
        self.m = m
        self.rb = rb
        self.rg = rg
        self.alpha = alpha

        self.num_users = train_baskets['user_id'].max() + 1
        self.num_items = train_baskets['item_id'].max() + 1
        self.user_reps = []
        self.nn_indices = []

        self._compute_user_representations(train_baskets)

    def _compute_user_representations(self, train_baskets):

        start = datetime.datetime.now()

        sorted_baskets = train_baskets.sort_values(['user_id', 'order_number'])
        sorted_baskets = sorted_baskets[['user_id', 'basket_id']].drop_duplicates()
        user_baskets_df = sorted_baskets.groupby('user_id')['basket_id'].apply(list).reset_index()
        user_baskets_dict = dict(zip(user_baskets_df['user_id'], user_baskets_df['basket_id']))

        basket_items_df = train_baskets[['basket_id', 'item_id']].drop_duplicates().groupby('basket_id')['item_id'] \
            .apply(list).reset_index()
        basket_items_dict = dict(zip(basket_items_df['basket_id'], basket_items_df['item_id']))

        end = datetime.datetime.now()
        print('Setup', (end-start).total_seconds() * 1000)

        start = datetime.datetime.now()
        user_reps = tifu_reps(basket_items_dict, user_baskets_dict, self.num_items, self.m, self.rb, self.rg)
        end = datetime.datetime.now()
        print('Rust reps', (end-start).total_seconds() * 1000)

        start = datetime.datetime.now()
        self.user_reps = np.array(user_reps)
        representations = csr_matrix(self.user_reps)

        num_rows, num_cols = representations.shape


        self.caboose = Index(num_rows, num_cols, representations.indptr, representations.indices,
                             representations.data, self.k + self.kplus)


        end = datetime.datetime.now()
        print('Neighbors', (end-start).total_seconds() * 1000)


    def retrieve_for(self, user):
        how_many = self.k + self.kplus
        all = sorted(self.caboose.topk(user), key = lambda x: x[1], reverse=True)
        return all[:how_many]



    def representation(self, user):
        return self.user_reps[user]

    def predict(self, user, neighbors, how_many):
        user_rep = self.user_reps[user]

        nn_rep = np.zeros(self.num_items)
        for neighbor in neighbors:
            nn_rep += self.user_reps[neighbor]

        nn_rep /= len(neighbors)

        final_rep = (user_rep * self.alpha + (1 - self.alpha) * nn_rep).tolist()
        top_items = sorted(range(len(final_rep)), key=lambda pos: final_rep[pos], reverse=True)

        return top_items[:how_many]
