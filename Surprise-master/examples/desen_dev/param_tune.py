from surprise.prediction_algorithms.knns import KNNWithZScore
from surprise.model_selection import GridSearchCV, KFold

import numpy as np
from surprise import PredictionImpossible
import heapq

from surprise import Dataset, Reader
import pandas as pd


class KNNWithIntersectionZScore(KNNWithZScore):
    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        x_r = np.full(self.n_y, np.nan, dtype=np.float64)
        neighbor_r = x_r.copy()
        x_y, r = zip(*self.xr[x])
        x_r[list(x_y)] = r
        x_nan_mask = np.isnan(x_r)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])


        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            if sim > 0:
                nb_y, nb_r = zip(*self.xr[nb])
                neighbor_r[list(nb_y)] = nb_r
                overlapping_ind = np.flatnonzero(~(x_nan_mask | np.isnan(neighbor_r)))
                x_overlapping_r = x_r[overlapping_ind]
                neighbor_overlapping_r = neighbor_r[overlapping_ind]
                # reset
                neighbor_r.fill(np.nan)

                x_nb_mean = x_overlapping_r.mean()
                x_nb_std = ((x_overlapping_r - x_nb_mean) ** 2).mean() ** 0.5
                if x_nb_std == 0:
                    x_nb_std = self.sigmas[x]

                nb_x_mean = neighbor_overlapping_r.mean()
                nb_x_std = ((neighbor_overlapping_r - nb_x_mean) ** 2).mean() ** 0.5
                if nb_x_std == 0:
                    nb_x_std = self.sigmas[nb]

                sum_sim += sim
                sum_ratings += sim * ((r - nb_x_mean) / nb_x_std * x_nb_std + x_nb_mean)

                actual_k += 1

        if (actual_k < self.min_k) or (sum_sim == 0):
            est = self.means[x]
        else:
            est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


def get_sample_user(user_column):
    user_counts = user_column.value_counts()
    return user_counts.index.to_series().sample(100, weights=user_counts.values)


SIMILARITY = 'pearson'


if __name__ == '__main__':

    netflix = pd.read_csv('C:/Users/YuJun/Documents/local_python/recommender_system/data/netflix_filtered.csv')

    # netflix_sub = netflix[netflix['userID'].isin(get_sample_user(netflix['userID']))]

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(netflix, reader)

    k_fold = KFold(5, shuffle=True, random_state=0)
    for knn_class in [KNNWithIntersectionZScore]:
        grid = GridSearchCV(knn_class,
                            {'k': range(10, 201, 10),
                             'min_k': range(1, 15, 3),
                             'sim_options': {'min_support': range(1, 15, 3),
                                             'name': [SIMILARITY]}},
                            cv=k_fold, n_jobs=-1, joblib_verbose=0)

        grid.fit(data)
        df = pd.DataFrame(grid.cv_results)
        df.to_csv('C:/Users/YuJun/Documents/local_python/recommender_system/{}_item.csv'.format(knn_class.__name__), index=False)
        print('file saved')
