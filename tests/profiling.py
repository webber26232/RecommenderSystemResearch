import sys
PATH = sys.path[0].rsplit("/", 1)[0]
sys.path.insert(0, PATH)

from pyinstrument import Profiler

import time
import datetime
import random
import numpy as np
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from tabulate import tabulate
from surprise import PredictionImpossible
from surprise import KNNWithZScore
import heapq

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

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est = sum_ratings / sum_sim
        except ZeroDivisionError:
            est = self.means[x]
            # return mean

        details = {'actual_k': actual_k}
        return est, details


stable = 'http://surprise.readthedocs.io/en/stable/'

# set RNG
np.random.seed(0)
random.seed(0)

dataset = 'ml-1m'
data = Dataset.load_builtin(dataset)
kf = KFold(random_state=0)  # folds will be the same for all algorithms.

table = []

profiler = Profiler()
profiler.start()
# start = time.time()
out = cross_validate(KNNWithIntersectionZScore(), data, ['rmse', 'mae'], kf)
# cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
profiler.stop()

link = "KNNWithIntersectionZScore"
mean_rmse = '{:.3f}'.format(np.mean(out['test_rmse']))
mean_mae = '{:.3f}'.format(np.mean(out['test_mae']))

new_line = [link, mean_rmse, mean_mae]
print(tabulate([new_line], tablefmt="pipe"))  # print current algo perf
table.append(new_line)

header = ["100k",
          'RMSE',
          'MAE',
          ]
print(tabulate(table, header, tablefmt="pipe"))


with open("profile_results.html", "w") as f:
    f.write(profiler.output_html())

print("======================")
print(profiler.output_text(unicode=True, color=True))
