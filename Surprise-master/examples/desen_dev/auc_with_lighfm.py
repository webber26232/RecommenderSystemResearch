from surprise import SVD
from surprise import Dataset
from surprise.model_selection import train_test_split
from scipy.sparse import csr_matrix
from predict_ranks import *


data = Dataset.load_builtin('ml-100k')


train_set, test_set = train_test_split(data, test_size=.25)

test_set_ar = np.array(test_set, dtype='int32')
test_inter = csr_matrix((test_set_ar[:, 2], (test_set_ar[:, 0], test_set_ar[:, 1])))

rows, cols, vals = [], [], []
for key, values in train_set.ur.items():
    for value in values:
        rows.append(key)
        cols.append(value[0])
        vals.append(int(value[1]))
train_inter = csr_matrix((vals, (rows, cols)))

algo = SVD()


algo.fit(train_set)
predictions = algo.test(test_set)
predictions = [(prediction.uid, prediction.iid, prediction.r_ui) for prediction in predictions]
predictions = np.array(predictions, dtype='int32')
predictions = csr_matrix((predictions[:, 2], (predictions[:, 0], predictions[:, 1])))

test_ranks = csr_matrix((np.zeros_like(test_inter.data),
                         test_inter.indices,
                         test_inter.indptr),
                        shape=test_inter.shape)

test_ranks = test_ranks.astype(np.float32)

predict_ranks(CSRMatrixLocal(test_inter), CSRMatrixLocal(train_inter), CSRMatrixLocal(predictions), test_ranks.data)


scores = lightfm_auc_score(test_ranks, test_inter)
