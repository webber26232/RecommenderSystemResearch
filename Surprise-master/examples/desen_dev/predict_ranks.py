import numpy as np
from lightfm._lightfm_fast import calculate_auc_from_rank, CSRMatrix


class CSRMatrixLocal:
    """
    Utility class for accessing elements
    of a CSR matrix.
    """

    def __init__(self, csr_matrix):

        self.indices = csr_matrix.indices
        self.indptr = csr_matrix.indptr
        self.data = csr_matrix.data
        self.raw = csr_matrix

        self.rows, self.cols = csr_matrix.shape
        self.nnz = len(self.data)

    def get_row_start(self, row):
        """
        Return the pointer to the start of the
        data for row.
        """

        return self.indptr[row]

    def get_row_end(self, row):
        """
        Return the pointer to the end of the
        data for row.
        """

        return self.indptr[row + 1]

    def get_cell(self, row, col):
        return self.raw[row, col]


def predict_ranks(test_interactions,
                  train_interactions,
                  predicted,
                  ranks):

    predictions_size = 0

    # Figure out the max size of the predictions
    # buffer.
    for user_id in range(test_interactions.rows):
        predictions_size = max(predictions_size,
                               test_interactions.get_row_end(user_id)
                               - test_interactions.get_row_start(user_id))

    item_ids = [None] * predictions_size
    predictions = [None] * predictions_size

    for user_id in range(test_interactions.rows):

        row_start = test_interactions.get_row_start(user_id)
        row_stop = test_interactions.get_row_end(user_id)

        if row_stop == row_start:
            # No test interactions for this user
            continue

        # Compute predictions for the items whose
        # ranks we want to know
        for i in range(row_stop - row_start):

            item_id = test_interactions.indices[row_start + i]

            item_ids[i] = item_id
            predictions[i] = predicted.get_cell(user_id, item_id)

        # Now we can zip through all the other items and compute ranks
        for item_id in range(test_interactions.cols):

            # if in_positives(item_id, user_id, train_interactions):
                # continue

            prediction = predicted.get_cell(user_id, item_id)

            for i in range(row_stop - row_start):
                if item_id != item_ids[i] and prediction >= predictions[i]:
                    ranks[row_start + i] += 1.0


def lightfm_auc_score(ranks, test_interactions, train_interactions=None,
                      preserve_rows=False, num_threads=1):

    assert np.all(ranks.data >= 0)

    auc = np.zeros(ranks.shape[0], dtype=np.float32)

    if train_interactions is not None:
        num_train_positives = (np.squeeze(np.array(train_interactions
                                                   .getnnz(axis=1))
                                          .astype(np.int32)))
    else:
        num_train_positives = np.zeros(test_interactions.shape[0],
                                       dtype=np.int32)

    # The second argument is modified in-place, but
    # here we don't care about the inconsistency
    # introduced into the ranks matrix.
    calculate_auc_from_rank(CSRMatrix(ranks),
                            num_train_positives,
                            ranks.data,
                            auc,
                            num_threads)

    if not preserve_rows:
        auc = auc[test_interactions.getnnz(axis=1) > 0]

    return auc