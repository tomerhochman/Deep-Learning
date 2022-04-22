class KNN():

    def __init__(self, X_train, Y_train, k):
        self.X_train = X_train
        self.Y_train = Y_train
        self.k = k

    @staticmethod  # static method - a method which does not depend on an
    # instance and acts exactly like any standalone method outside of a classe
    # hence self is not passed as an argumetn
    def majority_vote(y_labels):
        votes, freq = np.unique(y_labels, return_counts=True)
        max_val = votes[np.argmax(freq)]
        return max_val

    def _k_neighbours(self, point):
        distances = np.linalg.norm(self.X_train - point, axis=1)
        new_order = distances.argsort()
        new_labels = self.Y_train[new_order]
        return new_labels[:self.k]

    def _predict(self, point):
        labels = self._k_neighbours(point)
        return KNN.majority_vote(labels)

    def train(self):
        # no training in this model, we will se later why this was done
        pass

    def test(self, X_test):
        results = []
        for test_row in X_test:
            results.append(self._predict(test_row))
        return np.array(results)


def success_percentage(Y_test, Y_classified):
    assert len(Y_test) == len(Y_classified)
    return (Y_test == Y_classified).sum() / len(Y_test)


