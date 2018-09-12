import pickle

import pandas
import numpy
import xgboost as xgb

from typos_correction.utils import rank_candidates


class CandidatesRanker:
    """
    Rank typos correcting candidates based on given features.
    XGBoost classifier used.
    """
    DEFAULT_TRAIN_ROUNDS = 4000
    DEFAULT_EARLY_STOPPING = 200
    DEFAULT_BOOST_PARAM = {'max_depth': 6,
                           'eta': 0.05,
                           'min_child_weight': 2,
                           'silent': 1,
                           'objective': 'binary:logistic',
                           'nthread': 16,
                           'subsample': 0.5,
                           'colsample_bytree': 0.5,
                           'alpha': 1,
                           'eval_metric': ['auc', 'error']}

    def __init__(self):
        self.train_rounds = self.DEFAULT_TRAIN_ROUNDS
        self.early_stopping = self.DEFAULT_EARLY_STOPPING
        self.boost_param = self.DEFAULT_BOOST_PARAM
        self.bst = None

    def set_boost_params(self, train_rounds: int = DEFAULT_TRAIN_ROUNDS,
                         early_stopping: int = DEFAULT_EARLY_STOPPING,
                         boost_param: dict = DEFAULT_BOOST_PARAM) -> None:
        self.train_rounds = train_rounds
        self.early_stopping = early_stopping
        self.boost_param = boost_param

    def _generate_tree(self) -> dict:
        tree = self.__dict__.copy()
        tree["bst"] = pickle.dumps(self.bst)
        if self.bst is not None:
            tree["bst_ntree_limit"] = self.bst.best_ntree_limit
        return tree

    def _load_tree(self, tree: dict)-> None:
        self.__dict__.update()
        self.bst = pickle.loads(tree["bst"])
        if self.bst is not None:
            self.bst.best_ntree_limit = tree["bst_ntree_limit"]

    def fit(self, typos: pandas.DataFrame, candidates: pandas.DataFrame,
            features: numpy.ndarray) -> None:
        labels = self._create_labels(typos.identifier, candidates)
        print(labels)

        edge = int(features.shape[0] * 0.9)

        dtrain = xgb.DMatrix(features[:edge, :], label=labels[:edge])
        dval = xgb.DMatrix(features[edge:, :], label=labels[edge:])

        self.boost_param['scale_pos_weight'] = (1.0 * (edge - numpy.sum(labels[:edge])) /
                                                numpy.sum(labels[:edge]))

        evallist = [(dtrain, 'train'), (dval, 'validation')]
        self.bst = xgb.train(self.boost_param, dtrain, self.train_rounds, evallist,
                             early_stopping_rounds=self.early_stopping)

    def rank(self, candidates: pandas.DataFrame, features: numpy.ndarray, n_candidates: int = 3,
             return_all: bool = True):
        dtest = xgb.DMatrix(features)
        test_proba = self.bst.predict(dtest, ntree_limit=self.bst.best_ntree_limit)

        return rank_candidates(candidates, test_proba, n_candidates, return_all)

    @staticmethod
    def _create_labels(identifiers: pandas.Series, candidates: pandas.DataFrame):
        labels = []
        for ind, row in candidates.iterrows():
            labels.append(int(row.candidate == identifiers.loc[row.id]))
        return numpy.array(labels)
