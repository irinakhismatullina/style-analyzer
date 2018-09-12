from multiprocessing import Pool
from typing import NamedTuple, Tuple, Union
from itertools import chain

from modelforge import split_strings, merge_strings
import pandas
import numpy
import keras
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
from gensim.models import FastText
from tqdm import tqdm

from pysymspell.symspell import SymSpell, EditDistance
from typos_correction.utils import (read_frequencies, read_vocabulary, add_context_info, collect_embeddings)
from typos_correction.nn_prediction import get_predictions


TypoInfo = NamedTuple("TypoInfo", [("index", int),
                                   ("typo", str),
                                   ("typo_vec", numpy.ndarray),
                                   ("predict_vec", numpy.ndarray),
                                   ("before_vec", numpy.ndarray),
                                   ("after_vec", numpy.ndarray),
                                   ("context_vec", numpy.ndarray),
                                   ("cos_before", float),
                                   ("cos_after", float),
                                   ("cos_context", float)])


class CorrectionsFinder:
    """
    Look for candidates for correction of typos and generates features
    for them. Candidates are generated in three ways:
    1. Closest on cosine distance to the given correction embedding prediction
    2. Closest on cosine distance to the compound vector of token context
    3. Closest on the edit distance and most frequent tokens from vocabulary
    """
    DEFAULT_RADIUS = 4
    DEFAULT_MAX_DISTANCE = 3
    DEFAULT_NEIGHBORS_NUMBER = 20
    DEFAULT_TAKEN_FOR_DISTANCE = 10

    def construct(self, vocabulary_file: str, frequencies_file: str, fasttext,
                  neighbors_number: int = DEFAULT_NEIGHBORS_NUMBER,
                  taken_for_distance: int = DEFAULT_TAKEN_FOR_DISTANCE,
                  max_distance: int = DEFAULT_MAX_DISTANCE,
                  radius: int = DEFAULT_RADIUS) -> None:
        self.checker = SymSpell(max_dictionary_edit_distance=max_distance)
        self.checker.load_dictionary(vocabulary_file)
        
        self.fasttext = fasttext

        self.neighbors_number = neighbors_number
        self.taken_for_distance = taken_for_distance
        self.max_distance = max_distance
        self.radius = radius

        self.tokens = read_vocabulary(vocabulary_file)
        self.frequencies = read_frequencies(frequencies_file)

    def _generate_tree(self) -> dict:
        tree = self.__dict__.copy()
        tree["checker"] = self.checker.__dict__.copy()
        tree["tokens"] = merge_strings(self.tokens)
        tree.pop("fasttext", None)
        return tree

    def _load_tree(self, tree: dict) -> None:
        self.__dict__.update(tree)

        self.tokens = split_strings(self.tokens)
        self.checker = SymSpell(max_dictionary_edit_distance=self.max_distance)
        checker_tree = tree["checker"]
        checker_tree["_deletes"] = {int(h): deletes
                                    for h, deletes in checker_tree["_deletes"].items()}
        self.checker.__dict__.update(checker_tree)

    def lookup_corrections(self, typos: list, threads_number: int,
                           start_pool_size: int = 64) -> list:
        if len(typos) > start_pool_size:
            with Pool(min(threads_number, len(typos))) as pool:
                candidates = list(tqdm(pool.imap(self._lookup_corrections_for_token, typos,
                                                 chunksize=min(256, 1 + len(typos) //
                                                               threads_number)),
                                       total=len(typos)))
        else:
            candidates = list(map(self._lookup_corrections_for_token, typos))

        return candidates

    def _lookup_corrections_for_token(self, typo_info: TypoInfo):
        candidates = []
        candidate_tokens = self._get_candidate_tokens(typo_info)

        dist_calc = EditDistance(typo_info.typo, "damerau")
        for candidate in set(candidate_tokens):
            candidate_vec = typo_info.typo_vec
            dist = 0
            if candidate != typo_info.typo:
                candidate_vec = self._vec(candidate)
                dist = dist_calc.damerau_levenshtein_distance(candidate, self.radius)

            if dist < 0:
                continue
            candidates.append(self._generate_features(typo_info, dist, candidate, candidate_vec))

        return candidates

    def _get_candidate_tokens(self, typo_info: TypoInfo) -> set:
        last_dist = -1
        taken_for_dist = 0

        candidate_tokens = []

        for suggestion in self.checker.lookup(typo_info.typo, 2, self.max_distance):
            if suggestion.distance != last_dist:
                taken_for_dist = 0
                last_dist = suggestion.distance
            if taken_for_dist < self.taken_for_distance:
                candidate = suggestion.term
                candidate_tokens.append(candidate)
                taken_for_dist += 1

        if last_dist == -1:
            candidate_tokens.append(typo_info.typo)

        predict_neighbors = self._closest(typo_info.predict_vec, self.neighbors_number)
        candidate_tokens.extend(predict_neighbors)

        if numpy.linalg.norm(typo_info.context_vec) != 0:
            context_neighbors = self._closest(typo_info.context_vec, self.neighbors_number)
            candidate_tokens.extend(context_neighbors)

        return set(candidate_tokens)

    def _generate_features(self, typo_info: TypoInfo, dist: int,
                           candidate: str, candidate_vec: numpy.ndarray):
        """
        Features for correction candidate.
        :param typo_info: instance of TypoInfo class
        :param dist: edit distance from candidate to typo
        :param candidate: candidate token
        :param candidate_vec: candidate token embedding
        :return: index, typo and candidate tokens, frequencies info,
                 cosine distances between embeggings and contexts,
                 edit distance between the tokens, embeddings of
                 the tokens and contexts
        """
        return ([typo_info.index, typo_info.typo, candidate,
                self._freq(typo_info.typo),
                self._freq(candidate),
                self._freq_relation(typo_info.typo, candidate),
                typo_info.cos_before, typo_info.cos_after, typo_info.cos_context,
                self._cos(candidate_vec, typo_info.before_vec),
                self._cos(candidate_vec, typo_info.after_vec),
                self._cos(candidate_vec, typo_info.context_vec),
                self._cos(typo_info.typo_vec, candidate_vec),
                dist] +
                list(typo_info.before_vec) + list(typo_info.after_vec) +
                list(typo_info.typo_vec) + list(candidate_vec) + list(typo_info.context_vec))

    def _vec(self, token: str):
        return self.fasttext.wv[token]

    def _freq(self, token: str):
        return self.frequencies.get(token, 0)
    
    def _cos(self, first, second):
        if numpy.linalg.norm(first) == 0 or numpy.linalg.norm(second) == 0:
            return 1
        return cosine(first, second)

    def _closest(self, item: Union[numpy.ndarray, str], quantity: int):
        return [token for token, _ in self.fasttext.wv.most_similar([item], topn=quantity)]

    def _freq_relation(self, first_token: str, second_token: str):
        return -numpy.log((1.0 * self._freq(first_token) + 1e-5) /
                          (1.0 * self._freq(second_token) + 1e-5))

    def __str__(self):
        return ("Vocabulary_size %d. \n"
                "Neighbors number %d. \n"
                "Maximum distance for search %d. \n"
                "Maximum distance allowed %d. \n"
                "Taken for distance %d.") % (len(self.tokens), self.neighbors_number,
                                             self.max_distance, self.radius,
                                             self.taken_for_distance)


class CandidatesGenerator:
    """
    Runs corrections embedding predictions and candidates finder and
    feature extractor.
    """
    # True to enable neural network candidates generation
    # Requires Keras library installed
    ENABLE_NN_PREDICT = False

    def __init__(self, fasttext, nn_file=None):
        super().__init__()
        self.fasttext = fasttext

        if self.ENABLE_NN_PREDICT:
            assert nn_file is not None, "NN prediction enabled, " \
                                        "but nn pretrained model file not specified"
            self.nn = keras.models.load_model(nn_file)

    def generate_candidates(self, data: pandas.DataFrame, finder: CorrectionsFinder, threads_number: int,
                            save_candidates_file=None, start_pool_size=64) -> pandas.DataFrame:
        """
        Generates candidates for typos inside data
        :param data: pandas.DataFrame, containing column "typo"
        :param finder: instance of CorrectionsFinder class
        :param save_candidates_file: file to save candidates to
        :param start_pool_size: length of data, starting from which
               multiprocessing is desired
        :return: pandas.DataFrame containing candidates for corrections
                 and features for their ranking for each typo
        """
        data = add_context_info(data)

        vecs = collect_embeddings(self.fasttext, data.typo)
        predicts = vecs

        if self.ENABLE_NN_PREDICT:
            predicts = get_predictions(self.fasttext, self.nn, data.typo)

        typos = [self._generate_typo_info(vecs[i], predicts[i], data.loc[data.index[i]])
                 for i in range(len(data))]

        candidates = finder.lookup_corrections(typos, threads_number, start_pool_size)

        candidates = pandas.DataFrame(list(chain.from_iterable((candidates))))
        candidates.columns = (["id", "typo", "candidate"] +
                              list(range(len(candidates.columns) - 3)))
        candidates["id"] = candidates["id"].astype(data.index.dtype)\

        if save_candidates_file is not None:
            candidates.to_pickle(save_candidates_file)

        return candidates

    def _generate_typo_info(self, vec: numpy.ndarray, predict: numpy.ndarray,
                            typo_row: pandas.Series) -> TypoInfo:
        before_vec = self._compound_vec(typo_row.before)
        after_vec = self._compound_vec(typo_row.after)
        context_vec = self._compound_vec(typo_row.before + typo_row.after)
        return TypoInfo(typo_row.name, typo_row.typo, vec, predict,
                        before_vec, after_vec, context_vec,
                        self._cos(vec, before_vec), self._cos(vec, after_vec), self._cos(vec, context_vec))

    def _vec(self, token):
        return self.fasttext.wv[token]
    
    def _cos(self, first, second):
        if numpy.linalg.norm(first) == 0 or numpy.linalg.norm(second) == 0:
            return 1
        return cosine(first, second)

    def _compound_vec(self, split):
        compound_vec = numpy.zeros(self._vec("a").shape)
        if len(split) == 0:
            return compound_vec
        else:
            for token in split:
                compound_vec += self._vec(token)
        return compound_vec


def get_candidates_features(candidates: pandas.DataFrame) -> numpy.ndarray:
    return candidates.drop(columns=["id", "typo", "candidate"]).as_matrix().astype(float)


def get_candidates_tokens(candidates: pandas.DataFrame) -> pandas.DataFrame:
    return candidates[["id", "typo", "candidate"]]
