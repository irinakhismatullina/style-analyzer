import re

import pandas

from lookout.style.typos_checker.typos_correction.corrector import TyposCorrector
from lookout.style.typos_checker.typos_correction.utils import read_vocabulary, suggestions_to_df


class CommentsCheckingModel(AnalyzerModel):
    """
    A modelforge model to store Rules instances.
    It is required to store all the Rules for different programming languages in a single model,
    named after each language.
    Note that Rules must be fitted and Rules.base_model is not saved.
    """
    NAME = "code-format"
    VENDOR = "source{d}"

    corrector = TyposCorrector(threads_number=4)
    corrector.load("corrector.asdf")
    vocabulary = read_vocabulary("data/common_voc.csv")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def correct_comments(self, comments:list) -> pandas.DataFrame:
        typos = []
        for comment in comments:
            split = re.compile('\w+').findall(comment)
            for i in range(len(split)):
                if split[i] in self.vocabulary:
                    continue
                left = max(0, i - 2)
                right = min(len(split), i + 3)
                typos.append([split[i], split[left:i], split[i + 1:right]])
        typos = pandas.DataFrame(typos, columns=["typo", "before", "after"])
        suggestions = self.corrector.suggest(typos, n_candidates=3, return_all=False)
        return suggestions_to_df(typos, suggestions)

    def dump(self) -> str:
        return "Typos correcting model with vocabulary size %d" % len(self.vocabulary)

    def _generate_tree(self) -> dict:
        return {}

    def _load_tree(self, tree: dict) -> None:
        return {}
