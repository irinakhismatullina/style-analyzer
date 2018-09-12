import re
from collections import defaultdict

import pandas

from lookout.style.typos_checker.typos_correction.corrector import TyposCorrector
from lookout.style.typos_checker.typos_correction.utils import read_vocabulary, suggestions_to_df


class CommentsCheckingModel(AnalyzerModel):
    """
    Model correcting typos inside comments
    """
    NAME = "comments-typos"
    VENDOR = "source{d}"

    corrector = TyposCorrector(threads_number=4)
    corrector.load("lookout/style/typos_checker/corrector.asdf")
    vocabulary = read_vocabulary("lookout/style/typos_checker/common_voc.csv")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def correct_comments(self, comments: list) -> pandas.DataFrame:
        typos = []
        for comment_id, comment in comments:
            split = re.compile("\w+").findall(comment)
            for i in range(len(split)):
                if split[i] in self.vocabulary:
                    continue
                left = max(0, i - 2)
                right = min(len(split), i + 3)
                typos.append([comment_id, split[i], split[left:i], split[i + 1:right]])
        typos = pandas.DataFrame(typos, columns=["comment_id", "typo", "before", "after"])
        suggestions = self.corrector.suggest(typos, n_candidates=3, return_all=False)
        comments_suggestions = defaultdict(dict)
        for key, corrections in suggestions.items():
            comments_suggestions[typos.loc[key].comment_id][typos.loc[key].typo] = corrections
        return comments_suggestions

    def train

    def dump(self) -> str:
        return "Typos correcting model with vocabulary size %d" % len(self.vocabulary)

    def _generate_tree(self) -> dict:
        return {}

    def _load_tree(self, tree: dict) -> None:
        pass

