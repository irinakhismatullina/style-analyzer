from collections import defaultdict
import logging
from pprint import pformat
from typing import Dict, Iterable, Any

from skopt import BayesSearchCV
from skopt.space import Categorical, Integer

from lookout.core.analyzer import Analyzer, AnalyzerModel, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import File
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import with_changed_uasts_and_contents, with_uasts_and_contents
from lookout.style.format.diff import find_new_lines
from lookout.style.format.features import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.rules import TopDownGreedyBudget, TrainableRules


class TypoAnalyzer(Analyzer):
    log = logging.getLogger("TypoAnalyzer")
    model_type = FormatModel # TODO: add new model
    version = "1"
    description = "Source code comment typos"

    @with_changed_uasts_and_contents
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_request_stub: DataStub, **data) -> [Comment]:
        changes = data["changes"]
        comments = []
        for change in changes:
            old_comments= set([node.token
                                   for node in uast2sequence(change.base.uast)
                                   if bblfsh.role_id("COMMENT")
                                   in node.roles])
            id_nodes = [node
                        for node in uast2sequence(change.head.uast)
                        if bblfsh.role_id("COMMENT") in node.roles and
                        node.token not in old_identifiers]
            comments = [node.token for node in id_nodes]
            if len(comments) > 0:
                suggestions = self.model.check_comments(comments,
                                                           self.checker)
                for index, id_node in enumerate(id_nodes):
                    if index in suggestions.keys():
                        corrections = suggestions[index]
                        for token in corrections.keys():
                            comment = Comment()
                            comment.file = change.head.path
                            corrections_line = ""
                            for candidate in corrections[token]:
                                corrections_line += candidate[0] + \
                                                    " (%d)," % \
                                                    int(candidate[1] * 100)
                            comment.text = "Typo inside comment '%s' " \
                                           "in token '%s'. " \
                                           "Possible corrections:" % \
                                           (id_node.token, token) +\
                                           corrections_line[:-1]
                            comment.line = id_node.start_position.line
                            comment.confidence = int(
                                corrections[token][0][1] * 100)
                            comments.append(comment)
        self.checker = None
        return comments

    @classmethod
    @with_uasts_and_contents
    def train(cls, ptr: ReferencePointer, config: Dict[str, Any], data_request_stub: DataStub,
              **data) -> AnalyzerModel:
        """
        Train a model given the files available.

        :param ptr: Git repository state pointer.
        :param config: configuration dict.
        :param data: contains "files" - the list of files in the pointed state.
        :param data_request_stub: connection to the Lookout data retrieval service, not used.
        :return: AnalyzerModel containing the learned rules, per language.
        """
        cls.log.info("train %s %s %s", ptr.url, ptr.commit, data)
        files = data["files"]
        for file in files:
            cls.log.info("%s %s %d", file.path, file.language, len(file.uast.children))
        model = TyposModel().construct(cls, ptr)
        return model.train()
