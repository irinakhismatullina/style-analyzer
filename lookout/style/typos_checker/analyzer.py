import logging
from typing import Dict, Any

import bblfsh
from sourced.ml.algorithms import uast2sequence

from lookout.core.analyzer import Analyzer, AnalyzerModel, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import with_changed_uasts_and_contents, with_uasts_and_contents
from lookout.style.typos_checker.model import CommentsCheckingModel


class CommentsCheckingAnalyzer(Analyzer):
    log = logging.getLogger("CommentsCheckingAnalyzer")
    model_type = CommentsCheckingModel
    version = "1"
    description = "Source code comment typos"

    @with_changed_uasts_and_contents
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_request_stub: DataStub, **data) -> [Comment]:
        commentFilter = "//*[@roleComment]"
        changes = data["changes"]
        comments = []

        for change in changes:
            old_comments = set([node.token for node in bblfsh.filter(change.base.uast, commentFilter)])
            comment_nodes = [node for node in bblfsh.filter(change.head.uast, commentFilter)
                             if node.token not in old_comments]
            comment_tokens = [node.token for node in comment_nodes]
            if len(comment_tokens) > 0:
                suggestions = self.model.correct_comments(comment_tokens)
                for index, comment_node in enumerate(comment_nodes):
                    if index in suggestions.keys():
                        corrections = suggestions[index]
                        line_suggestions = ""
                        for token in corrections.keys():
                            corrections_line = ""
                            for candidate in corrections[token]:
                                corrections_line += " `%s` (%d)," % \
                                                    (candidate[0], int(candidate[1] * 100) )
                            line_suggestions += "Typo inside comment in word `%s`. " \
                                           "Possible corrections: %s\n" % \
                                           ( token, corrections_line[:-1] )
                        comment = Comment()
                        comment.file = change.head.path
                        comment.line = comment_node.start_position.line
                        comment.text = line_suggestions
                        comments.append(comment)
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
        model = CommentsCheckingModel()
        return model
