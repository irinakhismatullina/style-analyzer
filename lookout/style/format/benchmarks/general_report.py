"""Facilities to report the quality of a given model on a given dataset."""
from collections import Counter, defaultdict, OrderedDict
import copy
import glob
from itertools import chain
import json
import logging
import os
import pprint
from typing import Any, Iterable, List, Mapping, NamedTuple, Optional, Tuple, Union

from bblfsh import BblfshClient
import jinja2
from lookout.core import slogging
from lookout.core.analyzer import ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import File
from lookout.core.data_requests import DataService, request_files
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report, confusion_matrix

from lookout.style.format.analyzer import FileFix, FormatAnalyzer
from lookout.style.format.benchmarks.time_profile import profile
from lookout.style.format.classes import CLASS_REPRESENTATIONS
from lookout.style.format.descriptions import describe_rule
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.utils import generate_comment, merge_dicts, prepare_files
from lookout.style.format.virtual_node import VirtualNode


def convert_to_sklearn_format(vnodes: Iterable[VirtualNode],
                              ) -> Tuple[List[int], List[int], List[VirtualNode], List[str]]:
    """
    Convert VirtualNode-s sequence to true targets, predicted targets and corresponding vnodes.

    :param vnodes: VirtualNode-s sequence.
    :return: true targets, predicted targets, corresponding vnodes and target names.
    """
    y, y_pred, vnodes_y = [], [], []
    class_sequences_to_labels = defaultdict(lambda: len(class_sequences_to_labels))
    for vnode in vnodes:
        if vnode.y is None:
            continue
        vnodes_y.append(vnode)
        if hasattr(vnode, "y_old"):
            y.append(class_sequences_to_labels[vnode.y_old])
            y_pred.append(class_sequences_to_labels[vnode.y])
        else:
            y.append(class_sequences_to_labels[vnode.y])
            y_pred.append(class_sequences_to_labels[vnode.y])

    labels_to_class_sequences = {label: seq for seq, label in class_sequences_to_labels.items()}
    target_names = ["".join(CLASS_REPRESENTATIONS[cls] for cls in labels_to_class_sequences[i])
                    for i in range(len(labels_to_class_sequences))]

    return y, y_pred, vnodes_y, target_names


def _load_jinja2_template(report_template_filename: str) -> jinja2.Template:
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
                             extensions=["jinja2.ext.do"])
    env.filters.update({
        "pformat": pprint.pformat,
        "deepcopy": copy.deepcopy,
    })
    loader = jinja2.FileSystemLoader((os.path.join(os.path.dirname(__file__), "..", "templates"),),
                                     followlinks=True)
    template = loader.load(env, report_template_filename)
    # the following is really needed, otherwise e.g. range is undefined
    template.globals = template.environment.globals
    return template


def generate_quality_report(language: str, ptr: ReferencePointer, vnodes, max_files: int):
    """Generate report: classification report, confusion matrix, files with most errors."""
    y, y_pred, vnodes_y, target_names = convert_to_sklearn_format(vnodes)
    # classification report
    c_report = classification_report(y, y_pred, output_dict=True, target_names=target_names,
                                     labels=list(range(len(target_names))))
    avr_keys = ["macro avg", "micro avg", "weighted avg"]
    c_sorted = OrderedDict((key, c_report[key])
                           for key in sorted(c_report, key=lambda k: -c_report[k]["support"])
                           if key not in avr_keys)
    for key in avr_keys:
        c_sorted[key] = c_report[key]
    # confusion matrix
    mat = confusion_matrix(y, y_pred)
    # sort files by mispredictions
    file_mispred = []
    for gt, pred, vn in zip(y, y_pred, vnodes_y):
        if gt != pred:
            file_mispred.append(vn.path)
    file_stat = Counter(file_mispred)
    to_show = file_stat.most_common()
    if max_files > 0:
        to_show = to_show[:max_files]

    template = _load_jinja2_template("quality_report.md.jinja2")
    # TODO(vmarkovtsev): move all the logic inside the template
    res = template.render(language=language, ptr=ptr, cl_report=c_sorted, conf_mat=mat,
                          target_names=target_names, files=to_show)
    return res


def generate_model_report(model: FormatModel,
                          languages: Optional[Union[str, Iterable[str]]] = None) -> str:
    """
    Generate report about model - description for each rule, min/max support, min/max confidence.

    :param model: trained format model.
    :param languages: Languages for which report should be created. You can specify one \
                      language as string, several as list of strings or None for all languages in \
                      the model.
    :return: report in str format.
    """
    languages = languages if languages is not None else model.languages
    languages = languages if isinstance(languages, Iterable) else [languages]
    for language in languages:
        if language not in model:
            raise NotFittedError(language)
    template = _load_jinja2_template("model_report.md.jinja2")
    return template.render(model=model, languages=languages, FeatureExtractor=FeatureExtractor,
                           describe_rule=describe_rule)


@profile
def print_reports(input_pattern: str, bblfsh: str, language: str, model_path: str,
                  config: Union[str, dict] = "{}", log_level: str = "INFO") -> None:
    """Print quality and model reports for a given model on a given dataset."""
    slogging.setup(log_level, False)
    log = logging.getLogger("quality_report")

    class FakeStub:
        def __init__(self, files: Iterable[File]):
            self.files = files

        def GetFiles(self, *args, **kwargs):
            return self.files

    class FakeDataService:
        def __init__(self, bblfsh_client: BblfshClient, files: Iterable[File]):
            self.bblfsh_client = bblfsh_client
            self.files = files

        def get_bblfsh(self):
            return self.bblfsh_client._stub

        def get_data(self):
            return FakeStub(self.files)

    class FakePointer:
        def to_pb(self):
            return None

    config = config if isinstance(config, dict) else json.loads(config)
    model = FormatModel().load(model_path)
    if language not in model:
        raise NotFittedError()
    rules = model[language]
    client = BblfshClient(bblfsh)
    files = prepare_files(glob.glob(input_pattern, recursive=True), client, language)
    log.info("Model parameters: %s" % rules.origin_config)
    log.info("Rules stats: %s" % rules)
    log.info("Number of files: %s" % (len(files)))
    for report in QualityReportAnalyzer(model, input_pattern, config).analyze(
            FakePointer(), None, data_service=FakeDataService(client, files)):
        print(report.text)


class ReportAnalyzer(FormatAnalyzer):
    """
    Base class for different kind of reports.

    * analyze - generate report for all files. If you want only aggregated report set aggregate
    flag to True in analyze config.
    * train - train or load the model.

    Child classes are required to implement 2 methods:
    * generate_report
    * generate_model_report (optional - by default it will return empty string)
    """

    Changes = NamedTuple("Changes", (("base", File), ("head", File)))
    defaults_for_analyze = merge_dicts(FormatAnalyzer.defaults_for_analyze,
                                       {"aggregate": False})

    def generate_quality_report(self, fixes: Iterable[FileFix]) -> str:
        """
        Generate report.

        :param fixes: fixes with all required information for report generation.
        :return: Report.
        """
        raise NotImplementedError()

    def generate_model_report(self) -> str:
        """
        Generate report about the trained model.

        :return: Report.
        """
        return ""

    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_service: DataService, **data) -> List[Comment]:
        """
        Analyze ptr_from revision and a make quality report for all files in it.

        If you want to get an aggregated report set aggregate flag to True in analyze config.

        :param ptr_from: Git repository state pointer to the base revision.
        :param ptr_to: Git repository state pointer to the head revision. Not used.
        :param data_service: Connection to the Lookout data retrieval service.
        :param data: Contains "files" - the list of changes in the pointed state.
        :return: List of comments.
        """
        comments = []
        fixes = []
        files = request_files(data_service.get_data(), ptr_from, contents=True, uast=True)
        for fix in self.generate_file_fixes(
                data_service, [ReportAnalyzer.Changes(File(), f) for f in list(files)]):
            filepath = fix.head_file.path
            if fix.error:
                continue
            if self.config["aggregate"]:
                fixes.append(fix)
            else:
                report = self.generate_quality_report(fixes=[fix])
                comments.append(generate_comment(
                    filename=filepath, line=0, confidence=100, text=report))
        if self.config["aggregate"]:
            report = self.generate_quality_report(fixes=fixes)
            comments.append(generate_comment(
                filename="", line=0, confidence=100, text=report))
        comments.append(generate_comment(
            filename="", line=0, confidence=100, text=self.generate_model_report()))
        return comments

    @classmethod
    def train(cls, ptr: ReferencePointer, config: Mapping[str, Any], data_service: DataService,
              **data) -> FormatModel:
        """
        Train a model given the files available or load existing model.

        If you set config["model"] to path in the file system model will be loaded otherwise
        a model is trained in a regular way.

        :param ptr: Git repository state pointer.
        :param config: Configuration dict.
        :param data: Contains "files" - the list of files in the pointed state.
        :param data_service: Connection to the Lookout data retrieval service.
        :return: FormatModel containing the learned rules, per language.
        """
        return FormatModel().load(config["model"]) if "model" in config else \
            super().train(ptr, config, data_service)


class QualityReportAnalyzer(ReportAnalyzer):
    """
    Generate basic quality reports for the model.

    * analyze - generate report for all files. If you want only aggregated report set aggregate
    flag to True in analyze config.
    * train - train or load the model.

    It is possible to run this analyzer independently and query it with lookout-sdk.
    If you want to use pretrained model it is possible to specify it in config, for example:
    `--config-json='{"style.format.analyzer.FormatAnalyzer": {"model": "/saved/model.asdf"}}`
    Otherwise model will be trained with `FormatAnalyzer.train()`

    Usage examples:
    1) Launch analyzer: `analyzer run lookout.style.format.quality_report_analyzer -c config.yml`
    2) Query analyzer
    2.1) Get one quality report per file for pretrained model /saved/model.asdf:
    ```
    lookout-sdk review ipv4://localhost:2000 --git-dir /git/dir/ --from REV1 --to REV2 \
    --config-json='{"style.format.analyzer.FormatAnalyzer": {"model": "/saved/model.asdf"}}'
    ```
    2.2) Get aggregated quality report for all files without pretrained model
    ```
    lookout-sdk review ipv4://localhost:2000 --git-dir /git/dir/ --from REV1 --to REV2 \
    --config-json='{"style.format.analyzer.FormatAnalyzer": {"aggregate": true}}'
    ```
    """

    version = "1"
    description = "Source code formatting quality report generator: " \
                  "whitespace, new lines, quotes, etc."
    defaults_for_analyze = merge_dicts(ReportAnalyzer.defaults_for_analyze,
                                       {"max_files": 10})

    def generate_model_report(self) -> str:
        """
        Generate report about the trained model.

        :return: report.
        """
        return generate_model_report(model=self.model)

    def generate_quality_report(self, fixes: Iterable[FileFix]) -> str:
        """
        Generate quality report: classification report, confusion matrix, files with most errors.

        :return: report.
        """
        fixes = list(fixes)
        if not fixes:
            raise ValueError("There are no fixes for %s", self.model)
        vnodes = chain.from_iterable(fix.file_vnodes for fix in fixes)
        # FIXME(vmarkovtsev): we are taking the first fix here which does not work for >1 language
        return generate_quality_report(
            fixes[0].language, self.model.ptr, vnodes, self.config["max_files"])


analyzer_class = QualityReportAnalyzer