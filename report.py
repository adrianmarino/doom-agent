from lib.model.model_utils import get_best_weights_file_from
from lib.report.report_factory import AgentReportFactory
from lib.util.config import Config

cfg = Config('./config.yml')
report = AgentReportFactory.json_report(
    cfg,
    get_best_weights_file_from(cfg['checkpoint.path'])
)
print(report)
