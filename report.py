from lib.report.report_factory import AgentReportFactory
from lib.util.config import Config

cfg = Config('./config.yml')
report = AgentReportFactory.json_report(cfg)
print(report)
