from lib.report.agent_metric_report import AgentMetricReport
from lib.report.formatter.json_report_formatter import PrettyJsonFormatter


class AgentReportFactory:
    @staticmethod
    def json_report(cfg):
        report = AgentMetricReport(
            cfg['checkpoint.path'],
            cfg['metric.path'],
            cfg['env.variables'],
            cfg['hiperparams']
        )
        return report.format_to(PrettyJsonFormatter())
