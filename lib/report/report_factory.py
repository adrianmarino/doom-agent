from lib.report.agent_metric_report import AgentMetricReport
from lib.report.formatter.json_report_formatter import PrettyJsonFormatter


class AgentReportFactory:
    @staticmethod
    def json_report(cfg, weights_file=""):
        report = AgentMetricReport(
            cfg['checkpoint.path'],
            cfg['metric.path'],
            cfg['env.variables'],
            cfg['hiperparams'],
            weights_file
        )
        return report.format_to(PrettyJsonFormatter())
