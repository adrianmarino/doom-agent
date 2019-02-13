from lib.train.report.agent_metric_report import AgentMetricReport
from lib.train.report.formatter.json_report_formatter import PrettyJsonFormatter


class AgentReportFactory:
    @staticmethod
    def json_report(cfg, weights_file=''):
        report = AgentMetricReport(
            cfg['checkpoint_path'],
            cfg['metric_path'],
            cfg['report.metrics'],
            cfg,
            weights_file,
            cfg['report.last_times']
        )
        return report.format_to(PrettyJsonFormatter())
