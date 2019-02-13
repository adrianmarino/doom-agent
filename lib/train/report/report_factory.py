from lib.train.report.train_metric_report import TrainMetricReport
from lib.train.report.formatter.json_report_formatter import PrettyJsonFormatter


class AgentReportFactory:
    @staticmethod
    def json_report(cfg, weights_file=''):
        report = TrainMetricReport(
            cfg['checkpoint_path'],
            cfg['metric_path'],
            cfg['report.metrics'],
            cfg,
            weights_file,
            cfg['report.last_times']
        )
        return report.format_to(PrettyJsonFormatter())
