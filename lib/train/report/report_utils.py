from shutil import copyfile

from lib.train.model.model_utils import get_best_weights_file_from, get_loss_model_weights_path
from lib.train.report.report_factory import AgentReportFactory
from lib.util.os_utils import create_file_path
from lib.util.time_utils import str_time


def write_report(report_path, report, time, ext):
    report_file_path = create_file_path(report_path, f'{time}_report', ext)
    file = open(report_file_path, 'w')
    file.write(report)
    file.close()


def cp_best_weights_to_reports_path(checkpoint_path, report_path, timestamp):
    best_weights_file = get_best_weights_file_from(checkpoint_path)
    loss = get_loss_model_weights_path(best_weights_file)
    result_path = create_file_path(report_path, f'{timestamp}-weights-loss_{loss}', 'h5')
    copyfile(best_weights_file, result_path)
    return result_path


def build_train_report(cfg):
    time = str_time()
    weights_file = cp_best_weights_to_reports_path(cfg['checkpoint.path'], cfg['report.path'], time)
    report = AgentReportFactory.json_report(cfg, weights_file)
    write_report(cfg['report.path'], report, time, 'json')
    print(report)