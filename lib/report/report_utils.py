from lib.util.os_utils import create_file_path


def write_report(report_path, report, str_time, ext):
    report_file_path = create_file_path(report_path, f'{str_time}_report', ext)
    file = open(report_file_path, 'w')
    file.write(report)
    file.close()
