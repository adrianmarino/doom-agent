from lib.metric.metric_utils import train_metrics_summary
from lib.util.config import Config
from lib.util.dict_utils import print_dict

cfg = Config('./config.yml')

print_dict(train_metrics_summary(cfg['checkpoint.path'], cfg['metric.path'], cfg['env.variables']))
