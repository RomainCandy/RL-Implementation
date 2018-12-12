import os
import logging.config
import yaml

_default_path = 'logger_config.yaml'
_default_level = logging.INFO

if os.path.exists(_default_path):
    with open(_default_path, 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
else:
    logging.basicConfig(level=_default_level)

logger = logging.getLogger("MAIN")
result_logger = logging.getLogger("RESULT")
