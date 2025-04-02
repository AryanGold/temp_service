import yaml
from datetime import datetime

# Load configurations from conf file
def load_config(conf_file):
    with open(conf_file, 'r') as f:
        data = yaml.safe_load(f)

    return data

# Insert separator between year, month and day:
# "20250516" -> "2025-05-16"
def date_with_separator(date:str, sep:str = '-'):
    dt = datetime.strptime(date, "%Y%m%d")
    formatted_date = dt.strftime(f"%Y{sep}%m{sep}%d")
    return formatted_date
