import subprocess
import os
import sys
from pathlib import Path
import yaml

# Load configurations from conf file
def load_config(conf_file):
    with open(conf_file, 'r') as f:
        data = yaml.safe_load(f)

    return data

# Load Volumes list for pass to Docker
# Check if exists folder for docker volumes. if not then create it.
# filter_volumes - list of volumes names, if set then rturn volumes only from this list
def volumes_folders(c, filter_volumes=None):
    volumes = ''
    if 'VOLUMES' not in c:
        return ''

    for vol_name, val_paths in c['VOLUMES'].items():
        if filter_volumes is not None and vol_name not in filter_volumes:
            continue
        # Exrtact Host folder path, sample:
        # val_paths = /data/ext_libs:/ext_libs -> /data/ext_libs
        host_path = val_paths.split(':')[0]
        Path(host_path).mkdir(parents=True, exist_ok=True)

        volumes += '-v {} '.format(val_paths)
    return volumes

# Remove docker image by Tag name. Usefull for fully rebuild container from scratch.
def remove_docker_image(tag):
    command = """
    docker rmi --force {TAG}
    """.format(TAG=tag) 

    process = subprocess.Popen(command, shell=True)
    exit_code = process.wait()

# Get container volume path by name
# Sample "'NS_APP': '/data/ns_app:/ns_app'" -> "/ns_app"
def get_volume_path(c, vol_name):
    return c['VOLUMES'][vol_name].split(':')[1]

def check_aws_credentials_file(file_path):
    file_path = Path(file_path)
    if file_path.exists():
        return True
    return False