#! /usr/bin/python3

# Script for run DataAgent service.
# sample: python ./run.py

import subprocess
import argparse
import getpass
import os
import sys
from pathlib import Path

from misc import utils_run

conf_file = 'conf/conf.yaml'

def docker_build():
    c = utils_run.load_config(conf_file)
    project = c['projectName']
    tag = project

    # Remove docker image by Tag name. Usefull for fully rebuild container from scratch.
    if False:        
        utils_run.remove_docker_image(tag)

    command = """
    docker build \
        --tag {TAG} \
        --file ./docker/Dockerfile \
        --build-arg USER={USER} \
        --build-arg USER_ID={USER_ID} \
        --build-arg GROUP_ID={GROUP_ID} \
        .
    """.format(TAG=tag, USER=getpass.getuser(), USER_ID=os.getuid() + 10, GROUP_ID=os.getgid() + 10)

    subprocess.run(command, shell=True)

    # Show Docker image size
    subprocess.run('docker images | grep {}'.format(tag), shell=True)

def docker_run():
    c = utils_run.load_config(conf_file)
    project = c['projectName']

    # For conveniant purpose miroring project sources into container.
    # So we able to edit sources outside container but run inside container.
    curr_file_path = os.path.realpath(__file__)
    proj_path = str(Path(curr_file_path).parent)
    volumes = '-v {}:/project'.format(proj_path)
    
    tag = project
    command = """
    docker run --rm -it \
        --ipc=host \
        --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
        {VOLUMES} \
        {TAG}:latest 
    """.format(TAG=tag, VOLUMES=volumes)

    subprocess.run(command, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', nargs='?', help='Command type, sample: "python run.py build".')
    options = parser.parse_args()

    if options.cmd == 'build':
        docker_build()
    else:
        docker_build()
        docker_run()

if __name__ == "__main__":
    main()