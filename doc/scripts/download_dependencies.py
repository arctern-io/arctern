# -----------------------------------------------------------------------------
# Name:         download_dependencies.py
# Purpose:      Script to read the conda list export file and download every dependency package in the list for all
#               available platforms
# Dependencies: requests, python 3.5
# -----------------------------------------------------------------------------

import requests
from pathlib import Path
import sys
import os

#region Gets path to environment file and downloads folder via command line
env_file = None
try:
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    if arg1 is not None:
        env_file = arg1
    else:
        env_file = 'arctern_dependencies.txt'

    if arg2 is not None:
        channel_path = arg2
    else:
        file_path = os.path.dirname(os.path.realpath(__file__))
        channel_path = os.path.join(file_path, 'channel')
except:
    env_file = 'arctern_dependencies.txt'
    file_path = os.path.dirname(os.path.realpath(__file__))
    channel_path = os.path.join(file_path, 'channel')

print("Using environment list file: " + env_file)
#endregion

#region Read env file:
path_list1 = []
with open(env_file, 'r') as env_file_handle:
    for line in env_file_handle.readlines():
        path_list1.append(line)
#endregion

#region Download contents
for url in path_list1:
    p1 = url.rstrip('\r\n')
    path_splits = p1.split('/')
    file_name = path_splits[-1]

    print("Getting ", file_name)

    current_platform = path_splits[-2]
    if path_splits[-4] != "label":
        download_folder = os.path.join(channel_path, current_platform)
    else:
        labels = path_splits[-3]
        download_folder = os.path.join(channel_path, "label", labels, current_platform)
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
        print("Created ", download_folder)

    if path_splits[-4] != "label":
        download_file_path = os.path.join(channel_path, current_platform, file_name)
    else:
        labels = path_splits[-3]
        download_file_path = os.path.join(channel_path, "label", labels, current_platform, file_name)
    if os.path.exists(download_file_path):
        os.remove(download_file_path)

    print("URL: " + p1)
    resp1 = requests.get(p1)
    if resp1.status_code == 200:
        with open(download_file_path, 'wb') as f_handle:
            f_handle.write(resp1.content)
            print('\t Downloaded {}\{}'.format(current_platform, file_name))
    else:
        print("\t Error with download: " + current_platform + " : " + resp1.__str__())

#endregion
