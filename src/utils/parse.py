### FROM JACK HAWTHORNE MASTER THESIS

from . import locator
import toml
import time
from collections import OrderedDict


def parse_schema():
    with open(locator.get_schema(), 'r') as f:
        s = OrderedDict(toml.load(f))
    for k, v in s.items():
        s[k] = OrderedDict(v)
    return s


def parse_config(project_path):
    if project_path is None:
        return {}
    config = {}
    for f in locator.get_config_files(project_path):
        with open(f, 'r') as tom:
            if f.stem not in config:
                config[f.stem] = {}
            config[f.stem].update(toml.load(tom))
    return config


def parse_profile(data, index, index_by='column', skip=0):
    skip = int(skip)
    numeric_index = True
    try:
        index = int(index)
    except ValueError:
        numeric_index = False
    profile = []
    if index_by == 'column':
        if numeric_index:
            for i, row in enumerate(data):
                if i >= skip and not row[index]:
                    profile.append(float(row[index]))
        else:
            for i, row in enumerate(data):
                if i == skip:
                    headers = {name: j for j, name in enumerate(row)}
                elif i > skip:
                    j = headers[index]
                    if row[j] != '':
                        profile.append(float(row[j]))
        # elif index_by == 'row':
        return profile

if __name__ == "__main__":
    pass