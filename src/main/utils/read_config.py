import yaml
from main.definition import ROOT_DIR


def read_config(file_name):
    file_name = ROOT_DIR + "/resources/" + file_name

    # Reading from YML file.
    with open(file_name, "r") as ymlfile:
        configs = yaml.load(ymlfile)

    ymlfile.close()
    return configs
