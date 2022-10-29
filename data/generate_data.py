import numpy as np
import pandas as pd 

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from sklearn import preprocessing

def generate_data_from_url(url_name):
    '''
    Generate data and labels from a url name --  KEEL repository
    '''
    zipfile = ZipFile(BytesIO(urlopen(url_name).read()))
    zipfile.extractall('./files/')
    data = open('./files/'+ url_name.split('/')[-1].split('.zip')[0] + '.dat').read()

    data_split = data.split('@data')[-1]
    data_split = data_split.split('\n')[1:]
    data_split = np.array(data_split[:-1])

    labels = []
    data = []
    for line in data_split:
        line = line.replace(" ", "")
        line = line.split(',')
        labels.append(line[-1])
        data.append(np.array(line[:-1], dtype=float))

    data =np.array(data)
    labels = np.array(preprocessing.LabelEncoder().fit_transform(labels))
    return data, labels


def generate_data_from_dat_file(file_name):
    '''
    Generate data and labels from a .dat file.
    '''
    data = open(file_name).read()

    data_split = data.split('@data')[-1]
    data_split = data_split.split('\n')[1:]
    data_split = np.array(data_split[:-1])

    labels = []
    data = []
    for line in data_split:
        line = line.replace(" ", "")
        line = line.split(',')
        labels.append(line[-1])
        data.append(np.array(line[:-1], dtype=float))

    data = np.array(data)
    labels = np.array(preprocessing.LabelEncoder().fit_transform(labels))
    return data, labels