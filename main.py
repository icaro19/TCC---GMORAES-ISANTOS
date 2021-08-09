from numpy import mean
from numpy import std

import csv
import pandas as pd
import datetime as dt

# abre arquivo com id de cada dispositivo e sua localizacao
d = {}
with open("id_addrs_fixo.txt") as f:
    f.__next__()
    for line in f:
#        print(line.split("|"))
        (key, ignore, val, ig) = line.split("|")
        d[int(key)] = val

# abre csv e trata
X = pd.read_csv('dataset-fixo-tcc-joao.csv', index_col='ID')
X.to_dict()
tempo = {}
tempo = X.select_dtypes(include=['object'])
ntempo = {}
ntempo = tempo.convert_dtypes()

#print(ntempo)
