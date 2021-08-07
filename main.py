from numpy import mean
from numpy import std

import csv
import pandas as pd

# abre arquivo com id de cada dispositivo e sua localizacao
d = {}
with open("id_addrs_fixo.txt") as f:
    f.__next__()
    for line in f:
#        print(line.split("|"))
        (key, ignore, val, ig) = line.split("|")
        d[int(key)] = val

print(d)

# abre csv e copia para dict
X = pd.read_csv('dataset-fixo-tcc-joao.csv', index_col='ID')
X.to_dict()
print(X)
