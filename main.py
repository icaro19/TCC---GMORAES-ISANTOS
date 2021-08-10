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

X = pd.read_csv('dataset-fixo-tcc-joao.csv')
X.columns = ['ID', 'date_time', 'device_signal', 'device_id', 'id_addr', 'pack_type']
 
aux = []

for index, content in enumerate(X['date_time']):
    columnSplit = content.split(" ")
    aux.append(columnSplit[1])
    
X['date_time'] = aux

contador = 0

for index, content in enumerate(X['date_time'], start = 1):
    
    hoursContent = content[0] + content[1]
    minutesContent = content[3] + content[4]
    secondsContent = content[6] + content[7]

    hoursBefore = X['date_time'][index-1][0] + X['date_time'][index-1][1]
    minutesBefore = X['date_time'][index-1][3] + X['date_time'][index-1][4]
    secondsBefore = X['date_time'][index-1][6] + X['date_time'][index-1][7]
    
   # print('content '+str(index-1)+': '+hoursBefore+':'+minutesBefore+':'+secondsBefore)
   # print('content '+str(index)+': '+hoursContent+':'+minutesContent+':'+secondsContent)
   # print('--------------------------------')

    '''if (hoursContent == hoursBefore) and (minutesContent == minutesBefore) and (secondsContent == secondsBefore):
        
        print('comparando elementos ' + str(index) + ' e ' + str(index-1) + ': Horas: ' + hoursContent + ' e '+ hoursBefore + '; minutos: '
              + minutesContent + ' e ' + minutesBefore+ '; segundos: ' + secondsContent + ' e ' + secondsBefore)
        
        contador+=1
        print('existem ' + str(contador) + ' pacotes no momento ' + hoursContent+':'+minutesContent+':'+secondsContent)
    
    else:
        
        contador = 0'''