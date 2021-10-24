from randomForest import RF_TCC, RF_TCC_Predict, RF_TCC_TesteKFoldDifferentValues
from numpy import mean
from numpy import std

import csv
import pandas as pd
import datetime as dt
import randomForest

X = pd.read_csv('dataset-fixo-tcc-joao.csv')
X.columns = ['ID', 'date_time', 'device_signal', 'device_id', 'id_addr', 'pack_type']
 
aux = []
dateTimeColumn = X['date_time']

for index, content in enumerate(dateTimeColumn):
    columnSplit = content.split(" ")
    aux.append(columnSplit[1][:-7])

d = {
    'date_time': [],
    'device_signal': [],
    'device_id': [],
    'id_addr': []
}

d['date_time'] = aux
d['device_signal'] = X['device_signal'].tolist()
d['device_id'] = X['device_id'].tolist()
d['id_addr'] = X['id_addr'].tolist()

d['date_time'].append(0)

contador = 1

aux = {
    'date_time': [],
    'device_signal': [],
    'device_id': [],
    'id_addr': []
}

x = []

y = []

mediaDeviceSignal = [0] * 5
contadorDeviceSignal = [0] * 5

for index, (key, content) in enumerate(d.items()):
    
    if key == 'date_time':
        
        for index2, content2 in enumerate(content):
            
            if(content2 == 0):
                break

            if content[index2+1] == content2:
                
                contador+=1

                if d['device_id'][index2] == 'sala':
                    
                    mediaDeviceSignal[0] += d['device_signal'][index2]
                    contadorDeviceSignal[0]+=1
                        
                elif d['device_id'][index2] == 'quarto1':
                    
                    mediaDeviceSignal[1] += d['device_signal'][index2]
                    contadorDeviceSignal[1]+=1

                elif d['device_id'][index2] == 'quarto2':
                
                    mediaDeviceSignal[2] += d['device_signal'][index2]
                    contadorDeviceSignal[2]+=1
                    
                elif d['device_id'][index2] == 'servico':
                   
                    mediaDeviceSignal[3] += d['device_signal'][index2]
                    contadorDeviceSignal[3]+=1

                elif d['device_id'][index2] == 'cozinha':
                    
                    mediaDeviceSignal[4] += d['device_signal'][index2]
                    contadorDeviceSignal[4]+=1
            
            else:

                if contadorDeviceSignal[0] != 0:
                    mediaDeviceSignal[0] /= contadorDeviceSignal[0]
                else:
                    mediaDeviceSignal[0] = 0

                if contadorDeviceSignal[1] != 0:
                    mediaDeviceSignal[1] /= contadorDeviceSignal[1]
                else:
                    mediaDeviceSignal[1] = 0

                if contadorDeviceSignal[2] != 0:
                    mediaDeviceSignal[2] /= contadorDeviceSignal[2]
                else:
                    mediaDeviceSignal[2] = 0

                if contadorDeviceSignal[3] != 0:
                    mediaDeviceSignal[3] /= contadorDeviceSignal[3]
                else:
                    mediaDeviceSignal[3] = 0

                if contadorDeviceSignal[4] != 0:
                    mediaDeviceSignal[4] /= contadorDeviceSignal[4]
                else:
                    mediaDeviceSignal[4] = 0
                x.append(mediaDeviceSignal)

                nomesID = {
                            41312: 'quarto 2',
                            41483: 'banheiro',
                            41814: 'cozinha',
                            41642: 'quarto 1',
                            40553: 'corredor',
                            41394: 'sala',
                            40768: 'quarto 3'
                        }

                y.append(nomesID[d['id_addr'][index2]])
            
                mediaDeviceSignal = [0] * 5
                contadorDeviceSignal = [0] * 5
            
            contador = 1

#modelo do vetor de sinal: sala, quarto 1, quarto 2, serviço, cozinha
#RF_TCC(x, y)
RF_TCC_TesteKFoldDifferentValues(x, y)