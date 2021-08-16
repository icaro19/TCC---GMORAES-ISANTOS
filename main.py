from randomForest import RF_TCC
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

fingerprints = {

    'quarto1': [],
    'quarto2': [],
    'quarto3': [],
    'sala': [],
    'cozinha': [],
    'banheiro': [],
    'corredor': []
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
                    mediaDeviceSignal[0] = -99

                if contadorDeviceSignal[1] != 0:
                    mediaDeviceSignal[1] /= contadorDeviceSignal[1]
                else:
                    mediaDeviceSignal[1] = -99

                if contadorDeviceSignal[2] != 0:
                    mediaDeviceSignal[2] /= contadorDeviceSignal[2]
                else:
                    mediaDeviceSignal[2] = -99

                if contadorDeviceSignal[3] != 0:
                    mediaDeviceSignal[3] /= contadorDeviceSignal[3]
                else:
                    mediaDeviceSignal[3] = -99

                if contadorDeviceSignal[4] != 0:
                    mediaDeviceSignal[4] /= contadorDeviceSignal[4]
                else:
                    mediaDeviceSignal[4] = -99
                
                x.append(mediaDeviceSignal)

                if d['id_addr'][index2] == 41312:

                    y.append('quarto 2')
                    #stringQuarto2 = content2 + ' quarto 2: [' + str(mediaDeviceSignal[0]) + ' ' + str(mediaDeviceSignal[1]) + ' ' + str(mediaDeviceSignal[2]) + ' ' + str(mediaDeviceSignal[3]) + ' ' + str(mediaDeviceSignal[4]) + ']'
                    #fingerprints['quarto2'].append(stringQuarto2)
                    #print(stringQuarto2)


                elif d['id_addr'][index2] == 41483:
                    
                    y.append('banheiro')
                    #stringBanheiro = content2 + ' banheiro: [' + str(mediaDeviceSignal[0]) + ' ' + str(mediaDeviceSignal[1]) + ' ' + str(mediaDeviceSignal[2]) + ' ' + str(mediaDeviceSignal[3]) + ' ' + str(mediaDeviceSignal[4]) + ']'
                    #fingerprints['banheiro'].append(stringBanheiro)
                    #print(stringBanheiro)

                elif d['id_addr'][index2] == 41814:
                   
                    y.append('cozinha')
                    #stringCozinha = content2 + ' cozinha: [' + str(mediaDeviceSignal[0]) + ' ' + str(mediaDeviceSignal[1]) + ' ' + str(mediaDeviceSignal[2]) + ' ' + str(mediaDeviceSignal[3]) + ' ' + str(mediaDeviceSignal[4]) + ']'
                    #fingerprints['cozinha'].append(stringCozinha)
                    #print(stringCozinha)

                elif d['id_addr'][index2] == 41642:
                    
                    y.append('quarto 1')
                    #stringQuarto1 = content2 + ' quarto 1: [' + str(mediaDeviceSignal[0]) + ' ' + str(mediaDeviceSignal[1]) + ' ' + str(mediaDeviceSignal[2]) + ' ' + str(mediaDeviceSignal[3]) + ' ' + str(mediaDeviceSignal[4]) + ']'
                    #fingerprints['quarto1'].append(stringQuarto1)
                    #print(stringQuarto1)

                elif d['id_addr'][index2] == 40553:
                    
                    y.append('corredor')
                    stringCorredor = content2 + ' corredor: [' + str(mediaDeviceSignal[0]) + ' ' + str(mediaDeviceSignal[1]) + ' ' + str(mediaDeviceSignal[2]) + ' ' + str(mediaDeviceSignal[3]) + ' ' + str(mediaDeviceSignal[4]) + ']'
                    fingerprints['corredor'].append(stringCorredor)
                    #print(stringCorredor)

                elif d['id_addr'][index2] == 41394:

                    y.append('sala')
                    #stringSala = content2 + ' sala: [' + str(mediaDeviceSignal[0]) + ' ' + str(mediaDeviceSignal[1]) + ' ' + str(mediaDeviceSignal[2]) + ' ' + str(mediaDeviceSignal[3]) + ' ' + str(mediaDeviceSignal[4]) + ']'
                    #fingerprints['sala'].append(stringSala)
                    #print(stringSala)

                elif d['id_addr'][index2] == 40768:

                    y.append('quarto 3')
                    #stringQuarto3 = content2 + ' quarto 3: [' + str(mediaDeviceSignal[0]) + ' ' + str(mediaDeviceSignal[1]) + ' ' + str(mediaDeviceSignal[2]) + ' ' + str(mediaDeviceSignal[3]) + ' ' + str(mediaDeviceSignal[4]) + ']'
                    #fingerprints['quarto3'].append(stringQuarto3)
                    #print(stringQuarto3)
                
                mediaDeviceSignal = [0] * 5
                contadorDeviceSignal = [0] * 5

                contador = 1

RF_TCC(x, y)