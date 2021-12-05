from numpy import mean, true_divide
from numpy import std

import pandas as pd

def FP_TCC_CreateMixFingerprints():
    X = pd.read_csv('dataset-mix-tcc-joao.csv')
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

    d['date_time'].append('end')

    contador = 1

    x = []

    y = []

    isNextTheEnd = False

    mediaDeviceSignal = [0] * 5
    contadorDeviceSignal = [0] * 5

    for index, (key, content) in enumerate(d.items()):

        if key == 'date_time':

            for index2, content2 in enumerate(content):

                if content[index2+1] == 'end':
                    
                    isNextTheEnd = True

                if content[index2+1] == content2 or isNextTheEnd == True:

                    contador += 1

                    if d['device_id'][index2] == 'sala':

                        mediaDeviceSignal[0] += d['device_signal'][index2]
                        contadorDeviceSignal[0] += 1

                    elif d['device_id'][index2] == 'quarto1':

                        mediaDeviceSignal[1] += d['device_signal'][index2]
                        contadorDeviceSignal[1] += 1

                    elif d['device_id'][index2] == 'quarto2':

                        mediaDeviceSignal[2] += d['device_signal'][index2]
                        contadorDeviceSignal[2] += 1

                    elif d['device_id'][index2] == 'servico':

                        mediaDeviceSignal[3] += d['device_signal'][index2]
                        contadorDeviceSignal[3] += 1

                    elif d['device_id'][index2] == 'cozinha':

                        mediaDeviceSignal[4] += d['device_signal'][index2]
                        contadorDeviceSignal[4] += 1

                if content[index2+1] != content2 or isNextTheEnd == True:

                    if contadorDeviceSignal[0] != 0:
                        mediaDeviceSignal[0] /= contadorDeviceSignal[0]
                    else:
                        mediaDeviceSignal[0] = -88

                    if contadorDeviceSignal[1] != 0:
                        mediaDeviceSignal[1] /= contadorDeviceSignal[1]
                    else:
                        mediaDeviceSignal[1] = -88

                    if contadorDeviceSignal[2] != 0:
                        mediaDeviceSignal[2] /= contadorDeviceSignal[2]
                    else:
                        mediaDeviceSignal[2] = -88

                    if contadorDeviceSignal[3] != 0:
                        mediaDeviceSignal[3] /= contadorDeviceSignal[3]
                    else:
                        mediaDeviceSignal[3] = -88

                    if contadorDeviceSignal[4] != 0:
                        mediaDeviceSignal[4] /= contadorDeviceSignal[4]
                    else:
                        mediaDeviceSignal[4] = -88
                    
                    x.append(mediaDeviceSignal)

                    nomesID = {
                                45094: 'cozinha',
                                45120: 'cozinha',
                                45101: 'cozinha',
                                45136: 'cozinha',
                                
                                45181: 'quarto 2',
                                45161: 'quarto 2',
                                45143: 'quarto 2',
                                45174: 'quarto 2',
                                
                                45194: 'quarto 3',
                                45201: 'quarto 3',
                                45213: 'quarto 3',
                                45219: 'quarto 3',

                                45241: 'corredor',
                                45250: 'corredor',
                                45266: 'corredor',
                                45232: 'corredor',

                                45288: 'quarto 1',
                                45303: 'quarto 1',
                                45276: 'quarto 1',
                                45308: 'quarto 1',

                                45345: 'sala',
                                45333: 'sala',                          
                                45322: 'sala',
                                45352: 'sala',

                                45385: 'banheiro',
                                45424: 'banheiro',
                                45410: 'banheiro',
                                45397: 'banheiro'
                            }
                    
                    y.append(nomesID[d['id_addr'][index2]])

                    mediaDeviceSignal = [0] * 5
                    contadorDeviceSignal = [0] * 5

                contador = 1

                if isNextTheEnd == True:
                        break
     
    return x, y