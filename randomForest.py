import numpy as np

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import pprint 


def RF_TCC_Predict(model, x_treino, y_treino, x_teste, y_teste):
   
    #testa o modelo
    model.fit(x_treino, y_treino) 

    #faz a predição do teste
    resultado = model.predict(x_teste)

    #target_names = sorted(y_teste.unique())
    return classification_report(y_teste, resultado)

def RF_TCC_DifferentTrainSizesPredict(x, y):

    medidas = []
    stringMedidas = []
    stringMatrizes = []
    accuracy = []

    f = open("metrics.txt", "w+")

    #obtendo os melhores hiper-parâmetros para o nosso modelo
    best_hyperparams, stringAcc = RF_TCC_FindBestParams(x, y)

    #criando o modelo com os hiper-parâmetros obtidos
    model = RandomForestClassifier(n_estimators = best_hyperparams['n_estimators'],
                                   max_depth = best_hyperparams['max_depth'],
                                   min_samples_leaf = best_hyperparams['min_samples_leaf'],
                                   min_samples_split = best_hyperparams['min_samples_split'],
                                   max_leaf_nodes = best_hyperparams['max_leaf_nodes'],
                                   max_features = best_hyperparams['max_features']
                                   )

    labels = ['banheiro', 'corredor', 'cozinha', 'quarto 1', 'quarto 2', 'quarto 3', 'sala']

    #gerando o treino e teste básicos
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

    #gerando um conjunto de 1% do treino
    x_treino1percent, x_testeDescartável, y_treino1percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.01, shuffle = False, stratify = None)
    model.fit(x_treino1percent, y_treino1percent)
    resultado1percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado1percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado1percent))
    cm1percent = confusion_matrix(y_teste, resultado1percent, labels = labels)
    stringMatrizes.append('Confusion matrix with 1% [of D_Treino]:' + '\n' + str(cm1percent) + '\n\n')

    #gerando um conjunto de 2% do treino
    x_treino2percent, x_testeDescartável, y_treino2percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.02, shuffle = False, stratify = None)
    model.fit(x_treino2percent, y_treino2percent)
    resultado2percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado2percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado2percent))
    cm2percent = confusion_matrix(y_teste, resultado2percent, labels = labels)
    stringMatrizes.append('Confusion matrix with 2%: [of D_Treino]' + '\n' + str(cm2percent) + '\n\n')

    #gerando um conjunto de 5% do treino
    x_treino5percent, x_testeDescartável, y_treino5percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.05, shuffle = False, stratify = None)
    model.fit(x_treino5percent, y_treino5percent)
    resultado5percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado5percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado5percent))
    cm5percent = confusion_matrix(y_teste, resultado5percent, labels = labels)
    stringMatrizes.append('Confusion matrix with 5% [of D_Treino]:' + '\n' + str(cm5percent) + '\n\n')

    #gerando um conjunto de 10% do treino
    x_treino10percent, x_testeDescartável, y_treino10percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.1, shuffle = False, stratify = None)
    model.fit(x_treino10percent, y_treino10percent)
    resultado10percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado10percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado10percent))
    cm10percent = confusion_matrix(y_teste, resultado10percent, labels = labels)
    stringMatrizes.append('Confusion matrix with 10% [of D_Treino]:' + '\n' + str(cm10percent) + '\n\n')

    #gerando um conjunto de 25% do treino           
    x_treino25percent, x_testeDescartável, y_treino25percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.25, shuffle = False, stratify = None)
    model.fit(x_treino25percent, y_treino25percent)
    resultado25percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado25percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado25percent))
    cm25percent = confusion_matrix(y_teste, resultado25percent, labels = labels)
    stringMatrizes.append('Confusion matrix with 25% [of D_Treino]:' + '\n' + str(cm25percent) + '\n\n')

    #gerando um conjunto de 50% do treino
    x_treino50percent, x_testeDescartável, y_treino50percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.50, shuffle = False, stratify = None)
    model.fit(x_treino50percent, y_treino50percent)
    resultado50percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado50percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado50percent))
    cm50percent = confusion_matrix(y_teste, resultado50percent, labels = labels)
    stringMatrizes.append('Confusion matrix with 50% [of D_Treino]:' + '\n' + str(cm50percent) + '\n\n')

    #gerando um conjunto de 75% do treino
    x_treino75percent, x_testeDescartável, y_treino75percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.75, shuffle = False, stratify = None)
    model.fit(x_treino75percent, y_treino75percent)
    resultado75percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado75percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado75percent))
    cm75percent = confusion_matrix(y_teste, resultado75percent, labels = labels)
    stringMatrizes.append('Confusion matrix with 75% [of D_Treino]:' + '\n' + str(cm75percent) + '\n\n')

    #rodando o modelo com 100% do treino
    model.fit(x_treino, y_treino)
    resultado100percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado100percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado100percent))
    cm100percent = confusion_matrix(y_teste, resultado100percent, labels = labels)
    stringMatrizes.append('Confusion matrix with 100% [of D_Treino]:' + '\n' + str(cm100percent) + '\n\n')


    indexSizeTest = {0: 1,
                     1: 2,
                     2: 5,
                     3: 10,
                     4: 25,
                     5: 50,
                     6: 75,
                     7: 100
                    }

    for i in range(len(stringAcc)):
        f.write(stringAcc[i])

    for i in range(len(stringMatrizes)):
        f.write(stringMatrizes[i])

    for i in range(len(medidas)):
        accuracy.append(medidas[i]['accuracy'])
        f.write('Model scores with ' + str(indexSizeTest[i]) + '%' + ' of whole train dataset:\n' + str(stringMedidas[i]) + '\n')

    RF_TCC_CreateGraph([1, 2, 5, 10, 25, 50, 75, 100], accuracy, 'tamanho do conjunto de treino (%)', 'Acurácia (porção decimal)')

    f.close()

def RF_TCC_GenerateReports(x_treino, y_treino, x_teste, y_teste, size, model):

    if size != 1.0:

        x_treino, x_testeDescartável, y_treino, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = size, shuffle = False, stratify = None)
        
    else:
        pass

    cm = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    cm = np.array([cm,cm,cm,cm,cm,cm,cm])

    labels = ['banheiro', 'corredor', 'cozinha', 'quarto 1', 'quarto 2', 'quarto 3', 'sala']

    metricasMediasDict = {

                            'banheiro': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                            'corredor': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                            'cozinha': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                            'quarto 1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                            'quarto 2': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                            'quarto 3': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                            'sala': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                            'accuracy': 0
                         }

    for _ in range(10):

        model.fit(x_treino, y_treino)
        resultado = model.predict(x_teste)
        report = classification_report(y_teste, resultado, output_dict = True)

        for label, dictValores in report.items():

            if isinstance(dictValores, dict):

                for metricas in dictValores:

                    if label in labels:

                        metricasMediasDict[label][metricas] += (dictValores[metricas]/10)
                
            
        metricasMediasDict['accuracy'] += ( report['accuracy'] / 10)
        
        cm += confusion_matrix(y_teste, resultado, labels = labels)

        cm = np.around(cm, 2)

    cm /= 10

    cm = cm.tolist()

    return metricasMediasDict, cm

def RF_TCC_PredictMeans(x, y, tipo):

    medidas = []
    stringMatrizes = []
    accuracy = []

    tipoGrafico = {
                    0: ['metrics_centralizado.txt', 'grafico_acuracia_centralizado.png', 'Gráfico de acurácia de D_centralizado'],
                    1: ['metrics_distribuido.txt', 'grafico_acuracia_distribuido.png', 'Gráfico de acurácia de D_distribuido'],
                    2: ['metrics_unidos.txt', 'grafico_acuracia_unidos.png', 'Gráfico de acurácia de D_centralizado + D_distribuido']
                  }

    f = open(tipoGrafico[tipo][0], "w+")

    #criando o modelo com os hiper-parâmetros obtidos
    model = RandomForestClassifier(n_estimators = 5,
                                   max_depth = 4,
                                   min_samples_leaf = 2,
                                   min_samples_split = 2,
                                   max_leaf_nodes = 10,
                                   max_features = 'auto'
                                   )

    #gerando o treino e teste básicos
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

    #gerando um conjunto de 1% do treino
    resultado1percent, cm1percent = RF_TCC_GenerateReports(x_treino, y_treino, x_teste, y_teste, 0.01, model)
    medidas.append(resultado1percent)
    stringMatrizes.append(cm1percent)

    #gerando um conjunto de 2% do treino
    resultado2percent, cm2percent = RF_TCC_GenerateReports(x_treino, y_treino, x_teste, y_teste, 0.02, model)
    medidas.append(resultado2percent)
    stringMatrizes.append(cm2percent)

    #gerando um conjunto de 5% do treino
    resultado5percent, cm5percent = RF_TCC_GenerateReports(x_treino, y_treino, x_teste, y_teste, 0.05, model)
    medidas.append(resultado5percent)
    stringMatrizes.append(cm5percent)

    #gerando um conjunto de 10% do treino
    resultado10percent, cm10percent = RF_TCC_GenerateReports(x_treino, y_treino, x_teste, y_teste, 0.1, model)
    medidas.append(resultado10percent)
    stringMatrizes.append(cm10percent)

    #gerando um conjunto de 25% do treino           
    resultado25percent, cm25percent = RF_TCC_GenerateReports(x_treino, y_treino, x_teste, y_teste, 0.25, model)
    medidas.append(resultado25percent)
    stringMatrizes.append(cm25percent)

    #gerando um conjunto de 50% do treino
    resultado50percent, cm50percent = RF_TCC_GenerateReports(x_treino, y_treino, x_teste, y_teste, 0.5, model)
    medidas.append(resultado50percent)
    stringMatrizes.append(cm50percent)

    #gerando um conjunto de 75% do treino
    resultado75percent, cm75percent = RF_TCC_GenerateReports(x_treino, y_treino, x_teste, y_teste, 0.75, model)
    medidas.append(resultado75percent)
    stringMatrizes.append(cm75percent)

    #rodando o modelo com 100% do treino
    resultado100percent, cm100percent = RF_TCC_GenerateReports(x_treino, y_treino, x_teste, y_teste, 1.0, model)
    medidas.append(resultado100percent)
    stringMatrizes.append(cm100percent)

    indexSizeTest = {0: 1,
                     1: 2,
                     2: 5,
                     3: 10,
                     4: 25,
                     5: 50,
                     6: 75,
                     7: 100
                    }

    for i in range(len(stringMatrizes)):

        f.write('Confusion matrix with ' + str(indexSizeTest[i]) + '%' + ' of D_Treino]:\n')
        
        for j in range(len(cm1percent)):

            f.write(str(stringMatrizes[i][j]) + '\n')

        f.write('\n')

    for i in range(len(medidas)):
        accuracy.append(medidas[i]['accuracy'])
        f.write('Model scores with ' + str(indexSizeTest[i]) + '%' + ' of whole train dataset:\n')
        
        for key in medidas[i].keys():

            f.write(str(key) + ': ' + str(medidas[i][key]) + '\n')

    

    RF_TCC_CreateGraph([1, 2, 5, 10, 25, 50, 75, 100], accuracy, 'tamanho do conjunto de treino (%)', 'Acurácia (porção decimal)', tipoGrafico[tipo])

    f.write(str(cm1percent))
    f.write(str(resultado1percent))

    f.close()

def RF_TCC_FindBestParams(x, y):

    stringAcc = []

    print('Generating best set of hyperparameters for the model. . .')
    n_estimators = [int(x) for x in np.linspace(start = 5, stop = 100, num = 10)]

    max_depth = [2, 4]

    min_samples_split = [2, 4, 6]

    min_samples_leaf = [1, 2, 4]

    max_leaf_nodes = [2, 5, 10]

    max_features = ['auto', 'sqrt']

    param_grid = {'n_estimators': n_estimators,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'max_leaf_nodes': max_leaf_nodes,
                  'max_features': max_features
                  }

    model = RandomForestClassifier()

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)
    
    rf_grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = 'accuracy', refit = False, verbose = 0, n_jobs = -1)
    
    rf_grid.fit(x_treino, y_treino)

    str1 = 'Best set of hyperparameters obtained: ' + str(rf_grid.best_params_) + '\n'
    str2 = 'Accuracy obtained: ' + "{:.2f}".format(rf_grid.best_score_) + '\n\n'
    
    print(str1) 
    print(str2)

    stringAcc.append(str1)
    stringAcc.append(str2)

    return(rf_grid.best_params_, stringAcc)

def RF_TCC_CreateGraph(x, y, x_name, y_name, tipo):

    plt.xlabel(x_name)
    plt.ylabel(y_name)

    default_x_ticks = range(len(x))

    plt.plot(default_x_ticks, y, "o")

    plt.ylim([0.8, 1.0])
    plt.xticks(default_x_ticks, x)

    plt.title(tipo[2])

    plt.savefig(tipo[1])

    plt.close()