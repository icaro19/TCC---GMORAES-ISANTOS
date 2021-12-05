import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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
    accuracy = []

    #obtendo os melhores hiper-parâmetros para o nosso modelo
    best_hyperparams = RF_TCC_FindBestParams(x, y)

    #criando o modelo com os hiper-parâmetros obtidos
    model = RandomForestClassifier(n_estimators = best_hyperparams['n_estimators'],
                                   max_depth = best_hyperparams['max_depth'],
                                   min_samples_leaf = best_hyperparams['min_samples_leaf'],
                                   min_samples_split = best_hyperparams['min_samples_split'],
                                   max_leaf_nodes = best_hyperparams['max_leaf_nodes'],
                                   max_features = best_hyperparams['max_features']
                                   )

    #gerando o treino e teste básicos
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

    #gerando um conjunto de 1% do treino
    x_treino1percent, x_testeDescartável, y_treino1percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.01, shuffle = False, stratify = None)
    model.fit(x_treino1percent, y_treino1percent)
    resultado1percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado1percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado1percent))
    
    #gerando um conjunto de 2% do treino
    x_treino2percent, x_testeDescartável, y_treino2percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.02, shuffle = False, stratify = None)
    model.fit(x_treino2percent, y_treino2percent)
    resultado2percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado2percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado2percent))

    #gerando um conjunto de 5% do treino
    x_treino5percent, x_testeDescartável, y_treino5percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.05, shuffle = False, stratify = None)
    model.fit(x_treino5percent, y_treino5percent)
    resultado5percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado5percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado5percent))

    #gerando um conjunto de 10% do treino
    x_treino10percent, x_testeDescartável, y_treino10percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.1, shuffle = False, stratify = None)
    model.fit(x_treino10percent, y_treino10percent)
    resultado10percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado10percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado10percent))

    #gerando um conjunto de 25% do treino           
    x_treino25percent, x_testeDescartável, y_treino25percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.25, shuffle = False, stratify = None)
    model.fit(x_treino25percent, y_treino25percent)
    resultado25percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado25percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado25percent))

    #gerando um conjunto de 50% do treino
    x_treino50percent, x_testeDescartável, y_treino50percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.50, shuffle = False, stratify = None)
    model.fit(x_treino50percent, y_treino50percent)
    resultado50percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado50percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado50percent))

    #gerando um conjunto de 75% do treino
    x_treino75percent, x_testeDescartável, y_treino75percent, y_testeDescartável = train_test_split(x_treino, y_treino, train_size = 0.75, shuffle = False, stratify = None)
    model.fit(x_treino75percent, y_treino75percent)
    resultado75percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado75percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado75percent))

    #rodando o modelo com 100% do treino
    model.fit(x_treino, y_treino)
    resultado100percent = model.predict(x_teste)
    medidas.append(classification_report(y_teste, resultado100percent, output_dict = True))
    stringMedidas.append(classification_report(y_teste, resultado100percent))

    indexSizeTest = {0: 1,
                     1: 2,
                     2: 5,
                     3: 10,
                     4: 25,
                     5: 50,
                     6: 75,
                     7: 100
                    }

    for i in range(len(medidas)):
        
        accuracy.append(medidas[i]['accuracy'])
        print('Model scores with ', indexSizeTest[i], '%', 'of whole train dataset:\n' ,stringMedidas[i])

    RF_TCC_CreateGraph([1, 2, 5, 10, 25, 50, 75, 100], accuracy, 'tamanho do conjunto de teste (%)', 'Acurácia (porção decimal)')
        
def RF_TCC_FindBestParams(x, y):


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

    print('Best set of hyperparameters obtained: ', rf_grid.best_params_) 
    print('Accuracy obtained: ', "{:.2f}".format(rf_grid.best_score_))

    return(rf_grid.best_params_)

def RF_TCC_CreateGraph(x, y, x_name, y_name):

    plt.xlabel(x_name)
    plt.ylabel(y_name)

    default_x_ticks = range(len(x))

    plt.plot(default_x_ticks, y)

    plt.ylim([0.5, 1.0])
    plt.xticks(default_x_ticks, x)

    plt.show()