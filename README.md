# TCC---GMORAES-ISANTOS

Este repositório contém o projeto do Trabalho de Conclusão de Curso de Gabriel Moraes da Silva e Ícaro Barros Santos Barcelos. O projeto utiliza o algoritmo ensemble de classificação Random Forest, dois .csv como entrada que contém informações de pacotes Wi-Fi enviados e recebidos como entrada para prever a posição de um dispositivo em um ambiente.

O código recebe como entrada um .csv disponibilizado a partir de outro TCC. 

Os arquivos .csv chamados "dataset-fixo-tcc-joao" e "dataset-mix-tcc-joao.csv" representam pacotes de sinal transmitidos do centro e dos cantos de um cômodo, respectivamente. Cada linha contém:

-"ID", um identificador único do pacote;
-"date_time", representa o horário que o pacote foi recebido;
-"device_signal", o nível de sinal de um roteador Wi-Fi em que o pacote foi recebido;
-"device_id", o nome do roteador Wi-Fi que recebeu o pacote;
-"id_addr", o código do cômodo onde foi transmitido o pacote;
-"pack_type", o tipo do pacote enviado.

Os arquivos .py chamados "fix_set" e "mix_set" tratam os dados dos arquivos .csv e retornam X e y, que servirão para treinar e testar o algoritmo de Random Forest.

O arquivo .py chamado "RandomForest" contém a implementação do modelo classificador Random Forest, geração de gráficos para melhor visualização dos resultados do modelo, geração de métricas avaliativas e a geração de arquivos .txt com as mesmas.
