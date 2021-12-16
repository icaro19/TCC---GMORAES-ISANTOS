from randomForest import RF_TCC_DifferentTrainSizesPredict, RF_TCC_PredictMeans

from fix_set import FP_TCC_CreateFixFingerprints
from mix_set import FP_TCC_CreateMixFingerprints

#modelo do vetor de sinal: sala, quarto 1, quarto 2, serviço, cozinha

x, y = FP_TCC_CreateFixFingerprints()

RF_TCC_PredictMeans(x, y, 0)

x_mix, y_mix = FP_TCC_CreateMixFingerprints()

RF_TCC_PredictMeans(x, y, 1)

x.extend(x_mix)

y.extend(y_mix)

RF_TCC_PredictMeans(x, y, 2)