import numpy as np
import pandas as pd
from keras import backend as K


# define a simple dataloader
def load_data(path):
    columns = ["V1a", "V1v", "V1c", "w", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12",
               "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20"]
    # columns = ["V1a", "V1v", "V1c", "w", "meshVolume","relativeVolume", "surfaceArea","relativeArea", "thickness","poreDiameter",
    #            "thicknessAM","areaMean", "areaMin", "areaMax","s1", "s2", "s3","s4", "s5", "s6", "s7", "s8", "s9", "s10",
    #            "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19","s20"]
    # columns = ["V1a", "V1v", "V1c", "w", "relativeVolume", "relativeArea", "thickness", "poreDiameter", "areaMean",
    #            "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16",
    #            "s17", "s18", "s19", "s20"]

    filename = path
    data1_1 = pd.read_excel(filename, sheet_name='class1', skiprows=None, names=columns)
    data1_2 = pd.read_excel(filename, sheet_name='class2', skiprows=None, names=columns)
    data1_3 = pd.read_excel(filename, sheet_name='class12', skiprows=None, names=columns)
    data1 = pd.concat([data1_1, data1_2, data1_3], axis=0)
    # data1.pop('s1')  # as the data feature s1 is very small and close to 0
    data2 = data1.to_numpy(dtype=np.float32)
    # np.random.shuffle(data2)
    return data2


# define the loss function and metrics mean absolute error (MAE)
def loss_function(y_true, y_pred):
    return float(K.mean(K.square(y_true - y_pred)))


def loss_mae(y_true, y_pred):
    return float(K.mean(K.abs(y_true - y_pred)))
