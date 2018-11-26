import numpy as np

def cmvnOperation(feature):
    if type(feature) == np.ndarray:
        feature = np.array(feature)
        
    N = feature.shape[0]
    mean_m = np.sum(feature, axis=0) / N
    std_m = np.sqrt(np.sum(pow(feature, 2), axis=0)/ N - pow(mean_m, 2) )
    
    for column in range(feature.shape[1]):
        if std_m[column] != 0:
            feature[:, column] = (feature[:, column] - mean_m[column]) / std_m[column]
        else:
            feature[:, column] = feature[:, column] - mean_m[column]
    return feature
