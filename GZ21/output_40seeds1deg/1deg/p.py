import numpy as np
d = np.load('spatial_pred_seed_4011.npz')
for k in d.keys():
    print(k, d[k].shape)
