import torch
import numpy as np

def calculate_mask(layer,percent=90):
    w = layer.weight.detach().numpy()
    w = np.absolute(w)
    threshold = np.percentile(np.array(w), percent)

    bool_mask = w > threshold
    mask = torch.from_numpy(bool_mask.astype(int)).type(torch.FloatTensor)
    return mask