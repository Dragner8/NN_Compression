import numpy as np
import torch
from sklearn.cluster import KMeans
def linear_tensor_quant(val, max, min, bits):
    val = np.minimum(val, max);

    val = np.maximum(val, min);

    data = (np.multiply(val, 2 ** bits / abs(max - min)))

    data = np.around(data)
    data = np.multiply(data, abs(max - min) / 2 ** bits)
    return data



def fixed_point_tensor_quant(data_in, word_len):
    clip_table = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                  131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456,
                  536870912, 1.0737418240e+009,  2.1474836480e+009,  4.2949672960e+009, 8.589934592000000e+09,
                  1.717986918400000e+10,
                  3.4359738636800000e+10, 6.871947673600000e+10, 1.374389534720000e+11, 2.748779069440000e+11,
                  5.497558138880000e+11,  1.099511627776000e+12, 2.199023255552000e+12, 4.398046511104000e+12,
                  8.796093022208000e+12];


    frac_len=word_len-1

    scale_factor = clip_table[frac_len];

    scale_factor_inv = 1. / scale_factor;


    data_out = np.around(np.multiply(data_in, scale_factor))

    sat_limit = clip_table[word_len];

    data_out = np.minimum(data_out, sat_limit - 1);
    data_out = np.maximum(data_out, 1 - sat_limit);

    data_out = np.multiply(data_out, scale_factor_inv);


    return data_out



def kmeans_tensor_quant(param, k=16, codebook=None, update_labels=False, **unused):

    import torch
    param_shape = param.size()
    num_el = param.numel()
    param_1d = param.view(num_el)
    if codebook is not None:
        param_1d[codebook.labels_ == 0] = 0

    if codebook is None :
        param_numpy = param_1d.cpu().detach().numpy()
        param_nz = param_numpy[param_numpy != 0]
        param_nz = param_nz.reshape(param_nz.size, 1)


        codebook = KMeans(n_clusters=k-1, init='k-means++', n_jobs=-1).fit(param_nz)  # one less cluster due to zero-fixed
        centers = codebook.cluster_centers_
        centers = np.append(0.0, centers)  # append zero as centroid[0]
        codebook.cluster_centers_ = centers.reshape(centers.size, 1)
        codebook.labels_ = codebook.predict(param_numpy.reshape(num_el, 1))
        codebook.cluster_centers_ = torch.from_numpy(codebook.cluster_centers_).float()
        codebook.labels_ = torch.from_numpy(codebook.labels_).long()
        if param.is_cuda:
            codebook.cluster_centers_ = codebook.cluster_centers_.cuda(param.device)



    else:
        if update_labels:
            sorted_centers, indices = torch.sort(codebook.cluster_centers_, dim=0)
            boundaries = (sorted_centers[1:] + sorted_centers[:-1]) / 2
            sorted_labels = torch.ge(param_1d - boundaries, 0).long().sum(dim=0)
            codebook.labels_ = indices.index_select(0, sorted_labels).view(num_el)
        for i in range(1, k):
            # not from (0, k), because we fix the zero centroid
            codebook.cluster_centers_[i, 0] = param_1d[codebook.labels_ == i].mean()

    param_quantize = codebook.cluster_centers_[codebook.labels_].view(param_shape)
    if not param.is_contiguous():
        param_quantize = param_quantize.contiguous()
    #param.set_(param_quantize)

    return codebook, param_quantize