import numpy as np


def values(hist_dataset, hist_im_query):
    nim_dataset, nvals_dataset = np.shape(hist_dataset)
    nims, nvals_query = np.shape(hist_im_query)

    if nvals_query != nvals_dataset:
        print("Dimensions don't match")

    hist_im_query = np.tile(hist_im_query, (nim_dataset, 1))

    return hist_im_query


def euclidean(hist_dataset, hist_im_query):
    hist_im_query = values(hist_dataset, hist_im_query)
    distance = np.sum(np.sqrt((hist_dataset - hist_im_query) ** 2), axis=1)

    return distance


def distance_l(hist_dataset, hist_im_query):
    hist_im_query = values(hist_dataset, hist_im_query)
    distance = np.sum(np.absolute(hist_dataset - hist_im_query), axis=1)

    return distance


def distance_x2(hist_dataset, hist_im_query):
    hist_im_query = values(hist_dataset, hist_im_query)
    distance = np.sum(np.power((hist_dataset - hist_im_query), 2) / (hist_dataset + hist_im_query), axis=1)

    return distance


def intersection(hist_dataset, hist_im_query):
    hist_im_query = values(hist_dataset, hist_im_query)
    distance = np.sum(np.minimum(hist_dataset, hist_im_query), axis=1)
    return distance


def hellinger(hist_dataset, hist_im_query):
    hist_im_query = values(hist_dataset, hist_im_query)
    distance = np.sum(np.sqrt(hist_dataset*hist_im_query), axis = 1)
    return distance


def kl_divergence(hist_dataset, hist_im_query):
    hist_im_query = values(hist_dataset, hist_im_query)
    distance = np.sum(hist_im_query*np.log(hist_im_query/hist_dataset), axis = 1)
    return distance


def shannon_entropy(var):
    return -np.sum(var*np.log2(var), axis = 1)


def js_divergence(hist_dataset, hist_im_query):
    hist_im_query = values(hist_dataset, hist_im_query)
    distance = shannon_entropy(1/2*(hist_dataset+hist_im_query))-1/2*(shannon_entropy(hist_dataset)+shannon_entropy(hist_im_query))
    return distance


def calculate_distances(hist_dataset, hist_im_query, mode='euclidean'):
    if mode == 'euclidean':
        return euclidean(hist_dataset, hist_im_query)
    if mode == 'distance_L':
        return distance_l(hist_dataset, hist_im_query)
    if mode == 'distance_x2':
        return distance_x2(hist_dataset, hist_im_query)
    if mode == 'intersection':
        return intersection(hist_dataset, hist_im_query)
    if mode == 'kl_divergence':
        return kl_divergence(hist_dataset, hist_im_query)
    if mode == 'js_divergence':
        return js_divergence(hist_dataset, hist_im_query)
    if mode == 'hellinger':
        return hellinger(hist_dataset, hist_im_query)