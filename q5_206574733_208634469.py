from PIL import Image
import numpy as np
import pickle
from sklearn.decomposition import PCA


def run(reduce_rank=True):
    """
    runs KNN according to the chosen mode: reduced rank or not. operates all other functions
    :param: reduce_rank: boolean parameter to determine if to reduce the rank of the points or not
    :return: None.
    """
    print("Using PCA:") if reduce_rank else print("Without PCA")
    for k in [200]:
        if reduce_rank:
            for s in [5, 10, 50, 100, 200]:
                KNN(k, reduce_rank, s)
        else:
            KNN(k)


def KNN(k, reduced_rank=False, s=1):
    """
    K-Nearest-Neighbors classifier
    predict_label a label to the points in the test set given a labeled train set.
    calculate the error rate
    :param k: number of neighbors
    :param reduced_rank: boolean parameter, boolean parameter to determine if to reduce the rank of the points or not
    :param s: if reduced_rank is True, s is the rank which the data matrix will be reduced to
    :return: None.
    """
    train_points, train_labels, test_points, test_labels = load_data()  # data in rows
    train_points = grayscale(train_points)
    test_points = grayscale(test_points)

    predictions = []
    if reduced_rank:
        # rank reduction
        pca = PCA(s)  # pca object
        pca.fit(train_points)
        Us = pca.components_.T  # the main s components of the data as columns
        train_points_reduced_rank = np.matmul(Us.T, train_points.T).T  # projected data in rows
        test_points_reduced_rank = np.matmul(Us.T, test_points.T).T  # projected data in rows
        # predict the label and calculate error rate
        for t_point in test_points_reduced_rank:
            predictions.append(predict_label(t_point, train_points_reduced_rank, train_labels, k))
        error_rate = calculate_error_rate(test_points_reduced_rank, test_labels, predictions)
        print(f"for k = {k}, s = {s}, the error rate is: {error_rate}")

    else:
        # predict the label and calculate error rate
        for t_point in test_points:
            predictions.append(predict_label(t_point, train_points, train_labels, k))
        error_rate = calculate_error_rate(test_points, test_labels, predictions)
        print(f"for k = {k}, the error rate is: {error_rate}")


def predict_label(test_point, train_set, train_label, k):
    """
    predict the label of a test point
    :param test_point: a test point to predict its label
    :param train_set: the train set, used to calculate the distances
    :param train_label: labels of the train set points
    :param k: number of nearest neighbors
    :return: the predicted label for test_point
    """
    # dictionary of each point from train set and its distance from the given test_point
    distances = {point_index: np.linalg.norm(p - test_point) for point_index, p in enumerate(train_set)}
    sorted_indices = sorted(distances.keys(), key=lambda x: distances[x])
    k_closest_neighbors_labels = [train_label[i] for i in sorted_indices[0:k]]
    label = max(set(k_closest_neighbors_labels), key=k_closest_neighbors_labels.count)
    return label


def unpickle(file):
    with open(file, 'rb') as fo:
        my_dict = pickle.load(fo, encoding='bytes')
        return my_dict


def load_data():
    """
    load the cifar-10 data from the files
    :return: train_set - 50,000 train-set samples
             train_labels - labels of training set
             test_set - 10,000 test-set samples
             test_labels - labels of test set
    """
    for i in range(1, 6):
        path = f"C:/Users/nimro/PycharmProjects/algebric methods 095295/206574733_208634469/cifar-10-batches-py/data_batch_{i}"
        batch = unpickle(path)
        if i == 1:
            train_set = (batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)).astype(
                'float32')
            train_labels = batch[b'labels']
        else:
            train_set_tmp = (batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)).astype(
                'float32')
            train_lbls_tmp = batch[b'labels']
            train_set = np.concatenate((train_set, train_set_tmp), axis=0)
            train_labels = np.concatenate((train_labels, train_lbls_tmp), axis=0)

    path = f"C:/Users/nimro/PycharmProjects/algebric methods 095295/206574733_208634469/cifar-10-batches-py/test_batch"
    batch = unpickle(path)
    test_set = (batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)).astype('float32')
    test_labels = batch[b'labels']
    # Please note that due to unreasonable run time we decided in order to answer the questions
    # to take only 1000 points from the training set and 100 points from the test.
    # return train_set, train_labels, test_set, test_labels
    return train_set[:1000, :], train_labels[:1000], test_set[:100, :], test_labels[:100]


def grayscale(set):
    """
    given a set of color pictures from cifar-10, make them grayscaled
    :param set: a set of cifar-10 pictures
    :return:
    """
    set_size = set.shape[0]
    set_dimensions = set.shape[:3]

    grayscaled = np.zeros(set_dimensions).astype('float32')
    for i in range(set_size):
        image = Image.fromarray(set[i].astype("uint8"))
        image = image.convert('L')
        grayscaled[i] = np.array(image)
    # change dimensions of the data s.t it will appear as a row
    data_as_rows = grayscaled.reshape(grayscaled.shape[0], -1)
    return data_as_rows


def calculate_error_rate(test_points, test_labels, predictions):
    """
    calculate the error rate of the KNN prediction
    :param test_points: all test points
    :param test_labels: all test true labels
    :param predictions: KNN label prediction for each data point
    :return: proportion of wrong predictions (:= error rate)
    """
    test_size = test_points.shape[0]
    missed = 0
    for i, y in enumerate(test_labels):
        if y != predictions[i]:
            missed += 1
    error_rate = missed / test_size
    return error_rate
