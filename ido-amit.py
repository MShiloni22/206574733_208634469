from PIL import Image
import numpy as np
import pickle
from sklearn.decomposition import PCA
from collections import Counter

NUM_COLORS = 3
COLORS = ["red", "green", "blue"]


def knn_for_cifar(k, s, use_pca):
    train_data, train_labels, test_data, test_labels = load_cifar()
    predictions = []
    U, _, _ = np.linalg.svd(train_data, full_matrices=False)
    Us = U[:, [0, s-1]]
    train_data_proj = np.matmul(Us.T, train_data)
    test_data_proj = np.matmul(Us.T, test_data)
    for i, x in enumerate(test_data_proj[:100, :]):
        predictions.append(predict(x, train_data_proj, train_labels, k))

    # if use_pca:
    #     pca = PCA(s)
    #     pca.fit(train_data)
    #     train_data_proj = np.matmul(pca.components_, train_data.transpose()).transpose()
    #     test_data_proj = np.matmul(pca.components_, test_data.transpose()).transpose()
    #     for i, x in enumerate(test_data_proj[:100, :]):
    #         predictions.append(predict(x, train_data_proj, train_labels, k))
    # else:
    #     for i, x in enumerate(test_data[:100, :]):
    #         predictions.append(predict(x, train_data, train_labels, k))

    wrong = 0
    for i, y in enumerate(test_labels[:100]):
        if y != predictions[i]:
            wrong += 1
    if use_pca:
        error = wrong / test_data_proj[:100, :].shape[0]
    else:
        error = wrong / test_data[:100, :].shape[0]
    print(f"for k = {k} and s = {s} the error = {error}")


def predict(test_point, train_data, train_labels, k):
    # mapping every point in the training data to its distance from a single test point, storing by indices
    distance_by_index = {point_index: np.linalg.norm(point - test_point)
                         for point_index, point in enumerate(train_data)}
    sorted_indices = sorted(distance_by_index.keys(), key=lambda x: distance_by_index[x])
    neighbors = [train_labels[i] for i in sorted_indices[:k]]
    counts = Counter(neighbors) # Counter returns a dict with the labels and the number of occurrences in list neighbors
    most_common = counts.most_common(1)[0][0]
    return most_common


def unpickle(file):
    with open(file, 'rb') as fo:
        my_dict = pickle.load(fo, encoding='bytes')
        return my_dict


def load_cifar():
    for i in range(1, 6):
        path = f"C:/Users/mshil/PycharmProjects/Semester_03/AlgebricMethods_Course/206574733_208634469/cifar-10-batches-py/data_batch_{i}"
        batch = unpickle(path)
        if i == 1:
            train_data = (batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)).astype('float32')
            train_labels = batch[b'labels']
        else:
            train_data_temp = (batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)).astype(
                'float32')
            train_labels_temp = batch[b'labels']
            train_data = np.concatenate((train_data, train_data_temp), axis=0)
            train_labels = np.concatenate((train_labels, train_labels_temp), axis=0)

    path = f"C:/Users/mshil/PycharmProjects/Semester_03/AlgebricMethods_Course/206574733_208634469/cifar-10-batches-py/test_batch"
    batch = unpickle(path)
    test_data = (batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)).astype('float32')
    test_labels = batch[b'labels']
    train_data_gray = np.zeros(train_data.shape[:3]).astype('float32')
    test_data_gray = np.zeros(train_data.shape[:3]).astype('float32')
    # convert into grayscale
    for i in range(train_data.shape[0]):
        image = Image.fromarray(train_data[i].astype("uint8"))
        image = image.convert('L')
        train_data_gray[i] = np.array(image)
    for i in range(test_data.shape[0]):
        image = Image.fromarray(test_data[i].astype("uint8"))
        image = image.convert('L')
        test_data_gray[i] = np.array(image)
    # flatten the arrays into 2d arrays
    x_train_flat = train_data_gray.reshape(train_data_gray.shape[0], -1)
    x_test_flat = test_data_gray.reshape(test_data_gray.shape[0], -1)

    return x_train_flat, train_labels, x_test_flat, test_labels


# if we will run this, we need to change main to this:
#from cifar_knn_w_pca import knn_for_cifar
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # for use_pca in [True, False]:
        use_pca = False
        for k in range(3, 54, 10):
            if use_pca:
                for s in range(5, 26, 5):
                    knn_for_cifar(k, s, use_pca)
            else:
                knn_for_cifar(k, 0, use_pca)