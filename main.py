from PIL import Image
import numpy as np
import pickle
import time


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def unpickle(file):
    with open(file, 'rb') as fo:
        batch_dict = pickle.load(fo,  encoding='bytes')
    return batch_dict


def get_train_data():
    file_path_1 = r"C:/Users/mshil/PycharmProjects/Semester_03/AlgebricMethods_Course/206574733_208634469/cifar-10-batches-py/data_batch_1"
    batch_dict_1 = unpickle(file_path_1)
    all_data = batch_dict_1[b'data']
    all_labels = batch_dict_1[b'labels']
    for i in range(2, 6):
        file_path = r"C:/Users/mshil/PycharmProjects/Semester_03/AlgebricMethods_Course/206574733_208634469/cifar-10-batches-py/data_batch_"+str(i)
        batch_dict = unpickle(file_path)
        data = batch_dict[b'data']
        labels = batch_dict[b'labels']
        all_data = np.concatenate((all_data, data), axis=0)
        all_labels = all_labels + labels
    all_data = data_gray(all_data)
    all_data = normalize(all_data).astype('uint8')
    return all_data.T, all_labels


def normalize(data):
    mean = np.mean(data, axis=1)
    train_data_normalized = np.apply_along_axis(lambda x: x-mean, 0, data)
    return train_data_normalized


def get_test_data():
    file_path = r"C:/Users/mshil/PycharmProjects/Semester_03/AlgebricMethods_Course/206574733_208634469/cifar-10-batches-py/test_batch"
    batch_dict = unpickle(file_path)
    test_data = batch_dict[b'data']
    test_labels = batch_dict[b'labels']
    test_data = data_gray(test_data)
    return test_data.T, test_labels


def data_gray(data):
    data_list = []
    for single_img in data:
        single_img_reshaped = np.transpose(np.reshape(single_img, (3, 32, 32)), (1, 2, 0))
        image = Image.fromarray(single_img_reshaped.astype('uint8'))
        image = image.convert("L")
        my_array = np.array(image)
        my_array = np.reshape(my_array, (1, -1))
        data_list.append(my_array)
    X = np.asarray(data_list)
    X = X[:, 0, :]
    return X


def create_distances_vector(train, test_point):
    inner_products = np.matmul(train.T, test_point)
    norms = np.square(np.linalg.norm(train, axis=0))
    distances_vec = norms - 2*inner_products
    return distances_vec


def find_k_closest(distances_vector, labels, k_list):
    idx = np.argsort(distances_vector)
    idx = list(idx)
    label_per_k = dict()
    for k in k_list:
        k_closest = idx[:k]
        k_labels = []
        for train_point in k_closest:
            k_labels.append(labels[train_point])
        test_label = most_common(k_labels)
        label_per_k[k] = test_label
    return label_per_k


def most_common(labels_list):
    return max(set(labels_list), key=labels_list.count)


def calculate_errors(prediction_dict, k_list, s_list,  test_labels):
    errors_per_run = dict()
    for k in k_list:
        for s in s_list:
            errors = 0
            for i in range(len(test_labels)):
                if prediction_dict[s, k, i] != test_labels[i]:
                    errors += 1
            errors_per_run[s, k] = errors/len(test_labels)
    return errors_per_run


if __name__ == '__main__':
    start_overall = time.time()
    s_list = [1, 5, 10, 25, 50, 100, 500, 1020]
    k_list = [1, 3, 5, 9, 15, 50, 100, 200]
    train_data, train_labels = get_train_data()
    test_data, test_labels = get_test_data()
    predictions_dict = dict()
    U, _, _ = np.linalg.svd(train_data, full_matrices=False)
    end_data_extraction = time.time()
    time_per_s = dict()
    for s in s_list:
        start_s = time.time()
        Us = U[:, :s]
        train_proj = np.matmul(Us.T, train_data)
        test_proj = np.matmul(Us.T, test_data)
        inner_products = np.matmul(train_proj.T, test_proj)
        norms = np.square(np.linalg.norm(train_proj, axis=0))
        end_distance_cal = time.time()
        for i in range(test_proj.shape[1]):
            distances = -2*inner_products[:, i] + norms
            prediction = find_k_closest(distances, train_labels, k_list)
            for k_value in prediction:
                predictions_dict[s, k_value, i] = prediction[k_value]
        end_s = time.time()
        time_per_s[s] = end_s - start_s
    error_dict = calculate_errors(predictions_dict, k_list, s_list, test_labels)
    print(error_dict)
    end_overall = time.time()
