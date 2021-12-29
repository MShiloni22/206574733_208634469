from PIL import Image
import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# if it works, we need to add all other batch files to the list
files_list = [r"C:\Users\mshil\PycharmProjects\Semester_03\AlgebricMethods_Course\206574733_208634469\cifar-10-batches-py\data_batch_1"]
s_list = [5]
k_list = [5]
train_list = []
reduced_data_dict = {} # will hold the reduced data for knn for each s value

# step1: turning all data to grayscale, arranging data in one matrix
for file in files_list:
    my_dict = unpickle(file)
    all_data = my_dict[b'data']
    all_labels = my_dict[b'labels']
    # all_data contains all the images in a batch now. Single image can be
    # accessed by the number of the row containing it
    for i in range(len(all_data)):
        single_img = np.array(all_data[i])
        single_img_reshaped = np.transpose(np.reshape(single_img, (3, 32, 32)), (1, 2, 0))
        image = Image.fromarray(single_img_reshaped.astype('uint8'))
        image = image.convert("L")
        my_array = np.array(image)
        my_vector = my_array.reshape((1, 32*32))
        train_list.append(my_vector)
train_t = np.concatenate(train_list, axis=0)
train = np.transpose(train_t)

# step2: PCA and decreasing data's dimension, running knn per each s in s_list
for s in s_list:
    Us_t = np.zeros((s, 1024))
    train_copy = np.matrix(train)
    for i in range(s):
        U, _, _ = np.linalg.svd(train_copy, full_matrices=False, compute_uv=True)
        next_u = np.transpose(U)[0]
        Us_t[i] = next_u
        train_copy = np.matmul((np.identity(1024) - (np.matmul(np.transpose(next_u), next_u))), train_copy)
    train_reduced = np.zeros((s, len(train[1].astype(int))))
    for i in range(len(train[1])):
        train_reduced[:, i] = np.matmul(Us_t, train[:, i])
    reduced_data_dict[s] = train_reduced

