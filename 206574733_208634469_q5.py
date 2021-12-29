from PIL import Image
import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

files_list = [r"C:\Users\mshil\PycharmProjects\Semester_03\AlgebricMethods_Course\206574733_208634469\cifar-10-batches-py\data_batch_1"]
s_list = [5]
k_list = [5]
train_list = []
for file in files_list:
    my_dict = unpickle(file)
    img = my_dict[b'data']
    # img contains all the images in a batch now. Single image can be
    # accessed by the number of the row containing it
    for i in range(len(img)):
        single_img = np.array(img[i])
        single_img_reshaped = np.transpose(np.reshape(single_img, (3, 32, 32)), (1, 2, 0))
        image = Image.fromarray(single_img_reshaped.astype('uint8'))
        image = image.convert("L")
        my_array = np.array(image)
        my_vector = my_array.reshape((1, 32*32))
        train_list.append(my_vector)
train_t = np.concatenate(train_list, axis=0)
train = np.transpose(train_t)
for s in s_list:
    MIC_t = np.zeros((s, 1024))
    train_copy = np.matrix(train)
    for i in range(s):
        U, _, _ = np.linalg.svd(train_copy, full_matrices=False, compute_uv=True)
        next_u = np.transpose(U)[0]
        MIC_t[i] = next_u
        train_copy = np.matmul((np.identity(1024) - (np.matmul(np.transpose(next_u), next_u))), train_copy)
    MIC = np.transpose(MIC_t)