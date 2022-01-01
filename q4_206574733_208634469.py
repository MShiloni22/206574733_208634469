from PIL import Image
import numpy as np


def run():
    image = Image.open("q4_images/world_map_original.jpg")
    pix = np.array(image)

    # create and fill the RGB matrices
    (rows, cols, depth) = pix.shape
    rows = int(rows)
    cols = int(cols)
    depth = int(depth)
    reds = np.zeros((rows, cols))
    greens = np.zeros((rows, cols))
    blues = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            reds[i][j] = pix[i][j][0]
            greens[i][j] = pix[i][j][1]
            blues[i][j] = pix[i][j][2]

    rgb_list = [reds, greens, blues]
    k_list = [5, 80, 160, 320, 640]
    errors = {}
    for k in k_list:
        rgb_k_list = []  # here we will save the approximations of Reds, Greens and Blues
        for M in rgb_list:
            U, S, V = np.linalg.svd(M, full_matrices=True, compute_uv=True)
            # build the k-rank approximations for U, S, V
            Uk_t = np.zeros((k, rows))
            Sk = np.zeros((k, k))
            Vk_t = np.zeros((k, cols))
            for i in range(k):
                Uk_t[i] = np.transpose(U)[i]
                Vk_t[i] = V[i]
                Sk[i][i] = S[i]
            Uk = np.transpose(Uk_t)
            # build the k-rank approximations for M
            Mk = np.matmul(Uk, np.matmul(Sk, Vk_t))
            rgb_k_list.append(Mk)

        # union approximations of Reds, Greens and Blues to form a new picture
        pixels = np.zeros((rows, cols, depth)).astype(np.uint8)
        for i in range(rows):
            for j in range(cols):
                pixels[i][j][0] = rgb_k_list[0][i][j]
                pixels[i][j][1] = rgb_k_list[1][i][j]
                pixels[i][j][2] = rgb_k_list[2][i][j]
        new_image = Image.fromarray(pixels)
        # new_image.save("C:/Users/mshil/PycharmProjects/Semester_03/AlgebricMethods_Course/206574733_208634469/q4_images/world_map_"+str(k)+".jpg")
        new_image.show()

        # save the relative error of the Reds matrix, for each value k
        errors[k] = np.linalg.norm(reds - rgb_k_list[0], 'fro') ** 2 / np.linalg.norm(reds, 'fro') ** 2

    for k in k_list:
        print(f"for k={k}: relative error is {errors[k]}")
