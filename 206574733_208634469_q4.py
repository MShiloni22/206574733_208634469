from PIL import Image
import numpy as np

image = Image.open("unnamed.jpg")
pix = np.array(image)
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
k_list = [5, 80, 160, 200, 400]
errors = {}
for k in k_list:
    rgb_k_list = []
    for j in range(len(rgb_list)):
        M = rgb_list[j]
        U, S, V = np.linalg.svd(M, full_matrices=True, compute_uv=True)
        Uk_t = np.zeros((k, rows))
        Sk = np.zeros((k, k))
        Vk_t = np.zeros((k, cols))
        for i in range(k):
            Uk_t[i] = np.transpose(U)[i]
            Vk_t[i] = V[i]
            Sk[i][i] = S[i]
        Uk = np.transpose(Uk_t)
        Mk = np.matmul(Uk, np.matmul(Sk, Vk_t))
        rgb_k_list.append(Mk)
    pixels = np.zeros((rows, cols, depth)).astype(np.uint8)
    for i in range(rows):
        for j in range(cols):
            pixels[i][j][0] = rgb_k_list[0][i][j]
            pixels[i][j][1] = rgb_k_list[1][i][j]
            pixels[i][j][2] = rgb_k_list[2][i][j]
    new_image = Image.fromarray(pixels)
    new_image.show()
    errors[k] = np.linalg.norm(reds-rgb_k_list[0], 'fro')**2/np.linalg.norm(reds, 'fro')**2
for x in k_list:
    print(errors[x])