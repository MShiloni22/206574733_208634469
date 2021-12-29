import numpy as np
def PCA(X, num_components):
    X = X.astype('uint8')
    # Step-1
    X_meaned = (X - np.mean(X, axis=0).astype('uint8')).astype('uint8')
    print(X.dtype)
    # Step-2
    cov_mat = np.cov(X_meaned, rowvar=False, dtype='uint8')

    # Step-3
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat.astype('uint8'))

    # Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # Step-5
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # Step-6
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

    return X_reduced

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def main():
    print("ho")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
