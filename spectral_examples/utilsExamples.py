import numpy as np

def stack_lower_triangular_columns(matrix):
    n = matrix.shape[0]
    lower_triangular = np.tril(matrix)
    stacked_columns = []
    for j in range(n):
        for i in range(j, n):
            stacked_columns.append(lower_triangular[i, j])
    return np.array(stacked_columns)

def scale_off_diagonal(matrix, scale_factor):
    assert np.allclose(matrix, matrix.T), "Matrix must be symmetric"
    diag = np.diag(matrix)
    off_diag_scaled = matrix.copy()
    np.fill_diagonal(off_diag_scaled, 0)  
    off_diag_scaled *= scale_factor
    scaled_matrix = off_diag_scaled + np.diag(diag)
    return scaled_matrix

def vectorize(matrix):
    return stack_lower_triangular_columns(scale_off_diagonal(matrix, np.sqrt(2)))

def unstack_lower_triangular_columns(stacked_columns, n):
    lower_triangular = np.zeros((n, n))
    index = 0
    for col in range(n):
        for row in range(col, n):
            lower_triangular[row, col] = stacked_columns[index]
            index += 1
    
    return lower_triangular

def unvectorize(stacked_columns, n):
    lower_triangular = unstack_lower_triangular_columns(stacked_columns, n)
    diag = np.diag(lower_triangular)
    unscaled_lower_triangular = lower_triangular / np.sqrt(2)
    np.fill_diagonal(unscaled_lower_triangular, diag)
    symmetric_matrix = unscaled_lower_triangular + unscaled_lower_triangular.T - np.diag(diag)
    return symmetric_matrix


def logdet(X):
    L = np.linalg.cholesky(X)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return logdet