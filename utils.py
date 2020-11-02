import numpy as np

def readFromTxt(filename):
    with open(filename) as file:
        matrix = []
        for line in file:
            array = []
            line = line.split()
            for vertex in line:
                array.append(float(vertex))
            matrix.append(array)
    return matrix

def col_normalize(mtx):
    """Row-normalize sparse matrix"""
    rowsum = np.sum(mtx, axis = 0)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.zeros((mtx.shape[1], mtx.shape[1]))
    for i in range(r_inv.shape[0]):
        r_mat_inv[i][i] = r_inv[i]
    mtx = mtx.dot(r_mat_inv)
    return mtx

def symm_normalize(mtx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mtx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.zeros((mtx.shape[0], mtx.shape[0]))
    for i in range(r_inv.shape[0]):
        r_mat_inv[i][i] = r_inv[i]
    mtx = r_mat_inv.dot(mtx)
    mtx = mtx.dot(r_mat_inv)
    return mtx

def read_env_data(adj, node_info, out):
    adj = np.array(readFromTxt(adj))
    edge_index = [[], []]
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if adj[i][j] > 0.0:
                adj[i][j]=1.0
                edge_index[0].append(i)
                edge_index[1].append(j)
    adj = symm_normalize(adj)

    node_info = np.array(readFromTxt(node_info))
    alloc = node_info[0]
    features = np.array(node_info[1:])
    features = np.stack(features, axis=0 )
    features = col_normalize(features)
    out = np.array(readFromTxt(out))

    return np.array(adj), np.array(alloc), np.array(features), out, edge_index
