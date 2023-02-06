import numpy as np
import scipy.sparse
import torch
import torch.nn as nn

import src.modeling.data.config as cfg
import src.modeling.data.config as cfg


def scipy_to_pytorch(A, U, D):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    ptU = []
    ptD = []

    for i in range(len(U)):
        u = scipy.sparse.coo_matrix(U[i])
        i = torch.LongTensor(np.array([u.row, u.col]))
        v = torch.FloatTensor(u.data)
        ptU.append(torch.sparse.FloatTensor(i, v, u.shape))

    for i in range(len(D)):
        d = scipy.sparse.coo_matrix(D[i])
        i = torch.LongTensor(np.array([d.row, d.col]))
        v = torch.FloatTensor(d.data)
        ptD.append(torch.sparse.FloatTensor(i, v, d.shape))

    return ptU, ptD


def adjmat_sparse(adjmat, nsize=1):
    """Create row-normalized sparse graph adjacency matrix."""
    adjmat = scipy.sparse.csr_matrix(adjmat)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i, i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    row = adjmat.row
    col = adjmat.col
    data = adjmat.data
    i = torch.LongTensor(np.array([row, col]))
    v = torch.from_numpy(data).float()
    adjmat = torch.sparse.FloatTensor(i, v, adjmat.shape)
    return adjmat


def get_graph_params(filename, nsize=1):
    """Load and process graph adjacency matrix and upsampling/downsampling matrices."""
    data = np.load(filename, encoding="latin1", allow_pickle=True)
    A = data["A"]
    U = data["U"]
    D = data["D"]
    U, D = scipy_to_pytorch(A, U, D)
    A = [adjmat_sparse(a, nsize=nsize) for a in A]
    return A, U, D


def test():
    filename = cfg.MANO_sampling_matrix

    nsize = 1
    print(filename)
    A, U, D = get_graph_params(filename=filename, nsize=nsize)
    print(U[0])


import torch
import torch.onnx
import torch.sparse
from torch.nn import Module


def main():
    i = [
        [
            0,
        ],
        [
            0,
        ],
    ]
    v = [
        42,
    ]
    s = torch.sparse_coo_tensor(i, v, (2, 2))
    print(s)

    # def forward(self, input: torch.Tensor) -> torch.Tensor:
    m = torch.tensor([[2, 0], [0, 3]])
    print(m)
    a = torch.sparse.mm(s, m)
    print(a)


main()
