import torch
import torch.nn as nn

A = torch.randn(2, 3)
print(A)
print(A.shape)
B = torch.randn(3, 4)
print(B)
print(B.shape)
result = torch.einsum('ij,jk->ik', A, B)
print(result)
print(result.shape)

C = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print(C)
print(C.shape)

D = torch.tensor([
    [[1, 2, 3], [3, 4, 5]]
])
print(D)
print(D.shape)

embedding = nn.Embedding(26, 100)

input = torch.LongTensor([[1, 2], [2, 3], [3, 4],[1, 2], [2, 3], [3, 4]])

print (input)

print(embedding(input))
