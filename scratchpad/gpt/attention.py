from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax

# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])


# generating the weight matrices
random.seed(42) # to allow us to reproduce the same attention values
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))

# generating the queries, keys and values
query_1 = word_1 @ W_Q
key_1 = word_1 @ W_K
value_1 = word_1 @ W_V

query_2 = word_2 @ W_Q
key_2 = word_2 @ W_K
value_2 = word_2 @ W_V

query_3 = word_3 @ W_Q
key_3 = word_3 @ W_K
value_3 = word_3 @ W_V

query_4 = word_4 @ W_Q
key_4 = word_4 @ W_K
value_4 = word_4 @ W_V


# scoring the first query vector against all key vectors
scores = array([dot(query_1, key_1), dot(query_1, key_2), dot(query_1, key_3), dot(query_1, key_4)])
#print(scores)


# computing the weights by a softmax operation
weights = softmax(scores / key_1.shape[0] ** 0.5)
print(weights)
print(value_1)
print(value_2)
print(value_3)
print(value_4)

print(weights[0] * value_1)
print(weights[1] * value_2)
print(weights[2] * value_3)
print(weights[3] * value_4)
# computing the attention by a weighted sum of the value vectors
attention = (weights[0] * value_1) + (weights[1] * value_2) + (weights[2] * value_3) + (weights[3] * value_4)

print(attention)