import sys
import numpy as np

# first_arg = sys.argv[1]
# second_arg = sys.argv[2]


# def greetings(word1=first_arg, word2=second_arg):
#     print(type(int(word1)))
#     print(type(int(word2)))
#     # print("{} {}".format(word1, word2))



# # greetings()
# # greetings("Bonjour", "monde")
# t = np.array(['hi', int(9)])
# print(type(t[0]))
# print(type(t[1]))
# np.random.seed(999999999)
# print(np.random.randint(low = 0, high = 100, size = 5))

# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder(handle_unknown='ignore')
# X = [['Male', 1], ['Female', 3], ['Female', 2]]
# X_array = np.asarray(X)
# print(X_array.shape)
# enc.fit(X)
# print(enc.transform(X).toarray())

# idx_to_flip = np.array([3, 6, 9, 1])
# minority_idx_to_flip = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
# minority_idx_to_flip = minority_idx_to_flip[idx_to_flip]
# Y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# Y[minority_idx_to_flip] = 1 - Y[minority_idx_to_flip]
# print(Y)
# print(minority_idx_to_flip)

list = [1]

for item in list:
    print(item)