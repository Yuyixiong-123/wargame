import time
import numpy as np

dimension = int(2e4)
# A = np.random.dirichlet(np.ones(dimension), size=dimension)
# A = np.array(A, dtype=np.float16)
# np.save("filename.npy", A)
A = np.load("filename.npy")
A = np.concatenate((A, A, A, A), axis=1)
print(A.nbytes)
B = -A
# A = np.array([[-5, 0], [-8, -1]])
# B = np.array([[-5, -8], [0, -1]])


def get_max_index(a):
    #     仅返回索引列表中的第一个最大值索引
    # (array([2], dtype=int64),)  第一位元素才是索引，后面是类型
    index_set = np.where(a == np.max(a))[0]
    index = index_set[0]
    return index


class ARSA():
    def __init__(self, A, B) -> None:
        self.A = A
        self.B = B

    def search(self):
        for i in range(self.A.shape[0]):
            # B的行动控制列，找到让B响应最佳的列索引
            j = get_max_index(B[i, :])
            if A[i, j] == np.max(A[:, j]):
                # A的行动控制行，检查现在的i是不是让A响应最佳的行索引，如果是，则找到了纯策略纳什均衡
                return i, j, A[i, j], B[i, j]

    def stackelberg_search_Af(self):
        asna = A.shape[0]  # action space num of A
        dimen_reduc_A = np.ones(asna)
        for i in range(asna):
            if i % 3000 == 0:
                print(i)
            dimen_reduc_A[i] = A[i, get_max_index(B[i, :])]
        aa = get_max_index(dimen_reduc_A)  # action of A
        ab = get_max_index(B[aa, :])
        return aa, ab, dimen_reduc_A[aa], B[aa, ab]


if __name__ == '__main__':
    t1 = time.time()
    a = ARSA(A, B)
    aa = a.stackelberg_search_Af()
    print(aa)
    print(time.time()-t1)
