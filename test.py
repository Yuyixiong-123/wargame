import nashpy as nash
import itertools


###############################################################################
##                          生成状态控制空间，叉乘并去除重复                    ##
###############################################################################
# el = list(itertools.product([1, 2, 3, 4, 5, 6, 7, 8],
#                             (4, 5)))
# del_index = []
# for i, e in enumerate(el):
#     if len(list(e)) != len(set(e)):
#         print(i, e)
#         del_index.append(i)
# for i in reversed(del_index):
#     print(i)
#     del el[i]
# print(el)

###############################################################################
##                          测试大型矩阵拼接。                                ##
###############################################################################
import numpy as np
r1_1 = np.zeros((2, 2))
r1_2 = np.ones((2, 3))
r2_1 = np.ones((1, 2))
r2_2 = np.zeros((1, 3))

r1 = np.concatenate((r1_1, r1_2), axis=1)
r2 = np.concatenate((r2_1, r2_2), axis=1)
r = np.concatenate((r1, r2), axis=0)


###############################################################################
##                        测试10的6次方级别的纳什均衡求解                       ##
###############################################################################
r = np.random.uniform(-1, 1, (10, 5))
print(r)
rps = nash.Game(r)
eqs = rps.support_enumeration()
print(list(eqs)[0])
