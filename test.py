import nashpy as nash
import itertools
import numpy as np


###############################################################################
##       生成状态控制空间，叉乘并利用set函数去除重复，获得全部战斗选择序列    ##
###############################################################################
# el = list(itertools.product([1, 2, 3, 4, 5, 6, 7, 8],
#                             (4, 5)))
# del_index = []
# for i, e in enumerate(el):
#     if len(list(e)) != len(set(e)):
#         del_index.append(i)
# for i in reversed(del_index):
#     del el[i]
# print(el)

# el = list(itertools.product())
# print(el)
class Unit():
    def __init__(self, force_class, name, sn, x0, y0, p0, w0) -> None:
        self.force_class = force_class  # B\R
        self.name = name  # 具体类型如RD
        self.sn = sn  # serial number 对应矩阵中的序列号、索引
        self.xy = (x0, y0)  # 位置
        self.xy_list = [(x0, y0)]
        self.p = p0  # 存活平台
        self.p_list = [p0, ]
        self.w = w0  # 武器数量 #!注意武器数量是1基的
        self.w_list = [w0, ]

        self.reloa_list = []  # relocate record
        self.target_list = []  # fire_target record
        self.salvo_list = []  # salvo size record

        self.ams = []  # list, available moving space, depending on the corridors and obstacles
        self.ats = []  # list, available target space, depending on the positions
        self.ass = []  # list, available salvo space, depending on the weapon storage


# def __split_level(temp_result):
#     # aim: 将不同层级的结果拆分为一个层级，而且转为了list
#     # input:  [((1, 3), 7), ((1, 3), 9), ((1, 5), 7), ((1, 5), 9), ((2, 3), 7), ((2, 3), 9), ((2, 5), 7), ((2, 5), 9)]
#     # output: [[1, 3, 7], [1, 3, 9], [1, 5, 7], [1, 5, 9], [2, 3, 7], [2, 3, 9], [2, 5, 7], [2, 5, 9]]
#     new_result = []
#     for i in range(len(temp_result)):
#         a = list(temp_result[i][0])
#         b = temp_result[i][1]
#         a.append(b)
#         new_result.append(a)
#     return new_result


# def __get_c_from_ats(ru):
#     # aim: 将原来只有目标的ats ，加上移动和salvo变成一个控制序列c
#     # input: [3,4,5]
#     # output: [ [0,3,1]  [0,3,2]  [0,4,1] [0,5,1] ]
#     temp_ats = []
#     for i in range(len(ru.ats)):
#         td_c = list()
#         td_c.append(0)
#         td_c.append(ru.ats[i])
#         temp_ats += ((list(itertools.product((td_c,), ru.ass))))
#     return __split_level(temp_ats)


# ru = Unit('r', "RT1", sn=0, x0=4, y0=4, p0=50, w0=6)
# ru.ats = [6, 7, 8]
# ru.ass = [1, 2]
# print(__get_c_from_ats(ru))

print(type(1) == int)

###############################################################################
#!#                          测试大型矩阵拼接。                                ##
###############################################################################
# r1_1 = np.zeros((2, 2))
# r1_2 = np.ones((2, 3))
# r2_1 = np.ones((1, 2))
# r2_2 = np.zeros((1, 3))

# r1 = np.concatenate((r1_1, r1_2), axis=1)
# r2 = np.concatenate((r2_1, r2_2), axis=1)
# r = np.concatenate((r1, r2), axis=0)


###############################################################################
##                        测试10的6次方级别的纳什均衡求解                       ##
###############################################################################
# r = np.random.uniform(-1, 1, (10, 5))
# print(r)
# rps = nash.Game(r)
# eqs = rps.support_enumeration()
# print(list(eqs)[0])

###########################################################################################
###                               np.where                                              ###
###########################################################################################
# a = np.array([
#     1, 2, 3
# ]
# )

# a = np.array([
#     [1, 2, 3],
#     [3, 4, 5]
# ])

# print(np.where(a == 3))
# print(np.where(a == 3)[0])
# print(a[np.where(a == 3)])

# a[np.where(a == 3)]=0
# print(a)

# print(np.array([[3, 4, 5], [3, 4, 5]]).T)
