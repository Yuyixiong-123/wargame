
import logging.handlers
import logging
import json
import matplotlib.pyplot as plt
import math
import numpy as np
import PSO_NE as ne
import time
import random
import itertools
import pandas as pd

# columns = ['BB', 'BW1', 'BW2', 'RT1', 'RT2', 'RD1', 'RD2', 'RD3', 'FT']
# index = ['BB', 'BW1', 'BW2', 'RT1', 'RT2', 'RD1', 'RD2', 'RD3', 'FT']
# pk_df = pd.DataFrame(data=data, index=index, columns=columns, dtype=float)

# data = [
#     [0, 0, 0, 0.6, 0.6, 0.6, 0.5, 0.4, 0.3],
#     [0, 0, 0, 0, 0, 0.8, 0.7, 0.7, 0],
#     [0, 0, 0, 0, 0, 0.8, 0.7, 0.6, 0],
#     [0.2, 0.1, 0.1, 0, 0, 0, 0, 0, 0],
#     [0.2, 0.1, 0.1, 0, 0, 0, 0, 0, 0],
#     [0.7, 0.3, 0.3, 0, 0, 0, 0, 0, 0],
#     [0.5, 0.3, 0.2, 0, 0, 0, 0, 0, 0],
#     [0.5, 0.2, 0.2, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0]
# ]
###############################################################################
##                             configuration                                  ##
###############################################################################
max_salvo_size = 6

pk_br = [  # the kill probablities for Blue units killing red units
    [0.6, 0.6, 0.6, 0.5, 0.4, 0.3],
    [0, 0, 0.8, 0.7, 0.7, 0],
    [0, 0, 0.8, 0.7, 0.6, 0]]

pk_rb = [  # the kill probablities for Red units killing Blue units
    [0.2, 0.1, 0.1],
    [0.2, 0.1, 0.1],
    [0.7, 0.3, 0.3],
    [0.5, 0.3, 0.2],
    [0.5, 0.2, 0.2],
    [0, 0, 0]
]

relocate_move = {  # 使用编号索引来提取运动方向，按照中心开始的九宫格进行
    0: (0, 0),
    1: (0, -1),
    2: (1, -1),
    3: (1, 0),
    4: (1, 1),
    5: (0, 1),
    6: (-1, 1),
    7: (-1, 0),
    8: (-1, -1),
}


def split_integer(m, n):
    # 可以将整数m，按照最平均的要求，拆分为n份，返回拆分后的列表如[197, 197, 197, 198]
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n


def get_max_index(a):
    #     仅返回索引列表中的第一个最大值索引
    # (array([2], dtype=int64),)  第一位元素才是索引，后面是类型
    index_set = np.where(a == np.max(a))[0]
    index = index_set[0]
    return index


def stackelberg_search_Af(A, B):
    asna = A.shape[0]  # action space num of A
    dimen_reduc_A = np.ones(asna)
    for i in range(asna):
        # if i % 3000 == 0:
        #     logging.info('stackelberg_search_Af -- calculating{}'.format(i))
        dimen_reduc_A[i] = A[i, get_max_index(B[i, :])]
    aa = get_max_index(dimen_reduc_A)  # action of A
    ab = get_max_index(B[aa, :])    # action of B
    # return action_index and values
    return aa, ab, dimen_reduc_A[aa], B[aa, ab]


def tuple_add(tuple1, tuple2):
    zipped = zip(tuple1, tuple2)
    mapped = map(sum, zipped)
    return tuple(mapped)


def get_otherStyleTime():
    now = int(time.time())
    # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
    timeArray = time.localtime(now)
    otherStyleTime = time.strftime("%Y_%m_%d_%H_%M_%S", timeArray)
    return otherStyleTime


# config-log
log_filename = 'wargame.log'
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H-%M-%S %p"
fp = logging.handlers.RotatingFileHandler(log_filename, maxBytes=1024 * 1024 * 50,
                                          backupCount=30, encoding='utf-8')
fs = logging.StreamHandler()
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT,
                    datefmt=DATE_FORMAT, handlers=[fp, fs])

###############################################################################
##                             class of units                               ##
###############################################################################


class Unit():
    def __init__(self, force_class, name, sn, x0, y0, p0, w0) -> None:
        self.force_class = force_class  # B\R
        self.name = name  # 具体类型如RD
        self.sn = sn  # serial number 对应矩阵中的序列号、索引
        self.xy = (x0, y0)  # 在沙盘上的位置
        self.xy_list = [(x0, y0)]
        self.p = p0  # 存活平台
        self.p_list = [p0, ]
        self.w = w0  # 武器数量 #!注意武器数量是1基的
        self.w_list = [w0, ]

        self.reloa_list = []  # relocate record
        self.target_list = []  # fire_target record
        self.salvo_list = []  # salvo size record

        # list, available moving space, depending on the corridors and obstacles
        self.ams = [0]
        self.ats = []  # list, available target space, depending on the positions
        self.ass = []  # list, available salvo space, depending on the weapon storage

    def get_available_actions(self, Y_units):
        # 计算自己的可行移动集合，可行攻击集合，可行齐射数量：ams,ats,ass
        if self.name == 'FT':
            # FT不能移动，也没有攻击能力
            return None
        self.ams = []  # list, available moving space, depending on the corridors and obstacles
        self.ats = []  # list, available target space, depending on the positions
        self.ass = []  # list, available salvo space, depending on the weapon storage

        for i in range(9):
            # 测试是否各类移动是否可行。
            re_lo = tuple_add(self.xy, relocate_move[i])
            if re_lo[0] >= 0 and re_lo[0] <= WarEnv.map_size[0] and re_lo[1] >= 0 and re_lo[1] <= WarEnv.map_size[1]:
                self.ams.append(i)
        if self.w == 0:
            # 测试自己是否还有弹药，如果没有的话，ats,ass=[],直接结束
            return None

        for i in range(len(Y_units)):
            # 检测是否在同一格子里，代表是否可以攻击对方单位
            if self.xy == Y_units[i].xy:
                self.ats.append(Y_units[i].sn)

        #!debug 2023年2月9日19:29:10 发现之前都没有初始化ass……因为之前没有遇到过要打击的请款，所以都没有b
        for i in range(1, self.w+1):
            self.ass.append(i)

    def record_state(self):
        # 记录自身状态x,y,p,w，存入列表
        self.xy_list.append(self.xy)
        self.w_list.append(self.w)
        self.p_list.append(self.p)

    def record_control(self, c):
        # 记录自身控制变量，存入列表
        #! 控制变量C，由‘移动编码，目标对象，齐射武器数量’三个参数组成
        self.reloa_list.append(relocate_move[c[0]])  # relocate record
        self.target_list.append(c[1])  # fire_target record
        self.salvo_list.append(c[2])  # salvo size record


class WarEnv():
    map_size = (10, 10)  # the size of terrain map
    beta_w = 1  # 环境影响因子，0-1，好的环境取1，风雪不良天气取0
    stop_turn = 24  # 终止回合数

    def __init__(self) -> None:
        self.A_table = None
        self.q_table = None
        self.pij_table = None
        self.A_blue_survival_from_redAttack = None
        self.A_red_survival_from_blueAttack = None
        self.blue_force_units, self.red_force_units = self.gene_units()
        # 获得初始阶段各个单位的可行空间
        for bu in self.blue_force_units:
            bu.get_available_actions(self.red_force_units)
        for ru in self.red_force_units:
            ru.get_available_actions(self.blue_force_units)

        # 获得战斗参数表
        self.cal_action_para_tables()

        # 测试军力生成功能
        # self.blue_force_units[2].ats = [1, 2]
        # self.blue_force_units[0].ats = [1, 2]
        # print(self.__get_ma_seq())

    def gene_units(self):
        # 生成蓝军的三个单位
        BB = Unit('b', "BB", sn=0, x0=7, y0=6,  p0=10, w0=4)
        BW1 = Unit('b', "BW1", sn=1, x0=7, y0=6, p0=6, w0=4)
        BW2 = Unit('b', "BW2", sn=2, x0=7, y0=6,  p0=6, w0=4)
        blue_force_units = [BB, BW1, BW2]

        # 生成红军的6个单位，包含一个FT
        RT1 = Unit('r', "RT1", sn=0, x0=4, y0=4, p0=50, w0=6)
        RT2 = Unit('r', "RT2", sn=1, x0=4, y0=3, p0=50, w0=6)
        RD1 = Unit('r', "RD1", sn=2, x0=1, y0=1, p0=7, w0=3)
        RD2 = Unit('r', "RD2", sn=3, x0=1, y0=1, p0=7, w0=3)
        RD3 = Unit('r', "RD3", sn=4, x0=1, y0=1, p0=7, w0=3)
        FT = Unit('r', "FT", sn=5, x0=1, y0=1, p0=10, w0=0)
        red_force_units = [RT1, RT2, RD1, RD2, RD3, FT]

        return blue_force_units, red_force_units

    def cal_action_para_tables(self):
        # aim: 获得计算状态更新所需要的过程参数表 A，Q， Pij ，是全部对象可以共用的
        # input: self
        # output: 过程参数表 A

        # 定义三维的 A_red_survival_from_blueAttack,A_blue_survival_from_redAttack，先按照全部单位的最大齐射数max_salvo_size来定义矩阵大小，max_salvo_size需要人为简单计算得出
        A_blue_survival_from_redAttack = np.zeros(
            (len(self.blue_force_units), len(self.red_force_units), max_salvo_size))
        A_red_survival_from_blueAttack = np.zeros(
            (len(self.red_force_units), len(self.blue_force_units), max_salvo_size))

        # A is survival rate. BLUE'A depends on how red attack blue。这里是计算“如果打击”，则A是多少，如果如果选择了这样攻击的话，就从表里调用参数
        for i in range(len(self.blue_force_units)):
            for j in range(len(self.red_force_units)):
                for s in range(max_salvo_size):
                    # ! 在这里的s是从0开始的，所以后面有加一减一，其实这里s弄成1~max_salvo_size更好一些。
                    if s > (self.red_force_units[j].w - 1):
                        # 如果没有那么多武器，则不可能使用这个A值，所以就不用算了
                        continue
                    Qij = 1*(1 -
                             math.exp(- self.red_force_units[j].p / self.blue_force_units[i].p))
                    Pij = 1-pow((1-self.beta_w * pk_rb[j][i]), s+1)  # pk_rb
                    A_blue_survival_from_redAttack[i][j][s] = 1 - Qij * Pij

        # A is survival rate. RED'A depends on how blue attack red
        for i in range(len(self.red_force_units)):
            for j in range(len(self.blue_force_units)):
                for s in range(max_salvo_size):
                    if s > (self.blue_force_units[j].w - 1):
                        # 如果没有那么多武器，则不可能使用这个A值，所以就不用算了
                        continue
                    Qij = 1*(1 -
                             math.exp(- self.blue_force_units[j].p / self.red_force_units[i].p))
                    Pij = 1-pow((1-self.beta_w * pk_br[j][i]), s+1)  # pk_br
                    A_red_survival_from_blueAttack[i][j][s] = 1 - Qij * Pij

        self.A_blue_survival_from_redAttack = A_blue_survival_from_redAttack
        self.A_red_survival_from_blueAttack = A_red_survival_from_blueAttack

    def cal_next_state(self, cb, cr):
        # 对红蓝军选择的动作cb,cr，通过调用cal_action_para_tables，来计算各单位下一阶段的状态。
        # input: 各单位现在的状态，采用的下一步行动
        # output: 计算J所需要的各单位数量，列表
        #! cb，cr是字典数据结构，如{1：[0,3,4]} key为单位sn号，value为控制序列（移动，攻击对象，攻击波次）;therefor cb[bu.sn][0]代表移动方向编码。cr[ci][1]，cr[ci][2]代表攻击对象和salvo

        for bu in (self.blue_force_units):
            # 对每个blue unit更新状态，首先更新自己的移动状态,和武器数量
            move_tag = cb[bu.sn][0]
            bu.xy = tuple_add(bu.xy, relocate_move[move_tag])
            bu.w -= cb[bu.sn][2]

            # 更新自己的存活状态，即平台数
            for ci in cr.keys():
                # test whether some red-unit attack bu
                if cr[ci][1] == bu.sn:
                    #! debug记录 之前因为默认是0而非none，所以第一个单位会因此收到攻击
                    Ak = self.A_blue_survival_from_redAttack[bu.sn][ci][cr[ci][2]]
                    bu.p = round(bu.p * Ak)

            # 每个单位更新自己的状态，记录
            bu.record_state()
            bu.record_control(cb[bu.sn])

        for ru in (self.red_force_units):
            # 对每个red unit更新状态，首先更新自己的移动状态,和武器数量
            move_tag = cr[ru.sn][0]
            ru.xy = tuple_add(ru.xy, relocate_move[move_tag])
            ru.w -= cr[ru.sn][2]

            for ci in cb.keys():
                # test whether some blue-unit attack ru
                if cb[ci][1] == ru.sn:
                    Ak = self.A_red_survival_from_blueAttack[ru.sn][ci][cb[ci][2]]
                    ru.p = round(ru.p * Ak)

            # 每个单位更新自己的状态，记录
            ru.record_state()
            ru.record_control(cr[ru.sn])

        # 更新自己在下一阶段的可行空间。
        for bu in self.blue_force_units:
            bu.get_available_actions(self.red_force_units)
        for ru in self.red_force_units:
            ru.get_available_actions(self.blue_force_units)

        # 更新一下A表
        self.cal_action_para_tables()

    def cal_performance_index(self):
        #! aim: 计算J，后来已经不用了，因为主要是评估的eva函数，是最新版的加了距离评价的
        # input: 各单位数量
        # output: 由权重序列加权算的J：performance_index

        # 论文给出的权重体系
        blue_weight = [0.8, 0.5, 0.5, -0.1, -0.1, -0.3, -0.3, -0.2, -1]
        red_weight = [-0.7, -0.4, -0.3, 0.1, 0.1, 0.7, 0.5, 0.5, 1]

        # 计算蓝红双方的相对存活率
        survival_relative_num = []
        for bu in self.blue_force_units:
            survival_relative_num.append(bu.p / bu.p_list[0])
        for ru in self.rlue_force_units:
            survival_relative_num.append(ru.p / ru.p_list[0])

        # 计算存活效能指数
        J_b = np.sum(np.array(blue_weight).dot(
            np.array(survival_relative_num)))
        J_r = np.sum(np.array(red_weight).dot(np.array(survival_relative_num)))

        return J_b, J_r

    def eval_cr_cb(self, cb, cr):
        # 对现在的状态，和输入的任意一组（包含b，r force中全部单位的一次动作选择）控制序列，给出评分jr，jb。
        # 相当于是将cal_next_state，cal_performance_index两个函数，【在不执行的情况下】，融合在一起。但是仅考虑数量的变化。

        # 各个单位相对初始状态的存活率
        #! debug 如果没有攻击的话，survival_relative_num一直没有添加元素(2023年2月9日18:30:34)
        survival_relative_num = []

        for bu in (self.blue_force_units):
            # 测试计算bu在可能的cb,cr下的存活状态，即平台数
            be_attacked = False
            for ci in cr.keys():
                # test whether some red-unit attack bu
                if cr[ci][1] == bu.sn:
                    be_attacked = True
                    Ak = self.A_blue_survival_from_redAttack[bu.sn][ci][cr[ci][2]]
                    survival_relative_num.append(
                        round(bu.p * Ak) / bu.p_list[0])
                    break
            if not be_attacked:
                survival_relative_num.append(bu.p / bu.p_list[0])

        for ru in (self.red_force_units):
            # 测试计算ru在可能的cb,cr下的存活状态，即平台数
            be_attacked = False
            for ci in cb.keys():
                # test whether some blue-unit attack ru
                if cb[ci][1] == ru.sn:
                    be_attacked = True
                    Ak = self.A_red_survival_from_blueAttack[ru.sn][ci][cb[ci][2]]
                    survival_relative_num.append(
                        round(ru.p * Ak) / ru.p_list[0])
                    break
            if not be_attacked:
                survival_relative_num.append(ru.p / ru.p_list[0])

        # 计算权重
        blue_weight = [0.8, 0.5, 0.5, -0.1, -0.1, -0.3, -0.3, -0.2, -1]
        red_weight = [-0.7, -0.4, -0.3, 0.1, 0.1, 0.7, 0.5, 0.5, 1]

        # 想计算红蓝集群中心的相对距离
        sum_posi_blue = (0, 0)
        sum_posi_red = (0, 0)
        for bu in self.blue_force_units:
            sum_posi_blue = tuple_add(sum_posi_blue, bu.xy)
        for ru in self.red_force_units:
            sum_posi_red = tuple_add(sum_posi_red, ru.xy)
        # 红蓝群的相对中心位置
        aver_posi_blue = np.array(sum_posi_blue) / len(self.blue_force_units)
        aver_posi_red = np.array(sum_posi_red) / len(self.red_force_units)

        # a = np.square(np.array((0, 1)) - np.array((2, 1)))     ====> norm_dist = 0.1353
        # a = np.square(np.array((9, 1)) - np.array((2, 1)))     ====> norm_dist = 0.0009
        norm_dist = math.exp(- np.sqrt(np.sum(np.square(aver_posi_blue - aver_posi_red))))

        # 计算存活,距离效能指数
        J_b = (np.sum(np.array(blue_weight).dot(
            np.array(survival_relative_num)))) * norm_dist
        J_r = np.sum(np.array(red_weight).dot(np.array(survival_relative_num)))

        return J_b, J_r

    def __get_ma_seq(self):
        #! aim: 根据k阶段下，红蓝两军的情况，给出两军各自的行动【分类】序列
        # input: 红蓝两军units
        # output:两个字典对 所组成的列表
        # [('mmm', 'mmmmmm'), ('mma', 'mmmmmm'), ('amm', 'mmmmmm'), ('ama', 'mmmmmm')]

        rb_act_code_pair_list = []  # 最终得到的要返回的叉积结果，如上
        r_act_code_list = []  # 红方可能行动分类，如：mmm,mma
        b_act_code_list = []  # 蓝方可能行动分类，如：mmmmm,mmamm

        # 获得红军的动作序列
        for i in range(len(self.red_force_units)):
            temp_bri = 'm'
            if len(self.red_force_units[i].ats) > 0:  # 看是否能攻击
                temp_bri = 'ma'

            if i == 0:  # 第一次叉积时act_code_list没东西，就用一个字符串就行
                r_act_code_list = temp_bri
            else:
                r_act_code_list = list(
                    itertools.product(r_act_code_list, temp_bri))
                # 这里使用了列表拼接为字符串，从而使得循环进行叉积，避免了（（m,m）,m）的问题
                for j in range(len(r_act_code_list)):
                    r_act_code_list[j] = ''.join(r_act_code_list[j])

        # 获得蓝军的动作序列
        for i in range(len(self.blue_force_units)):
            temp_bri = 'm'
            if len(self.blue_force_units[i].ats) > 0:  # 看是否能攻击
                temp_bri = 'ma'

            if i == 0:  # 第一次叉积时act_code_list没东西，就用一个字符串就行
                b_act_code_list = temp_bri
            else:
                b_act_code_list = list(
                    itertools.product(b_act_code_list, temp_bri))
                for j in range(len(b_act_code_list)):
                    # 这里使用了列表拼接为字符串，从而使得循环进行叉积，避免了（（m,m）,m）的问题
                    b_act_code_list[j] = ''.join(b_act_code_list[j])

        # 叉乘红蓝两军的动作序列
        rb_act_code_pair_list = list(
            itertools.product(b_act_code_list, r_act_code_list))
        return rb_act_code_pair_list

    def __split_level(self, temp_result):
        # aim: 将不同层级的结果拆分为一个层级，而且转为了list
        # input:  [((1, 3), 7), ((1, 3), 9), ((1, 5), 7), ((1, 5), 9), ((2, 3), 7), ((2, 3), 9), ((2, 5), 7), ((2, 5), 9)]
        # output: [[1, 3, 7], [1, 3, 9], [1, 5, 7], [1, 5, 9], [2, 3, 7], [2, 3, 9], [2, 5, 7], [2, 5, 9]]
        new_result = []
        for i in range(len(temp_result)):
            a = list(temp_result[i][0])
            b = temp_result[i][1]
            a.append(b)
            new_result.append(a)
        return new_result

    def __var_list_product(self, lst):
        # aim: 获得长度不定的list中各元素的叉乘
        # input: E.g. [(7, 9), (3, 5),(7, 9), (3, 5)]
        # output: [[1, 3, 7, 3], [1, 3, 7, 5], [1, 3, 9, 3], [1, 3, 9, 5], [1, 5, 7, 3], [1, 5, 7, 5], [1, 5, 9, 3], [1, 5, 9, 5], [2, 3, 7, 3], [2, 3, 7, 5], [2, 3, 9, 3], [2, 3, 9, 5], [2, 5, 7, 3], [2, 5, 7, 5], [2, 5, 9, 3], [2, 5, 9, 5]]

        long = len(lst)

        if long > 2:
            temp_result = list(itertools.product(lst[0], lst[1]))
            for i in range(1, long-1):
                # 需要循环迭代连乘
                temp_result = list(itertools.product(temp_result, lst[i+1]))
                temp_result = self.__split_level(temp_result)
            return temp_result
        elif long == 2:
            return list(itertools.product(lst[0], lst[1]))
        elif long == 1:
            return list(lst[0])

    def __get_c_from_ats(self, ru):
        # aim: 将原来只有目标的ats ，加上移动和salvo变成一个控制序列c
        # input: ru， where ru.ats=[3,4,5]
        # output: [ [0,3,1]  [0,3,2]  [0,4,1] [0,5,1] ]
        temp_ats = []
        for i in range(len(ru.ats)):
            td_c = list()
            td_c.append(0)  # 因为是有目标的，所以移动就肯定是0，这里的0代表移动编码
            td_c.append(ru.ats[i])
            temp_ats += ((list(itertools.product((td_c,), ru.ass))))
        return self.__split_level(temp_ats)

    def __get_c_from_move(self, u):
        # aim: 将原来只有移动编码的ams ，加上为-1的目标和为0的salvo变成一个控制序列c
        # input: u where u.ams=[3,4,5]
        # output: [ [3,0,0]  [4,0,0]  [5,0,0] ]
        # ! debug 勘误，默认的不攻击状态不能是0，得是-1，否则就会默认攻击sn为0的目标
        temp_move = []
        for i in range(len(u.ams)):
            td_c = [-1, 0]
            td_c.insert(0, u.ams[i])
            temp_move.append(td_c)
        return temp_move

    def get_cr_list(self, act_type='aammm'):
        # aim: 根据可能的code如mmmmm，生成全部可能的cr控制序列
        # input: 如amm这种控制序列
        # output:[cr1,cr2,...]每一个cri是一个字典，cri中的key为sn，value为一个单位的控制序列[4,0,0]

        attack_units = []  # 保存选择攻击的单位sn
        move_units = []  # 保存选择移动的单位的sn
        for i in range(len(act_type)):
            if act_type[i] == 'a':
                attack_units.append(i)
            else:
                move_units.append(i)

        # 联合生成全部选择攻击的单位，所形成的cri的列表，如[ [(0,1,1),(0,2,1)]，  [(0,1,1),(0,2,2)]， ... ]
        # 有2个选择attack的ru时，att_units_action其数据结构为：[([0, 3, 1], [0, 6, 1]), ([0, 3, 1], [0, 6, 2]), ([0, 4, 1], [0, 6, 1]), ([0, 4, 1], [0, 6, 2])]
        att_units_action = []
        if len(attack_units) > 0:
            # 有3个选择attack的ru，ats_list就有3个元素，每个元素都是一个列表(ats_list.append(temp_c))，temp_c中的一个元素是该ru的一个控制序列如[0, 6, 2]
            ats_list = []
            for sn in attack_units:
                temp_c = self.__get_c_from_ats(self.red_force_units[sn])
                ats_list.append(temp_c)
            att_units_action = self.__var_list_product(ats_list)
            if type(att_units_action[0][0]) == int:
                # 此处考虑仅单个元素选择...action仅为2层嵌套列表的情况，则需要增加维度
                att_units_action = [att_units_action]

        # 联合生成全部选择移动的单位，所形成的cri的列表，如 [ [(1,0,0)]，  [(2,0,0)]， ... ]
        # move_units_action的数据结构为 [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [1, 0, 0]],...]
        move_units_action = []
        if len(move_units) > 0:
            move_list = []  # 有5个ru，就有5个元素（列表）：temp_c，每个元素temp_c是该单位的控制向量的集合
            for sn in move_units:
                temp_c = self.__get_c_from_move(self.red_force_units[sn])
                move_list.append(temp_c)
            move_units_action = self.__var_list_product(move_list)
            if type(move_units_action[0][0]) == int:
                # 此处考虑仅单个元素选择...action仅为2层嵌套列表的情况，则需要增加维度
                move_units_action = [move_units_action]
        # print(move_units_action)

        product_att_move = []
        # 移动命令可能性，与开火命令可能性，叉乘之后组合起来的列表.其中的每一个元素，3个move2个attack时，product_att_move数据结构如下：([[0, 0, 0], [0, 0, 0], [0, 0, 0]], ([0, 3, 1], [0, 6, 1]))
        cr_list = []
        # 列表，每一个元素都是一个地点，代表r_units的集体控制命令，每个元素的数据结构如下：{2: [0, 0, 0], 3: [2, 0, 0], 4: [4, 0, 0], 0: [0, 4, 1], 1: [0, 6, 1]}

        #! 需要对mmmma的组合形式做分类讨论，这涉及到product_att_move内的列表层级问题

        if len(move_units) > 0 and len(attack_units) > 0:
            # 针对同时有m和a的情况，需要有product_att_move，里面层级多一些
            product_att_move = list(itertools.product(
                move_units_action, att_units_action))
            for i in range(len(product_att_move)):
                one_c = {}
                for m_sn in range(len(move_units)):
                    one_c[move_units[m_sn]] = product_att_move[i][0][m_sn]
                for att_sn in range(len(attack_units)):
                    one_c[attack_units[att_sn]] = product_att_move[i][1][att_sn]
                cr_list.append(one_c)

        elif len(attack_units) == 0:
            # 针对仅有m的情况，有move_units_action就行了，里面层级少一些
            for i in range(len(move_units_action)):
                one_c = {}
                for m_sn in range(len(move_units)):
                    one_c[move_units[m_sn]] = move_units_action[i][m_sn]
                cr_list.append(one_c)

        elif len(move_units) == 0:
            # 针对仅有m的情况，有att_units_action就行了，里面层级少一些
            for i in range(len(att_units_action)):
                one_c = {}
                for att_sn in range(len(attack_units)):
                    one_c[attack_units[att_sn]] = att_units_action[i][att_sn]
                cr_list.append(one_c)
        # print(cr_list[2])
        return cr_list

    def get_cb_list(self, act_type='amm'):
        # aim: 根据可能的code如amm，生成全部可能的cb控制序列
        # input: 如amm这种控制序列
        # output:[cb1,cb2,...]每一个cbi是一个字典
        #! 详细注释思想见上get_cr_list

        attack_units = []  # 保存选择攻击的单位sn
        move_units = []  # 保存选择移动的单位的sn
        for i in range(len(act_type)):
            if act_type[i] == 'a':
                attack_units.append(i)
            else:
                move_units.append(i)

        # 联合生成全部选择攻击的单位，所形成的cbi的列表，如[ [(0,1,1),(0,2,1)]，  [(0,1,1),(0,2,2)]， ... ]
        # att_units_action其数据结构为：[([0, 3, 1], [0, 6, 1]), ([0, 3, 1], [0, 6, 2]), ([0, 4, 1], [0, 6, 1]), ([0, 4, 1], [0, 6, 2])]
        att_units_action = []
        if len(attack_units) > 0:
            ats_list = []  # 有2个bu，就有2个元素（列表）：temp_c，每个元素temp_c是该单位的控制向量的集合
            for sn in attack_units:
                temp_c = self.__get_c_from_ats(self.blue_force_units[sn])
                ats_list.append(temp_c)
            att_units_action = self.__var_list_product(ats_list)

            if type(att_units_action[0][0]) == int:
                # 此处考虑仅单个元素选择...action仅为2层嵌套列表的情况，则需要增加维度
                att_units_action = [att_units_action]

        # 联合生成全部选择移动的单位，所形成的cri的列表，如 [ [(1,0,0)]，  [(2,0,0)]， ... ]
        # move_units_action的数据结构为 [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [1, 0, 0]],...]
        move_units_action = []
        if len(move_units) > 0:
            move_list = []  # 有2个bu，就有2个元素（列表）：temp_c，每个元素temp_c是该单位的控制向量的集合
            for sn in move_units:
                temp_c = self.__get_c_from_move(self.blue_force_units[sn])
                move_list.append(temp_c)
            move_units_action = self.__var_list_product(move_list)

            if type(move_units_action[0][0]) == int:
                # 此处考虑仅单个元素选择...action仅为2层嵌套列表的情况，则需要增加维度
                move_units_action = [move_units_action]
        # print(move_units_action)

        product_att_move = []
        # 移动命令可能性，与开火命令可能性，叉乘之后组合起来的列表.其中的每一个元素，数据结构如下：([[0, 0, 0], [0, 0, 0], [0, 0, 0]], ([0, 3, 1], [0, 6, 1]))
        cb_list = []
        # 列表，每一个元素都是一个地点，代表r_units的集体控制命令，每个元素的数据结构如下：{2: [0, 0, 0], 3: [2, 0, 0], 4: [4, 0, 0], 0: [0, 4, 1], 1: [0, 6, 1]}

        if len(move_units) > 0 and len(attack_units) > 0:
            # 针对同时有m和a的情况，需要有product_att_move，里面层级多一些
            product_att_move = list(itertools.product(
                move_units_action, att_units_action))
            for i in range(len(product_att_move)):
                one_c = {}
                for m_sn in range(len(move_units)):
                    one_c[move_units[m_sn]] = product_att_move[i][0][m_sn]
                for att_sn in range(len(attack_units)):
                    one_c[attack_units[att_sn]] = product_att_move[i][1][att_sn]
                cb_list.append(one_c)

        elif len(attack_units) == 0:
            # 针对仅有m的情况，有move_units_action就行了，里面层级少一些
            for i in range(len(move_units_action)):
                one_c = {}
                for m_sn in range(len(move_units)):
                    one_c[move_units[m_sn]] = move_units_action[i][m_sn]
                cb_list.append(one_c)

        elif len(move_units) == 0:
            # 针对仅有m的情况，有att_units_action就行了，里面层级少一些
            for i in range(len(att_units_action)):
                one_c = {}
                for att_sn in range(len(attack_units)):
                    one_c[attack_units[att_sn]] = att_units_action[i][att_sn]
                cb_list.append(one_c)

        # print(cb_list[21])

        return cb_list

    def get_matrixBlock_SE(self, cb_list_block, cr_list_block):
        # aim: 计算由原来的cblist，srlist中分割出来的分块控制序列所组成的矩阵，并且计算效能值，最终给出stackelberg equilibrium
        # input: 分割后的cb_list_block, cr_list_block
        # output: 一个stackelberg equilibrium 对：（cb,cr序列，stackelberg equilibrium得分）

        # debug 这里是随机选择了100个，先做调试用，2023年2月9日19:32:37发现如果分割得“好”，会让len(cb_list_block)<101，绝了！
        cb_list_block = random.sample(
            cb_list_block, min(1000, len(cb_list_block)))
        cr_list_block = random.sample(
            cr_list_block, min(1000, len(cr_list_block)))

        cb_blength = len(cb_list_block)
        cr_blength = len(cr_list_block)

        # 收益矩阵，矩阵博弈计算的出发点
        b_payoff = np.zeros((cb_blength, cr_blength), dtype=np.float16)
        r_payoff = np.zeros((cb_blength, cr_blength), dtype=np.float16)

        for bi in range(cb_blength):
            if bi > 1 and bi % 100 == 1:
                # display the progress
                logging.info('cal payoff matrix(row): {}|{}'.format(bi, cb_blength))
            for rj in range((cr_blength)):
                """ print(cb_list_block[bi], cr_list_block[rj])：{0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]} {0: [2, 0, 0], 1: [6, 0, 0], 2: [6, 0, 0], 3: [2, 0, 0], 4: [4, 0, 0], 5: [7, 0, 0]} """
                # 依次抽取cb,cr，计算效能值，形成A,B矩阵，计算分区子矩阵的stackelberg equilibrium
                J_b, J_r = self.eval_cr_cb(
                    cb_list_block[bi], cr_list_block[rj])
                b_payoff[bi][rj] = J_b
                r_payoff[bi][rj] = J_r

        cb_index, cr_index, bpf, rpf = stackelberg_search_Af(
            b_payoff, r_payoff)  # blue pay off, red pay off
        se_control, se_value = (
            {'cb': cb_list_block[cb_index], 'cr': cr_list_block[cr_index]}, bpf)
        # 因为最后需要作比较的是blue的效能值，默认stackelberg equilibrium是Blue先走的，所以这里只返回bpf

        return se_control, se_value

    def get_cb_cr_1stepGT(self):
        # aim: 优选智能博弈的双方策略，这个策略首先用随机选择的，然后
        # input: 当前各单位状态
        # output: 优选出来的智能博弈的双方策略

        # [('mmm', 'mmmmmm'), ('mma', 'mmmmmm'), ('amm', 'mmmmmm'), ('ama', 'mmmmmm')]
        act_type_set = self.__get_ma_seq()

        # stackelberg equilibrium 求解出来的分块矩阵的控制向量和效能值。
        se_control_list = []
        se_value_list = []

        for (act_type_BF, act_type_RF) in act_type_set:
            # 获得blue force的全部动作序列，cb_list中的单个元素是blue force中各单位的一个控制向量组成的字典。
            cb_list = self.get_cb_list(act_type_BF)
            cb_length = len(cb_list)

            # 获得red force的全部动作序列，cb_list中的单个元素是blue force中各单位的一个控制向量组成的字典。
            cr_list = self.get_cr_list(act_type_RF)
            cr_length = len(cr_list)

            partition_n = math.ceil(cb_length * cr_length / 4e8)  # 矩阵分割数
            #! 采取分割红方控制序列的方式来做计算，将整个大的矩阵分成partition_n份，每一份对应的分块矩阵中的数据总数小于4e8
            pr_num_list = split_integer(
                cr_length, partition_n)  # pr_num_list格式如 [4,4,3]，partition_n=3时
            start_num = 0
            for pn in range(partition_n):
                cr_list_block = cr_list[start_num: start_num+pr_num_list[pn]]
                start_num = start_num+pr_num_list[pn]

                se_control, se_value = self.get_matrixBlock_SE(
                    cb_list, cr_list_block)
                # 获得该分块矩阵对应的控制向量和效能值，注意这个效能值是blue force的，因为默认是蓝军有主动权
                se_control_list.append(se_control)  # se_control是一个字典，key为cb，cr
                se_value_list.append(se_value)

        # 从全部stackelberg equilibrium中获得全局最优，返回对应的控制向量cb,cr
        final_control = se_control_list[get_max_index(se_value)]
        return final_control['cb'], final_control['cr']

    def draw_units_comparison_plot(self, units):
        fig = plt.figure()
        fig.clear()

        nameList = []
        ori_list = []
        final_list = []
        for u in units:
            nameList.append(u.name)
            ori_list.append(u.p_list[0])
            final_list.append(u.p_list[-1])

        p_comparison = pd.DataFrame(
            index=nameList, columns=['original_number', 'final_number'])
        p_comparison['original_number'] = ori_list
        p_comparison['final_number'] = final_list
        p_comparison.plot(kind='bar')
        plt.title(units[0].force_class+' force units comparison plot')
        plt.savefig(
            units[0].force_class+' force units comparison plot'+get_otherStyleTime()+'.png')

    def record_and_draw(self):
        # aim: 保存各个单位的作战过程和数据
        # input: 各单位列表  self.red_force_units...
        # output: 存为文本，绘制轨迹和数量对比图

        record_json = {}  # 保存全部单位的储存数据,包括各时间节点上的状态变量和控制命令变量
        for u in (self.blue_force_units+self.red_force_units):
            record_json[u.name] = {'xy_list': u.xy_list,
                                   'p_list': u.p_list, 'w_list': u.w_list, 'salvo_list': u.salvo_list, 'reloa_list': u.reloa_list, 'target_list': u.target_list}
        with open('record_data_{}.json'.format(get_otherStyleTime()), 'w') as f:
            json.dump(record_json, f)

        # 构造一个index为0-1两个数，columns是各个单位名称的pd，然后就可以绘制单位前后对比图了
        self.draw_units_comparison_plot(self.red_force_units)
        self.draw_units_comparison_plot(self.blue_force_units)

        fig = plt.figure()
        fig.clear()
        for u in (self.blue_force_units+self.red_force_units):
            xlist = []
            ylist = []
            for i in range(len(u.xy_list)):
                xlist.append(u.xy_list[i][0])
                ylist.append(u.xy_list[i][1])
            plt.plot(xlist, ylist, label=u.name)
        plt.legend()
        plt.show()
        plt.savefig('units move path.png')

    def warring(self):
        # 战争进程模型
        for i in range(self.stop_turn):
            cb, cr = self.get_cb_cr_1stepGT()
            logging.info('第{}个回合：选择动作{},{}'.format(i, cb, cr))
            self.cal_next_state(cb, cr)
            if self.red_force_units[-1].p / self.red_force_units[-1].p_list[0] < 0.4:
                # 终止条件：红方最后一个单位FT的损毁程度大于0.6
                break
        self.record_and_draw()


if __name__ == '__main__':
    we = WarEnv()
    we.warring()
