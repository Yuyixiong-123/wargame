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


###############################################################################
##                             class of units                               ##
###############################################################################
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

    def get_available_actions(self, Y_units):
        # 计算自己的可行移动集合，可行攻击集合，可行齐射数量：ams,ats,ass
        self.ams = []  # list, available moving space, depending on the corridors and obstacles
        self.ats = []  # list, available target space, depending on the positions
        self.ass = []  # list, available salvo space, depending on the weapon storage

        for i in range(9):
            # 测试是否各类移动是否可行。
            re_lo = self.xy+relocate_move[i]
            if re_lo[0] >= 0 and re_lo[0] <= WarEnv.map_size[0] and re_lo[1] >= 0 and re_lo[1] <= WarEnv.map_size[1]:
                self.ams.append(i)
        if self.w == 0:
            # 测试自己是否还有弹药，如果没有的话，ats=[],直接结束
            return None

        for i in range(len(Y_units)):
            # 检测是否在同一格子里，代表是否可以攻击对方单位
            if self.xy == Y_units[i].xy:
                self.ats.append(Y_units[i].sn)

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

    def update(self):
        # 通过调取cal_action_para_tables如Q,A表来计算自己下一阶段的状态
        pass


class WarEnv():
    map_size = (10, 10)  # the size of terrain map
    beta_w = 1  # 环境影响因子，0-1，好的环境取1，风雪不良天气取0

    def __init__(self) -> None:
        self.A_table = None
        self.q_table = None
        self.pij_table = None
        self.A_blue_survival_from_redAttack = None
        self.A_red_survival_from_blueAttack = None
        self.blue_force_units, self.red_force_units = self.gene_units()
        for bu in self.blue_force_units:
            bu.get_available_actions(self.red_force_units)
        for ru in self.red_force_units:
            ru.get_available_actions(self.blue_force_units)
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

        # 生成红军的6个单位，包含一个
        RT1 = Unit('r', "RT1", sn=0, x0=4, y0=4, p0=50, w0=6)
        RT2 = Unit('r', "RT2", sn=1, x0=4, y0=3, p0=50, w0=6)
        RD1 = Unit('r', "RD1", sn=2, x0=1, y0=1, p0=7, w0=3)
        RD2 = Unit('r', "RD2", sn=3, x0=1, y0=1, p0=7, w0=3)
        RD3 = Unit('r', "RD3", sn=4, x0=1, y0=1, p0=7, w0=3)
        FT = Unit('r', "RT1", sn=5, x0=1, y0=1, p0=1, w0=0)
        red_force_units = [RT1, RT2, RD1, RD2, RD3, FT]

        return blue_force_units, red_force_units

    def cal_action_para_tables(self):
        # aim: 获得计算状态更新所需要的过程参数表 A，Q， Pij ，是全部对象可以重复共用的
        # input: self
        # output: 过程参数表 A

        # 定义三维的 A_red_survival_from_blueAttack,A_blue_survival_from_redAttack
        A_blue_survival_from_redAttack = np.zeros(
            (len(self.blue_force_units), len(self.red_force_units), max_salvo_size))
        A_red_survival_from_blueAttack = np.zeros(
            (len(self.red_force_units), len(self.blue_force_units), max_salvo_size))

        # A is survival rate. BLUE'A depends on how red attack blue
        for i in range(len(self.blue_force_units)):
            for j in range(len(self.red_force_units)):
                for s in range(max_salvo_size):
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
            bu.xy = bu.xy+relocate_move[move_tag]
            bu.w -= cb[bu.sn][2]

            # 更新自己的存活状态，即平台数
            for ci in cr.keys():
                # test whether some red-unit attack bu
                if cr[ci][1] == bu.sn:
                    Ak = self.A_blue_survival_from_redAttack[bu.sn][ci][cr[ci][2]]
                    bu.p = round(bu.p * Ak)

            # 每个单位更新自己的状态，记录
            bu.record_state()
            bu.record_control(cb[bu.sn])

        for ru in (self.red_force_units):
            # 对每个red unit更新状态，首先更新自己的移动状态,和武器数量
            move_tag = cr[ru.sn][0]
            ru.xy = ru.xy+relocate_move[move_tag]
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
            bu.get_available_actions()
        for ru in self.red_force_units:
            ru.get_available_actions()

    def cal_performance_index(self):
        # aim: 计算J
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
        # 对现在的状态，和输入的任意一组控制序列，给出评分jr，jb。相当于是将cal_next_state，cal_performance_index两个函数，在不执行的情况下，融合在一起。但是仅考虑数量的变化。
        survival_relative_num = []

        for bu in (self.blue_force_units):
            # 测试计算bu在可能的cb,cr下的存活状态，即平台数
            for ci in cr.keys():
                # test whether some red-unit attack bu
                if cr[ci][1] == bu.sn:
                    Ak = self.A_blue_survival_from_redAttack[bu.sn][ci][cr[ci][2]]
                    survival_relative_num.append(
                        round(bu.p * Ak) / bu.p_list[0])

        for ru in (self.red_force_units):
            # 测试计算ru在可能的cb,cr下的存活状态，即平台数
            for ci in cb.keys():
                # test whether some blue-unit attack ru
                if cb[ci][1] == ru.sn:
                    Ak = self.A_red_survival_from_blueAttack[ru.sn][ci][cb[ci][2]]
                    survival_relative_num.append(
                        round(ru.p * Ak) / ru.p_list[0])

        blue_weight = [0.8, 0.5, 0.5, -0.1, -0.1, -0.3, -0.3, -0.2, -1]
        red_weight = [-0.7, -0.4, -0.3, 0.1, 0.1, 0.7, 0.5, 0.5, 1]
        # 计算存活效能指数
        J_b = np.sum(np.array(blue_weight).dot(
            np.array(survival_relative_num)))
        J_r = np.sum(np.array(red_weight).dot(np.array(survival_relative_num)))

        return J_b, J_r

    def __get_ma_seq(self):
        # aim: 根据k阶段下，红蓝两军的情况，给出两军各自的行动分类序列
        # input: 红蓝两军units
        # output:两个字典对 所组成的列表，如{0:m,1:m,2:a}
        # [('mmm', 'mmmmmm'), ('mma', 'mmmmmm'), ('amm', 'mmmmmm'), ('ama', 'mmmmmm')]

        rb_act_code_pair_list = []
        r_act_code_list = []
        b_act_code_list = []

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
                for i in range(len(r_act_code_list)):
                    r_act_code_list[i] = ''.join(r_act_code_list[i])

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
                for i in range(len(b_act_code_list)):
                    # 这里使用了列表拼接为字符串，从而使得循环进行叉积，避免了（（m,m）,m）的问题
                    b_act_code_list[i] = ''.join(b_act_code_list[i])

        # 叉乘红蓝两军的动作序列
        rb_act_code_pair_list = list(
            itertools.product(b_act_code_list, r_act_code_list))
        return rb_act_code_pair_list

    def __get_cr_list(self, act_type='mmama'):
        # aim: 根据可能的code如mmmmm，生成全部可能的cr控制序列
        # input: 如amm这种控制序列
        # output:[cr1,cr2,...]每一个cri是一个字典

        attack_unit = []  # 保存选择攻击的单位sn
        move_unit = []  # 保存选择移动的单位的sn
        for i in range(len(act_type)):
            if act_type[i] == 'a':
                attack_unit.append(i)
            else:
                move_unit.append(i)

        # 联合生成全部选择攻击的单位，所形成的cri的列表，如[ [(0,1,1),(0,2,1)]，  [(0,1,1),(0,2,2)]， ... ]
        att_units_action = []
        ats_list=[]
        for sn in attack_unit:
            ats_list.append(self.red_force_units[sn].ats)
        
        
        # 联合生成全部选择移动的单位，所形成的cri的列表，如 [ [(1,0,0)]，  [(2,0,0)]， ... ]
        move_units_action = []

        product_att_move = []
        cr_list = []

    def get_cb_cr_1stepGT(self):
        # aim: 优选智能博弈的双方策略，这个策略首先用随机选择的，然后
        # input: 当前各单位状态
        # output: 优选出来的智能博弈的双方策略

        # 给出可能的行动分类序列v1
        # ! 1个行动分类序列v1，由2个字典组成，分别代表红方和蓝方各单位的类型
        act_type_RF = []
        act_type_BF = []

        pass

        # 针对一个行动分类序列，给出分区调整后的行动分类序列v2，
        # ! 1个行动分类序列v2，由三个字典组成，一个是被拆出去的单位及其行动类型，一个是红方各单位的行动类型，一个是蓝方各单位的类型

        # 针对每一个行动分类序列v2，构造全部的cb，cr，形成列表

        # 依次抽取cb,cr，计算效能值，形成A,B矩阵，计算分区子矩阵的stackelberg equilibrium

        # 从全部stackelberg equilibrium中获得全局最优，返回对应的控制向量cb,cr


if __name__ == '__main__':
    we = WarEnv()
    print(we.blue_force_units[0].name)
