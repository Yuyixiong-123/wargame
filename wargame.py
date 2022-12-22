import numpy as np
import PSO_NE as ne
import time
import random
import itertools
import pandas as pd

data = [
    [0, 0, 0, 0.6, 0.6, 0.6, 0.5, 0.4, 0.3],
    [0, 0, 0, 0, 0, 0.8, 0.7, 0.7, 0],
    [0, 0, 0, 0, 0, 0.8, 0.7, 0.6, 0],
    [0.2, 0.1, 0.1, 0, 0, 0, 0, 0, 0],
    [0.2, 0.1, 0.1, 0, 0, 0, 0, 0, 0],
    [0.7, 0.3, 0.3, 0, 0, 0, 0, 0, 0],
    [0.5, 0.3, 0.2, 0, 0, 0, 0, 0, 0],
    [0.5, 0.2, 0.2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
]
# columns = ['BB', 'BW1', 'BW2', 'RT1', 'RT2', 'RD1', 'RD2', 'RD3', 'FT']
# index = ['BB', 'BW1', 'BW2', 'RT1', 'RT2', 'RD1', 'RD2', 'RD3', 'FT']
# pk_df = pd.DataFrame(data=data, index=index, columns=columns, dtype=float)


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


class Unit():
    def __init__(self, type, num, x, y, p, w) -> None:
        self.type = type
        self.num = num
        self.x = x
        self.y = y
        self.p = p
        self.w = w
        self.ams = None  # list, available moving space, depending on the corridors and obstacles
        self.aas = None  # list, available attack space, depending on the positions
        self.ass = None  # list, available salvo space, depending on the weapon storage
        self.A_tab = None

    def get_ams(self):
        pass
