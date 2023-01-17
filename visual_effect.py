import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# plt.scatter(7.4, 8.2, label='歼击机b1', c="blue", marker='v')
# plt.scatter(7.8, 8.5, label='歼击机b2', c="blue", marker='v')
# plt.scatter(8.2, 8.8, label='轰炸机BB', c="blue", marker='.')

# plt.scatter(2, 2, label='桥梁RT', c="red", marker='s')
# plt.scatter(2, 2.6, label='歼击机r1', c="red", marker='^')
# plt.scatter(2.5, 2, label='歼击机r2', c="red", marker='>')

# plt.xlim((0, 10))
# plt.ylim((0, 10))

# plt.title("兵棋推演验证模型-BH301")
# plt.legend(loc=4)
# plt.grid()
# plt.show()

nameList = ['歼击机b1',
            '歼击机b2',
            '轰炸机BB',
            '桥梁RT',
            '歼击机r1',
            '歼击机r2']

d = pd.DataFrame(index=nameList, columns=['原始单位数量', '结束单位数量'])
d['原始单位数量'] = [5, 5, 2, 1, 4, 3]
d['结束单位数量'] = [3, 3, 2, 0, 2, 1]
d.plot(kind='bar')
plt.show()

# plt.bar(nameList, [5, 5, 2, 1, 4, 3], label='原始单位数量')
# plt.bar(nameList, [3, 3, 2, 0, 2, 1], label='结束单位数量')

# plt.legend()
# plt.show()
