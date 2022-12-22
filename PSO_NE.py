'''
rps_p,bps_p: red/blue particles position
blue side always be calculate first
'''
from scipy.stats import dirichlet
import numpy as np
import random
import time

# np.random.seed(1)

# A = np.array([[8, 9, 3],
#               [2, 5, 6],
#               [4, 1, 7]])
# B = -A

# A = np.array([
#     [1, 235, 0, 0.1],
#     [0, 1, 235, 0.1],
#     [235, 0, 1, 0.1],
#     [1.1, 1.1, 1.1, 0]
# ])
# B = np.array([
#     [1, 0, 235, 1.1],
#     [235, 1, 0, 1.1],
#     [0, 235, 1, 1.1],
#     [0.1, 0.1, 0.1, 0]
# ])

# dimension = int(1e5)
# A = np.random.dirichlet(np.ones(dimension), size=dimension)
# B = -A

# get normalized first
# banch = (np.max(A)+np.max(B)-np.min(A)-np.min(B)) / 4
# A = A/banch
# B = B/banch
# print(A, B)


class PSO():
    def __init__(self, A, B, p_num=80, iter_num=10, w_start=1, w_end=0.1, c1=2, c2=2, vmax=2, cut=0.01, stagnate_factor=1e-6):
        self.A = A
        self.row_num, self.col_num = A.shape
        self.B = B
        self.w_start = w_start
        self.w_end = w_end
        self.p_num = p_num
        self.iter_num = iter_num
        self.c1 = c1
        self.c2 = c2
        self.vmax = vmax
        self.cut = cut
        self.stagnate_factor = stagnate_factor
        self.ini_ps()

    def ini_ps(self):
        # random mixed strategy for blue players
        # p_num rows, len(blue side strategy size)
        self.bps_posi = np.random.dirichlet(
            np.ones(self.col_num), size=self.p_num)
        self.bps_velo = np.ones((self.p_num, self.col_num))
        self.bps_pbest = np.ones((self.p_num, self.col_num))
        self.bps_gbest_history = np.ones((self.col_num))

        # random mixed strategy for red players
        self.rps_posi = np.random.dirichlet(
            np.ones(self.row_num), size=self.p_num)
        self.rps_velo = np.ones((self.p_num, self.row_num))
        self.rps_pbest = np.ones((self.p_num, self.row_num))
        self.rps_gbest_history = np.ones((self.row_num))

        # velocity and fitness is public for red and blue in PSO, cause the solution is a pair of strategy list
        self.ps_fit = np.zeros((self.p_num))
        self.ps_pbest_fit = np.ones(self.p_num)*1000
        self.ps_gbest_fit_history = 1000

    #  It is lucky to have a same formulation of fitness matrix in the paper.
    def evaluation(self):
        for i in range(self.p_num):
            #! calculate the i_th particle's fitness
            #!  We use the matrix multiply to replace the max-choose and Traverse process
            m1 = np.max(np.matmul(self.A, self.bps_posi[i]))
            m2 = np.max(np.matmul(self.rps_posi[i], self.B))

            red_u = np.matmul(
                np.matmul(self.rps_posi[i], self.A), self.bps_posi[i])
            blue_u = np.matmul(
                np.matmul(self.rps_posi[i], self.B), self.bps_posi[i])

            self.ps_fit[i] = m1-red_u+m2-blue_u
        # print(self.ps_fit)

    def get_NE(self):
        for iter in range(self.iter_num):
            # the damping of w from w_start to w_end
            w = self.w_end+(self.w_start-self.w_end) * \
                (self.iter_num-iter) / (self.iter_num)
            # w = self.w_start
            #  Evaluate all of the particles.
            self.evaluation()
            # w = np.mean(self.ps_fit)
            # w = 1

            #  update the gbest.
            min_fit = np.min(self.ps_fit)
            gbest_index = np.where(self.ps_fit == min_fit)[0][0]
            # rp_gbest = self.rps_posi(gbest_index)
            # bp_gbest = self.bps_posi(gbest_index)

            if min_fit < self.ps_gbest_fit_history:
                self.ps_gbest_fit_history = min_fit
                self.bps_gbest_history = self.bps_posi[gbest_index]
                self.rps_gbest_history = self.rps_posi[gbest_index]

            # stagnate and cut
            if iter > 2:
                if self.ps_gbest_fit_history - min_fit < self.stagnate_factor:
                    print('pre-end: stagnate and cut')
                    return self.bps_gbest_history, self.rps_gbest_history

            # enough to cut
            if min_fit < self.cut:
                print('pre-end: enough to cut')
                return self.bps_gbest_history, self.rps_gbest_history

            #  Get the Pbest and update
            for i in range(self.p_num):
                if self.ps_fit[i] < self.ps_pbest_fit[i]:
                    self.bps_pbest[i] = self.bps_posi[i]
                    self.rps_pbest[i] = self.rps_posi[i]

                #  Calculate the velocity of each row( Particle).
                #!  truncation by the V max.
                self.bps_velo[i] = w*self.bps_velo[i] +\
                    self.c1 * random.random() * (self.bps_pbest[i] - self.bps_posi[i]) +\
                    self.c2 * random.random() * (self.bps_gbest_history -
                                                 self.bps_posi[i])
                self.bps_velo[np.where(self.bps_velo > self.vmax)] = self.vmax

                self.rps_velo[i] = w*self.rps_velo[i] +\
                    self.c1 * random.random() * (self.rps_pbest[i] - self.rps_posi[i]) +\
                    self.c2 * random.random() * (self.rps_gbest_history -
                                                 self.rps_posi[i])
                self.rps_velo[np.where(self.rps_velo > self.vmax)] = self.vmax

                #  Update the positions of the particle swarm.
                # ! normalization for constrain
                self.bps_posi[i] = (self.bps_posi[i] + self.bps_velo[i])
                self.bps_posi[np.where(self.bps_posi < 0)] = 0
                self.bps_posi[i] = self.bps_posi[i] / np.sum(self.bps_posi[i])

                self.rps_posi[i] = self.rps_posi[i] + self.rps_velo[i]
                self.rps_posi[np.where(self.rps_posi < 0)] = 0
                self.rps_posi[i] = self.rps_posi[i] / np.sum(self.rps_posi[i])

            print(iter, self.ps_gbest_fit_history, '\n')
        return self.rps_gbest_history, self.bps_gbest_history


if __name__ == '__main__':
    t1 = time.time()
    p = PSO(A, B)
    p1, p2 = p.get_NE()
    # print(p1, p2)
    print('cal-time: ', time.time()-t1)

    # a = np.array([1, 2, 7])
    # print(np.dirichlet(a))
    # alpha = dirichlet.mean(alpha=a)
    # dir_mean = dirichlet.mean(alpha)
    # print(alpha)
    # print(p.ps_velc)
    # print(dir_mean)
    # print(np.matmul(A, a))
    # print(np.max(np.matmul(a, A)))
    # A = np.array([[8, 9, 3],
    #               [2, 5, 6],
    #               [4, 1, 7]])
