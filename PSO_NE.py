'''
rps_p,bps_p: red/blue particles position
'''
from scipy.stats import dirichlet
import numpy as np
import random

np.random.seed(1)

A = np.array([[8, 9, 3],
              [2, 5, 6],
              [4, 1, 7]])
B = -A
# print(B.shape)
# a = np.random.dirichlet(np.ones(3), size=1)


class PSO():
    def __init__(self, A, B, p_num=20, iter_num=50, w_start=1, w_end=0.1, c1=2, c2=2, vmax=2):
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
        self.ini_ps()
        # self.evaluation()
        self.get_NE()
        return self.bps_gbest_history, self.rps_gbest_history

    def ini_ps(self):
        # random mixed strategy for blue players
        self.bps_posi = np.random.dirichlet(
            np.ones(self.col_num), size=self.p_num)
        self.bps_velo = np.ones((self.col_num, self.p_num))
        self.bps_pbest = np.ones((self.col_num, self.p_num))
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
            #  We use the matrix multiply to replace the max and Traverse process
            m1 = np.max(np.matmul(self.A, self.bps_posi[i]))
            m2 = np.max(np.matmul(self.rps_posi[i], self.B))

            red_u = np.matmul(
                np.matmul(self.rps_posi[i], self.A), self.bps_posi[i])
            blue_u = np.matmul(
                np.matmul(self.rps_posi[i], self.B), self.bps_posi[i])

            self.ps_fit[i] = m1+red_u-m1-blue_u
        # print(self.ps_fit)

    def get_NE(self):
        for iter in range(self.num_iter):
            #  Evaluate all of the particles.
            self.evaluation()

            #  update the gbest.
            gbest_index = np.where(a=np.min(self.ps_fit))[0]
            rp_gbest = self.rps_posi(gbest_index)
            bp_gbest = self.bps_posi(gbest_index)

            if np.min(self.ps_fit) < self.ps_gbest_fit_history:
                self.ps_gbest_fit_history = np.min(self.ps_fit)
                self.bps_gbest_history = self.bps_posi(gbest_index)
                self.rps_gbest_history = self.rps_posi(gbest_index)

            #  Get the Pbest and update
            for i in range(self.p_num):
                if self.ps_fit[i] < self.ps_pbest_fit[i]:
                    self.bps_pbest[i] = self.bps_posi[i]
                    self.rps_pbest[i] = self.rps_posi[i]

                #  Calculate the velocity of each row( Particle).
                #!  truncation by the V max.
                self.bps_velo[i] = self.w_end*self.bps_velo[i] +\
                    self.c1 * random.random() * (self.bps_pbest[i] - self.bps_posi[i]) +\
                    self.c2 * random.random() * (self.bps_gbest_history -
                                                 self.bps_posi[i])
                self.bps_velo[np.where(self.bps_velo > self.vmax)] = self.vmax

                self.rps_velo[i] = self.w_end*self.rps_velo[i] +\
                    self.c1 * random.random() * (self.rps_pbest[i] - self.rps_posi[i]) +\
                    self.c2 * random.random() * (self.rps_gbest_history -
                                                 self.rps_posi[i])
                self.rps_velo[np.where(self.rps_velo > self.vmax)] = self.vmax

                #  Update the positions of the particle swarm.
                # ! normalization for constrain
                self.bps_posi[i] = (self.bps_posi[i] + self.bps_velo[i])
                self.bps_posi[i] = self.bps_posi[i] / np.sum(self.bps_posi[i])
                self.rps_posi[i] = self.rps_posi[i] + self.rps_velo[i]
                self.rps_posi[i] = self.rps_posi[i] / np.sum(self.rps_posi[i])

            print(iter, self.ps_gbest_fit_history)


p = PSO(A, B)
a = np.array([1, 2, 7])

# print(np.dirichlet(a))
alpha = dirichlet.mean(alpha=a)
# dir_mean = dirichlet.mean(alpha)
print(alpha)
print(p.ps_velc)
# print(dir_mean)
# print(np.matmul(A, a))
# print(np.max(np.matmul(a, A)))
# A = np.array([[8, 9, 3],
#               [2, 5, 6],
#               [4, 1, 7]])
