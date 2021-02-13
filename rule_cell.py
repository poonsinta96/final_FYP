
import random
import math


class Rule_cell:
    def __init__(self, label, ip_size):
        #initialisation
        self.label = label
        self.bullbear = label.split('/')[0]
        cluster_indexes = [int(y) for y in label.split('/')[1:-1]] 
        self.ip_size = ip_size
        self.x = cluster_indexes[:ip_size]
        self.y = cluster_indexes[ip_size:]

        self.reliability = 1
        self.age = 0
        self.learning_rate =0

        self.fs = 0


    def get_fs_xy(self, x_arr, y_arr):
        #given an array calculate the firing strength (product of all firing strength)
        fs_x = 1
        fs_y = 1
        for x in x_arr:
            fs_x *= x
        for y in y_arr:
            fs_y *= y
        
        #fs = (fs_x*self.ip_size/(self.ip_size + 1)) + (fs_y/(self.ip_size + 1))
        
        fs = fs_x * fs_y * self.reliability

        return fs

    def get_fs_x(self, x_arr):
        #given an array calculate the firing strength (product of all firing strength)
        fs_x = 1
        for x in x_arr:
            fs_x *= x
        
        #fs = (fs_x*self.ip_size/(self.ip_size + 1)) + (fs_y/(self.ip_size + 1))
        
        fs = fs_x  * self.reliability
        self.fs = fs
        return fs

    def get_xcluster(self):
        return self.x
    
    def get_ycluster(self):
        return self.y

    def reward(self):

        # r = self.r + 0.05
        # if r > 1 :
        #     r = 1

        # self.reliability = 1/(1+math.exp(-5 * (r-0.5)))*0.98 + 0.01
        # self.r = r

        print(self.label, self.reliability)

    def learn(self,ideal_ans, old_ans,z_t,real_ans):
        age = self.age
        age = age + 1

        lr = 1/(1+math.exp(-8 * (-0.05*age+0.5)))*0.98 + 0.01 #mimck memory
        self.learning_rate = lr

        #to prevent division of 0
        if real_ans == 0:
            real_ans += 0.01

        r = max( (old_ans + lr*(ideal_ans-old_ans))  / old_ans , 0.000000001)

        #print(r, old_ans, ideal_ans, lr, z_total,real_ans)
        self.reliability = r

