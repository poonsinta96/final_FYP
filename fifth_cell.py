
import random
import math


class Fifth_cell:
    def __init__(self, stream):
        #initialisation
        self.stream = stream
        self.hyper_ranges = []
        self.ages = []
        self.minmax = []
        self.fuzziness = []
        self.averages = []

        #for roll back purposes
        self.cache =[]


    def downward_flow_in(self, x):
        #if this is the first data row
        if self.hyper_ranges ==[]:
            self.hyper_ranges.append([x,x])
            self.ages.append(1)
            self.minmax.append([x,x])
            self.fuzziness.append(0.00001) #to prevent computational error
            self.averages.append(x)


        return self.hyper_ranges, x

    def get_new_fuzziness(self ,winning_cluster, val):
        mini = self.minmax[winning_cluster][0]
        maxi = self.minmax[winning_cluster][1]

        #update the mini and maxi if it changed
        if val< mini:
            mini = val
            self.minmax[winning_cluster][0] = mini
        if val> maxi:
            maxi = val
            self.minmax[winning_cluster][1] = maxi

        u = self.hyper_ranges[winning_cluster][0]
        v = self.hyper_ranges[winning_cluster][1]

        #print(maxi, v,u,mini)

        f = ((maxi - v)+(u-mini))/2
        if f == 0:
            return 0.001
        self.fuzziness[winning_cluster] =f 

        return f


    def get_learning_rate(self, val, winning_cluster):
        self.ages[winning_cluster] += 1
        age = self.ages[winning_cluster]
        initial_f = self.fuzziness[winning_cluster]
        
        final_f = self.get_new_fuzziness(winning_cluster, val)
        
        x = 1/(initial_f * ((final_f/initial_f)**(1/age)))
        #x = 1/(((final_f/initial_f)**(1/age)))

        beta = math.exp(-x)
        #print(age,initial_f, final_f,beta)
        return beta

    def learn(self ,winning_cluster,val):

        u = self.hyper_ranges[winning_cluster][0]
        v = self.hyper_ranges[winning_cluster][1]

        learning_rate = self.get_learning_rate(val, winning_cluster)

        #update averages
        old_average = self.averages[winning_cluster]
        new_age = self.ages[winning_cluster]
        old_age = new_age - 1
        average = (old_average * old_age + val ) / (new_age)
        self.averages[winning_cluster] = average

        new_u = u - learning_rate * (u - min(val,average))
        new_v = v + learning_rate * (max(val,average) - v)

        # new_u = u - learning_rate * (u - val)
        # new_v = v + learning_rate * (val - v)

        self.hyper_ranges[winning_cluster] = [new_u, new_v]
        #print('New hyperranges for',self.stream,self.hyper_ranges)

        return self.hyper_ranges[winning_cluster]

    def create_cluster(self, val):
        self.hyper_ranges.append([val,val])
        self.ages.append(0)
        self.minmax.append([val,val])
        self.fuzziness.append(abs(val/100)) #to prevent computational error
        self.averages.append(val)

    def defuzzify(self,mz_arr):
        numerator = 0
        denominator = 0
        for mz_pair in mz_arr:
            m = mz_pair[0]
            z = mz_pair[1]
            numerator += (m*z)
            denominator += z
        
        ans = numerator/denominator

        return ans
