import random
import math


class First_cell:
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

    def compute_hyper_ranges(self ,val):
        ans_hyper_ranges = []
        for cluster in range(len(self.hyper_ranges)):
            u = self.hyper_ranges[cluster][0]
            v = self.hyper_ranges[cluster][1]

            learning_rate,f = self.get_learning_rate(val, cluster)

            #update averages
            old_average = self.averages[cluster]
            new_age = self.ages[cluster]
            old_age = new_age - 1
            average = (old_average * old_age + val ) / (new_age)

            new_u = u - learning_rate * (u - min(val,average))
            new_v = v + learning_rate * (max(val,average) - v)
            
            ans_hyper_ranges.append([new_u, new_v])
            #print('New hyperranges for',self.stream,self.hyper_ranges)

        return ans_hyper_ranges

    def s_upward_flow_in(self, x):
        #if this is the first data row
        if self.hyper_ranges ==[]:
            self.hyper_ranges.append([x,x])
            self.ages.append(1)
            self.minmax.append([x,x])
            self.fuzziness.append(x/100) #to prevent computational error
            self.averages.append(x)

        hyper_ranges= self.compute_hyper_ranges(x)
        return hyper_ranges, x

    def upward_flow_in(self, x):
        #if this is the first data row
        if self.hyper_ranges ==[]:
            self.hyper_ranges.append([x,x])
            self.ages.append(1)
            self.minmax.append([x,x])
            self.fuzziness.append(x/100) #to prevent computational error
            self.averages.append(x)

        return self.hyper_ranges, x

    def get_new_fuzziness(self ,winning_cluster, val):
        mini = self.minmax[winning_cluster][0]
        maxi = self.minmax[winning_cluster][1]

        #update the mini and maxi if it changed
        if val< mini:
            mini = val
        if val> maxi:
            maxi = val

        u = self.hyper_ranges[winning_cluster][0]
        v = self.hyper_ranges[winning_cluster][1]


        f = ((maxi - v)+(u-mini))/2
        if f == 0:
            return 0.001

        return f


    def get_learning_rate(self, val, winning_cluster):
        age = self.ages[winning_cluster]
        initial_f = max(self.fuzziness[winning_cluster], 0.001)
        
        final_f = max(self.get_new_fuzziness(winning_cluster, val),0.001)

        #print(initial_f,final_f,age,self.minmax, self.hyper_ranges[winning_cluster])

        x = 1/(initial_f * ((final_f/initial_f)**(1/age)))
        #x = 1/(((final_f/initial_f)**(1/age)))

        beta = math.exp(-x)
        #print(age,initial_f, final_f,beta)
        return beta, final_f


    def learn(self ,winning_cluster,val):

        u = self.hyper_ranges[winning_cluster][0]
        v = self.hyper_ranges[winning_cluster][1]
        self.ages[winning_cluster] += 1

        #update the mini and maxi if it changed
        if val< self.minmax[winning_cluster][0]:
            self.minmax[winning_cluster][0] = val
        if val> self.minmax[winning_cluster][1]:
            self.minmax[winning_cluster][1] = val

        learning_rate,f = self.get_learning_rate(val, winning_cluster)
        self.fuzziness[winning_cluster] =f 


        #update averages
        old_average = self.averages[winning_cluster]
        new_age = self.ages[winning_cluster]
        old_age = new_age - 1
        average = (old_average * old_age + val ) / (new_age)
        self.averages[winning_cluster] = average

        new_u = u - learning_rate * (u - min(val,average))
        new_v = v + learning_rate * (max(val,average) - v)
        
        self.hyper_ranges[winning_cluster] = [new_u, new_v]
        #print('New hyperranges for',self.stream,self.hyper_ranges)

        return self.hyper_ranges[winning_cluster]

    def create_cluster(self, val):
        self.hyper_ranges.append([val,val])
        self.ages.append(1)
        self.minmax.append([val,val])
        self.fuzziness.append(abs(val/100)) #to prevent computational error
        self.averages.append(val)