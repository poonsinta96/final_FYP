import numpy as np
import matplotlib.pyplot as plt
import copy

def bull_and_bear(basic_address, sec_address,save_address):
    train_set = np.genfromtxt(basic_address, delimiter = '\t', skip_header=6)

    series = train_set[:,0]

    smoothen_factor = 20 #60
    smoothen_balancer = int(smoothen_factor/2)
    #for the first 15 values
    smoothen = [train_set[0,0] for f in range(smoothen_balancer)]

    first_anchor_index = smoothen_balancer
    second_anchor_index = smoothen_balancer
    turning_arr = []
    current_condition = 'null'

    for y in range(smoothen_balancer, len(series)):
        x = y - smoothen_balancer
        smoo = sum(series[x:y+smoothen_balancer])/smoothen_factor
        smoothen.append(smoo)

        #to identify bull condition
        
            
        if smoo > smoothen[second_anchor_index] * 1.2 or (current_condition == 'bull' and smoo > smoothen[second_anchor_index]):
            # if this is a continued one 
            if current_condition != 'bear' :
                current_condition = 'bull'
                if smoo > smoothen[second_anchor_index ]:
                    second_anchor_index = y - smoothen_balancer
            #if this is a new turning point
            else:
                turning_arr.append(first_anchor_index)
                first_anchor_index = second_anchor_index
                current_condition = 'bull'
        
        #to identify bear condition
        elif smoo < smoothen[second_anchor_index] * 0.8 or (current_condition == 'bear' and smoo < smoothen[second_anchor_index]):
            # if this is a continued one 
            if current_condition != 'bull':
                current_condition = 'bear'
                if smoo < smoothen[second_anchor_index]:
                    second_anchor_index = y - smoothen_balancer
            #if this is a new turning point
            else:
                turning_arr.append(first_anchor_index)
                first_anchor_index = second_anchor_index 
                current_condition = 'bear'


    turning_arr.append(first_anchor_index)
    turning_arr.append(second_anchor_index)
    print(turning_arr)

    #convert into number 
    turning_val = []
    for index in turning_arr:
        turning_val.append(smoothen[index])

    plt.plot(series,color = 'yellow')
    plt.plot(smoothen,color = 'blue')
    plt.plot(turning_arr, turning_val,color = 'red')

    #generate array to insert into dataset 
    bull_or_bear = [1 for x in range(15)]
    condition = 1
    index = 15
    cur_turn = turning_arr.pop(0)
    nex_turn = turning_arr.pop(0)

    for index in range(index, len(series)):
        if index != nex_turn:
            bull_or_bear.append(condition)
        else:
            if condition == 1:
                condition = 0
            elif condition == 0:
                condition = 1
            cur_turn = nex_turn
            if len(turning_arr) != 0:
                nex_turn = turning_arr.pop(0)
            else:
                if condition == 1:
                    condition = 0
                elif condition == 0:
                    condition = 1
            bull_or_bear.append(condition)




    #plt.plot(bull_or_bear, color = 'yellow')
    #plt.show()

    #add info into existing dataset 
    old_dataset = np.genfromtxt(sec_address, delimiter = ',')
    #print(old_dataset)
    bullbear = np.array(bull_or_bear)
    new_dataset= np.column_stack((old_dataset, bullbear))
    print(new_dataset)

    np.savetxt(save_address, new_dataset,delimiter = ',',fmt='%f')

#bull_and_bear('dataset/InputDataSet - S&P500_Train.txt','dataset/S&P500Perc_train(smoo).txt','dataset/S&P500Perc_WITH_BullOrBear_train(smoo).txt')


#bull_and_bear('dataset/InputDataSet - S&P500_Test.txt','dataset/S&P500Perc_test.txt','dataset/S&P500Perc_WITH_BullOrBear_test.txt')
#note that test_set is ALL bear

