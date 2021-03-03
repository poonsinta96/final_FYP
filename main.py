from __future__ import division, print_function
import numpy as np

import matplotlib.pyplot as plt
import skfuzzy as fuzz


from falcon import Falcon
from smoothen_dataset import smoo
from bull_and_bear_identifier import bull_and_bear

import pickle

import PySimpleGUI as sg
import os.path

def run_model(header):

    #Smoothen the dataset first
    smoo('products/'+header+'/train.csv', 'products/'+header+'/train_smoo.txt',30)#90 works well
    smoo('products/'+header+'/test.csv', 'products/'+header+'/test_smoo.txt',30)

    #Extract train and test set
    train_set = np.genfromtxt('products/'+header+'/train_smoo.txt', delimiter = '\t', skip_header=0)
    test_set = np.genfromtxt('products/'+header+'/test_smoo.txt', delimiter = '\t', skip_header=6)


    #Dataset information
    train_rows = len(train_set)
    test_rows = len(test_set)
    input_feature = 4   #5 if we are using normal train_set
    output_feature = 1


    #print(train_set)


    #creating a new train/test dataset of % change
    perc_train_set = np.zeros(train_rows*5).reshape(train_rows,5)
    for col in range(5):
        perc_train_set[:,col] = ((train_set[:,col+1]/train_set[:,col]) - 1) * 100

    np.savetxt('products/'+header+'/train_smoo_perc.txt', perc_train_set,delimiter = ',',fmt='%f')

    perc_test_set = np.zeros(test_rows*5).reshape(test_rows,5)
    for col in range(5):
        perc_test_set[:,col] = ((test_set[:,col+1]/test_set[:,col]) - 1) * 100

    np.savetxt('products/'+header+'/test_smoo_perc.txt', perc_test_set,delimiter = ',',fmt='%f')


    #print(perc_train_set)


    print()
    print("=================================This is the start of FALCON-AART=================================")
    print()


    model = Falcon(input_size =input_feature, output_size = output_feature)


    #use the dataset of bull and bear condition 
    turning_arr = bull_and_bear('products/'+header+'/train.csv','products/'+header+'/train_smoo_perc.txt','products/'+header+'/train_smoo_perc_bb.txt')
    bullbear_train_set = np.genfromtxt('products/'+header+'/train_smoo_perc_bb.txt', delimiter = ',', skip_header=0)

    model.train(bullbear_train_set,'products/'+header+'/train.csv',header,turning_arr)

    pickle.dump(model, open('products/'+header+'/saved_model','wb'))
    model = pickle.load(open('products/'+header+'/saved_model', 'rb'))

    bullbear_test_set = np.genfromtxt('products/'+header+'/test_smoo_perc.txt', delimiter = ',', skip_header=0)
    animation_data, gui_rule_data, processed_data = model.test(bullbear_test_set,'products/'+header+'/test.csv',header)

    #model.visualise()


    model.animation(animation_data,gui_rule_data, processed_data)




#run_model('old')
#doesnt work : DJI, GSPC
#work: FTSE

#run_model('GSPC') 
#run_model('DJI') 
#run_model('FTSE') #win sample 
#run_model('HSI')  #loss sample
#run_model('IXIC')  #loss sample
#run_model('N225') 
#run_model('STI') 

#run_model('ETF-AGG') #bullbear error
#run_model('ETF-SPY')
run_model('ETF-VGK') #animation error #loss as well
#run_model('ETF-VWO') #


