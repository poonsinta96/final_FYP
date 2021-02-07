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
    
    header = 'S&P 500'
    #Smoothen the dataset first
    smoo('dataset/InputDataSet - S&P500_Train.txt', 'new_dataset/'+header+'/train_smoo.txt',20)#90 works well
    smoo('dataset/InputDataSet - S&P500_Test.txt', 'new_dataset/'+header+'/test_smoo.txt',20)

    #Extract train and test set
    train_set = np.genfromtxt('new_dataset/'+header+'/train_smoo.txt', delimiter = '\t', skip_header=0)
    test_set = np.genfromtxt('new_dataset/'+header+'/test_smoo.txt', delimiter = '\t', skip_header=6)


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

    np.savetxt('new_dataset/'+header+'/train_smoo_perc.txt', perc_train_set,delimiter = ',',fmt='%f')

    perc_test_set = np.zeros(test_rows*5).reshape(test_rows,5)
    for col in range(5):
        perc_test_set[:,col] = ((test_set[:,col+1]/test_set[:,col]) - 1) * 100

    np.savetxt('new_dataset/'+header+'/test_smoo_perc.txt', perc_test_set,delimiter = ',',fmt='%f')


    #print(perc_train_set)

    print()
    print("=================================This is the start of FALCON-AART=================================")
    print()


    model = Falcon(input_size =input_feature, output_size = output_feature)


    #use the dataset of bull and bear condition 
    bull_and_bear('dataset/InputDataSet - S&P500_Train.txt','new_dataset/'+header+'/train_smoo_perc.txt','new_dataset/'+header+'/train_smoo_perc_bb.txt')

    bullbear_train_set = np.genfromtxt('new_dataset/'+header+'/train_smoo_perc_bb.txt', delimiter = ',', skip_header=0)

    model.train(bullbear_train_set,'dataset/InputDataSet - S&P500_Train.txt')

    pickle.dump(model, open('saved_model','wb'))
    model = pickle.load(open('saved_model', 'rb'))

    bullbear_test_set = np.genfromtxt('new_dataset/'+header+'/test_smoo_perc.txt', delimiter = ',', skip_header=0)
    model.test(bullbear_test_set,'dataset/InputDataSet - S&P500_Test.txt')

    #model.visualise()


    ################################################## THIS IS FOR GUI ##################################################
    """
    image_and_input_interface = [
        [sg.Text("Training has completed."), sg.Button('Results')],
        [sg.Image(key = "-IMAGE-")],
        [sg.InputText(), sg.InputText(),sg.InputText(),sg.InputText(),sg.Submit()]
    ]

    output_interface =[
        [sg.Image(key="train_results.png")],

        [sg.Text("-TOUT-")]
    ]

    # ----- Full layout -----
    layout = [
        [
            sg.Column(image_and_input_interface),
            sg.VSeperator(),
            sg.Column(output_interface)
        ]
    ]

    window = sg.Window("Fuzzy Complimentary Learning System", layout)

    while True:
        event,values = window.read()
        [print(event,values)]
        if event == 'Results':
            window["-IMAGE-"].update('visualisation.gv.png')
            window["-TOUT-"].update('TOP RULES: BULL/0/1/2/3/1')
        if event == sg.WIN_CLOSED:
            break

    window.close()
    """
