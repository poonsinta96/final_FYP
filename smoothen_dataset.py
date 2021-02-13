import numpy as np
import matplotlib.pyplot as plt
import copy

def smoo(basic_address,save_address, smoothen_factor):
    dataset = np.genfromtxt(basic_address, delimiter = ',')


    smoothen_factor = smoothen_factor #60
    
    #plt.plot(dataset[:,-1],color = 'yellow')
    num_of_x = dataset[0]

    smooth_dataset = dataset[:smoothen_factor,:]

    #plot first SMA 
    row_array = np.array([])
    for col_index in range(len(num_of_x)):
        ans = sum(dataset[0:smoothen_factor, col_index])/smoothen_factor
        row_array=np.append(row_array, ans)

    smooth_dataset = np.append(smooth_dataset, [row_array],axis = 0)

    #execute EMA 
    multiplier = 2/(smoothen_factor+1)

    for row_index in range(smoothen_factor+1, len(dataset)):
        row_array = np.array([])
        for col_index in range(len(num_of_x)):
            today_px =dataset[row_index, col_index]
            ystd_ema = smooth_dataset[row_index-1, col_index]
            ema = (today_px - ystd_ema ) * multiplier  + ystd_ema

            row_array = np.append(row_array, ema)
        
        smooth_dataset = np.append(smooth_dataset, [row_array],axis = 0)



    # for row_index in range(smoothen_factor, len(dataset)):
    #     row_array = np.array([])
    #     for col_index in range(len(num_of_x)):
    #         ans = sum(dataset[row_index-smoothen_factor:row_index, col_index])/smoothen_factor
    #         row_array=np.append(row_array, ans)

    #     smooth_dataset = np.append(smooth_dataset, [row_array],axis = 0)

    print(smooth_dataset)
    # plt.plot(smooth_dataset[:,-1],color = 'green')

    # plt.show()

    np.savetxt(save_address, smooth_dataset,delimiter = '\t',fmt='%f')




