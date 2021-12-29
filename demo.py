'''
Created on Dec 29, 2020

@author: voodoocode
'''

import numpy as np

import finn.sfc.td as td

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import finn.misc.timed_pool as tp

################################
# CONFIGURE ACCORDING TO SETUP #
################################

frequency_sampling = 5500
frequency_tgt = 30
frequency_range = 10
frequency_bin_sz = 0.5

thread_cnt = 1 #Number of threads used for the simulation

input_data = None

##################################
# CONFIGURE ACCORDING TO RESULTS #
##################################

minimal_angle_thresh = 5

#############################
# CODE BELOW, DO NOT MODIFY #
#############################

def dac(data_1, data_2, freq_bin_factor):
    fmin = int(frequency_tgt - frequency_range*freq_bin_factor)
    fmax = int(frequency_tgt + frequency_range*freq_bin_factor + 1)
    
    return td.run_dac(data_1, data_2, fmin, fmax, frequency_sampling, 
                    int(frequency_sampling*freq_bin_factor), int(frequency_sampling*freq_bin_factor), return_signed_conn = True, minimal_angle_thresh = minimal_angle_thresh)[1]

def main(data = None):
    """
    @param data: Expects a numpy array of format 2 x samples or 1 x samples to be used as input data. In case no input data is provided a generic input data is used.
    """   
    #Phase range
    phase_min = -270
    phase_max = 270
    phase_step = 2

    run(data, phase_min, phase_max, phase_step)
            
    plt.show(block = True)

def __run_inner(data1, data2, offset, phase_shift, freq_bin_factor):
    
    loc_offset = offset - int(np.ceil(frequency_sampling/frequency_tgt * phase_shift/360))
    loc_data = data2[(loc_offset):]
    data2_padded = np.zeros(loc_data.shape)
    data2_padded += loc_data
    
    return dac(data1, data2_padded, freq_bin_factor)

def run(data, phase_min, phase_max, phase_step):
    freq_bin_factor = 1/frequency_bin_sz
    
    if (data is None):
        path = "data_0.npy"
        data1 = np.load(path)[3, :]
        data2 = np.load(path)[2, :]
        
    #Data container
    offset = int(np.ceil(frequency_sampling/frequency_tgt))
    data1 = data1[(offset):]
    
    dac_scores = tp.run(thread_cnt, __run_inner, [(data1, data2, offset, phase_shift,
                                                   freq_bin_factor) for phase_shift in np.arange(phase_min, phase_max, phase_step)], max_time = None, delete_data = False)
    dac_scores = np.asarray(dac_scores)
    
    #Draw figure
    ref_shape = np.ones(len(np.arange(phase_min, phase_max, phase_step)));
    ref_shape[np.argwhere(np.arange(phase_min, phase_max, phase_step) > 0).squeeze()] = -1;
    ref_shape[np.argwhere(np.abs(np.arange(phase_min, phase_max, phase_step)) < minimal_angle_thresh).squeeze()] = 0;

    plt.scatter(np.arange(phase_min, phase_max, phase_step), ref_shape, color = "black", label = "idealized reference", marker = "s")
    plt.scatter(np.arange(phase_min, phase_max, phase_step), dac_scores, color = "blue", label = "DAC scores")
    plt.scatter(np.arange(phase_min, phase_max, phase_step)[np.argwhere(np.isnan(dac_scores)).squeeze()], np.zeros(np.argwhere(np.isnan(dac_scores)).squeeze().shape), color = "red", 
                label = "volume conductance")
    
    plt.legend()

main(input_data)


