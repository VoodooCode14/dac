'''
Created on Jun 2, 2020

@author: voodoocode
'''

import numpy as np

import finn.sfc.__misc as misc
import finn.sfc.td as td
import finn.sfc.cd as cohd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import finn.misc.timed_pool as tp

thread_cnt = 1
freq_bin_sz = 2

def mag_sq_coh(data, _0, _1, frequency_tgt, _2, _3):
    return cohd.run_msc(data)[frequency_tgt]
def img_coh(data, _0, _1, frequency_tgt, _2, _3):
    return cohd.run_ic(data)[frequency_tgt]
def dac_coh(data, _0, bins, frequency_tgt, _1, _2, freq_range = 5, min_phase_diff = 10):
    dac_range = cohd.run_dac(data, bins, frequency_tgt - freq_range*freq_bin_sz, frequency_tgt + freq_range*freq_bin_sz + 1, return_signed_conn = True, minimal_angle_thresh = min_phase_diff)    
    return (0 if (np.isnan(dac_range)) else dac_range)
def psi_coh(_0, data, bins, frequency_tgt, _1, _2, freq_range = 5):
    return cohd.run_psi(data, bins, frequency_tgt - freq_range*freq_bin_sz, frequency_tgt + freq_range*freq_bin_sz + 1)
def wpli_coh(_0, _1, _2, frequency_tgt, data1, data2):
    win_sz = 5500
    
    s_xy = list()
    for block_start in np.arange(0, np.min([len(data1), len(data2)]) - win_sz, win_sz):
        loc_data1 = data1[block_start:(block_start + win_sz)]
        loc_data2 = data2[block_start:(block_start + win_sz)]
        
        seg_data_X = misc.__segment_data(loc_data1, win_sz, "zero")
        seg_data_Y = misc.__segment_data(loc_data2, win_sz, "zero")
    
        (_, f_data_X) = misc.__calc_FFT(seg_data_X, 5500, win_sz, "hann")
        (_,    f_data_Y) = misc.__calc_FFT(seg_data_Y, 5500, win_sz, "hann")
    
        s_xy.append((np.conjugate(f_data_X[0, :]) * f_data_Y[0, :] * 2))

    s_xy = np.asarray(s_xy)
    
    return cohd.run_wpli(s_xy)[frequency_tgt]

def main():
    #Signal configuration
    window = "hann"
    pad_type = "zero"
    frequency_tgt_shift = np.concatenate((np.arange(-8, 8, 1/10), [8])); sigma = 25
    signal_amplitude_scaling = 10000
    signal_amplitde_helper = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.arange(-len(frequency_tgt_shift)/2, len(frequency_tgt_shift)/2+1, 1/signal_amplitude_scaling) - 0)**2 / (2 * sigma**2))
    signal_amplitude = [signal_amplitde_helper[int(loc_signal_amplitude_idx * signal_amplitude_scaling + signal_amplitude_scaling/2)] for (loc_signal_amplitude_idx, _) in enumerate(frequency_tgt_shift)]
    signal_amplitude = np.asarray(signal_amplitude)
    signal_amplitude *= sigma
    
    noise_weight = 0.2
    
    #Phase range
    phase_min = -270
    phase_max = 270
    phase_step = 2
        
    #Select methods
    methods = [mag_sq_coh, img_coh, wpli_coh, psi_coh, dac_coh]
    
    file_cnt = 4
    
    #Generate figure
    (_, axes0) = plt.subplots(file_cnt, 1)
    (_, axes1) = plt.subplots(len(methods), file_cnt)
    for i in range(len(methods)):
        for j in range(file_cnt):
            axes1[i, j].set_ylim(-1.1, 1.1)
    
    cols = ["P16 DM3 T5 SC", "P23 DM3 T2 SC", "P27 DM1 T6 SC", "P16 DM3 T5 MC"]
    rows = ["AbsCoh", "imagCoh", "wpli", "psi", "dac"]
            
    for ax, col in zip(axes1[0], cols):
        ax.set_title(col, size='small')
        
    for ax, row in zip(axes1[:,0], rows):
        ax.set_ylabel(row, rotation=90, size='small')
        
    demo_file = "data_0.npy"
    frequency_tgt = 30
    score0 = single_channel_shift(noise_weight, phase_min, phase_max, phase_step, pad_type, window, axes0, axes1, methods, frequency_peak = frequency_tgt, path = demo_file, axes_idx = 0)
      
    demo_file = "data_1.npy"
    frequency_tgt = 20
    score1 = single_channel_shift(noise_weight, phase_min, phase_max, phase_step, pad_type, window, axes0, axes1, methods, frequency_peak = frequency_tgt, path = demo_file, axes_idx = 1)
     
    demo_file = "data_2.npy"
    frequency_tgt = 29
    score2 = single_channel_shift(noise_weight, phase_min, phase_max, phase_step, pad_type, window, axes0, axes1, methods, frequency_peak = frequency_tgt, path = demo_file, axes_idx = 2)

    frequency_tgt = 30
    score3 = multi_channel_shift(methods, phase_min, phase_max, phase_step, axes0, axes1, frequency_tgt)
    
    score_list = np.asarray([score0, score1, score2, score3])
    np.save("score_list.npy", score_list)
            
    plt.show(block = True)

def __multi_channel_shift(methods, offset, frequency_sampling, frequency_peak, phase_shift, data2, noise_weight, data1, nfft):
        
    features = [None for _ in range(len(methods))]
    loc_offset = offset - int(np.ceil(frequency_sampling/frequency_peak * phase_shift/360))
    loc_data = data2[(loc_offset):]
    data22 = np.zeros(loc_data.shape)
    data22 += loc_data
    data22 += np.random.random(len(loc_data)) * noise_weight
    
    (bins, comp_coh) = td.run_cc(data1, data22, nperseg = nfft, pad_type = "zero", fs = 5500, nfft = nfft, window = "hann")
    
    signal_1_step_sz = len(data1)/10
    signal_2_step_sz = len(data22)/10
    comp_coh2 = [td.run_cc(data1[int(idx * signal_1_step_sz):int((idx + 1) * signal_1_step_sz)],
                           data22[int(idx * signal_2_step_sz):int((idx + 1) * signal_2_step_sz)],
                           nfft, "zero", 5500, nfft, "hann")[1] for idx in range(10)]
        
    for (method_idx, method) in enumerate(methods):
        features[method_idx] = method(comp_coh, comp_coh2, bins, frequency_peak, data1, data22)
        
    return features

def multi_channel_shift(methods, phase_min, phase_max, phase_step, axes0, axes1, frequency_peak, frequency_sampling = 5500, noise_weight = 0.0, axes_idx = None):
    
    path = "data_0.npy"
    data1 = np.load(path)[3, :]
    data2 = np.load(path)[2, :]
    
    nfft = int(frequency_sampling*freq_bin_sz)
    
    #Data container
    features = list()
    for _ in methods:
        features.append(list())
    
    offset = int(np.ceil(frequency_sampling/frequency_peak))
    
    data1 = data1[(offset):]
    
    features = tp.run(thread_cnt, __multi_channel_shift, [(methods, offset, frequency_sampling, frequency_peak,
                                       phase_shift, data2, noise_weight, data1, nfft) for phase_shift in np.arange(phase_min, phase_max, phase_step)], max_time = None, delete_data = False)
    features = np.asarray(features)
    features = features.transpose()
    features = features.tolist()
    
    #Draw figure
    if (axes_idx is None):
        axes_idx = 3
    
    axes0[axes_idx].psd(data1, Fs = 5500, NFFT = nfft)
    axes0[axes_idx].psd(data2, Fs = 5500, NFFT = nfft)
    axes0[axes_idx].set_xlim(0,100)
    axes0[axes_idx].set_ylim(0, 30)
    
    ref_shape = np.ones(len(np.arange(phase_min, phase_max, phase_step)));
    ref_shape[np.argwhere(np.arange(phase_min, phase_max, phase_step) > 0).squeeze()] = -1;
    ref_shape[np.argwhere(np.abs(np.arange(phase_min, phase_max, phase_step)) < 5).squeeze()] = 0;
    score = list()
    for (method_idx, _) in enumerate(methods):
        axes1[method_idx, axes_idx].scatter(np.arange(phase_min, phase_max, phase_step), features[method_idx])
        axes1[method_idx, axes_idx].scatter(np.arange(phase_min, phase_max, phase_step), ref_shape)
        score.append(np.sum(np.abs(features[method_idx] - ref_shape)))
        
    return score

def __single_channel_shift(data, offset, frequency_sampling, frequency_peak, phase_shift,
                           noise_weight, visualize, axes, signal_1, nperseg, pad_type, fs, nfft, window,
                           methods):
    
    loc_features = [None for _ in range(len(methods))]
    loc_offset = offset - int(np.ceil(frequency_sampling/frequency_peak * phase_shift/360))
    loc_data = data[(loc_offset):]
    signal_2 = np.zeros(loc_data.shape)
    signal_2 += loc_data
    signal_2 += np.random.random(len(loc_data)) * noise_weight

    if (visualize):
        axes.cla()
        axes.plot(signal_1[:500], color = "blue")
        axes.plot(signal_2[:500], color = "red")
        print(loc_offset, offset)
        plt.show(block = False)
        plt.pause(0.01)
    
    (bins, comp_coh) = td.run_cc(signal_1, signal_2, nperseg, pad_type, fs, nfft, window)
    
    signal_1_step_sz = len(signal_1)/10
    signal_2_step_sz = len(signal_2)/10
    comp_coh2 = [td.run_cc(signal_1[int(idx * signal_1_step_sz):int((idx + 1) * signal_1_step_sz)],
                           signal_2[int(idx * signal_2_step_sz):int((idx + 1) * signal_2_step_sz)],
                           nperseg, pad_type, fs, nfft, window)[1] for idx in range(10)]
    
    for (method_idx, method) in enumerate(methods):
        loc_features[method_idx] = method(comp_coh, comp_coh2, bins, frequency_peak, signal_1, signal_2)
    return loc_features

def single_channel_shift(noise_weight,
                      phase_min, phase_max, phase_step,
                      pad_type, window,
                      axes0, axes1,
                      methods = [mag_sq_coh, img_coh, dac_coh, psi_coh],
                      frequency_peak = 30, path = "", frequency_sampling = 5500, 
                      axes_idx = None):    
    visualize = False
    
    if (path[-5] == "0"):
        data = np.load(path)[3, :] #Channel with 'strongest' beta activity @30Hz
    elif(path[-5] == "1"):
        data = np.load(path)[3, :] #Channel with 'strongest' beta activity @20Hz
    elif(path[-5] == "2"):
        data = np.load(path)[1, :] #Channel with 'strongest' beta activity @20Hz
    else:
        raise AssertionError
    offset = int(np.ceil(frequency_sampling/frequency_peak))
    
    #overwriting paramters with sampling frequency from loaded data
    nperseg = int(frequency_sampling*freq_bin_sz)
    fs = frequency_sampling
    nfft = int(frequency_sampling*freq_bin_sz)
    
    #Data container
    features = list()
    for _ in methods:
        features.append(list())
    
    #Generate data
    loc_data = data[offset:]
    signal_1 = np.zeros((loc_data).shape)
    signal_1 += loc_data
    signal_1 += np.random.random(len(loc_data)) * noise_weight
    
    if (visualize):
        (fig, axes) = plt.subplots(1, 1)
        plt.show(block = False)
        plt.pause(0.01)
    else:
        axes = None
        
    features = tp.run(thread_cnt, __single_channel_shift, [(data, offset, frequency_sampling, frequency_peak, phase_shift,
                                                   noise_weight, visualize, axes, signal_1, nperseg, pad_type,
                                                   fs, nfft, window, methods) for phase_shift in np.arange(phase_min, phase_max, phase_step)], max_time = None, delete_data = False)
    features = np.asarray(features)
    features = features.transpose()
    features = features.tolist()
    
    if (visualize):
        fig.clear()
    
    if (axes_idx is None):
        axes_idx = 2

    #Draw figure
    axes0[axes_idx].psd(signal_1, Fs = fs, NFFT = fs)
    axes0[axes_idx].set_xlim(0,100)
    axes0[axes_idx].set_ylim(0, 30)
    
    ref_shape = np.ones(len(np.arange(phase_min, phase_max, phase_step)));
    ref_shape[np.argwhere(np.arange(phase_min, phase_max, phase_step) > 0).squeeze()] = -1;
    ref_shape[np.argwhere(np.abs(np.arange(phase_min, phase_max, phase_step)) < 5).squeeze()] = 0;
    score = list()
    for (method_idx, _) in enumerate(methods):
        axes1[method_idx, axes_idx].scatter(np.arange(phase_min, phase_max, phase_step), features[method_idx])
        axes1[method_idx, axes_idx].scatter(np.arange(phase_min, phase_max, phase_step), ref_shape)
        score.append(np.sum(np.abs(features[method_idx] - ref_shape)))
    
    return score

main()


