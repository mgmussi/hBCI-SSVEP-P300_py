'''
Testing for MNE with .CSV files
'''
import mne
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from mne.datasets import sample
from mne.channels import read_custom_montage
from mne.filter import filter_data
from mne.time_frequency import psd_welch
from mne.decoding import Scaler
mne_scaler = Scaler(scalings = 'median')
import pyxdf

import numpy as np
import pandas as pd
pd.options.display.float_format = '{:20.6f}'.format
import csv
from scipy.signal import butter, lfilter

import os
import matplotlib.pyplot as plt
from tkinter import filedialog
import sys

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes#, AggOperation

from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

# Define constants
#1
SPS = 250 #samples per second
NYQ = int(SPS*0.5) #Nyquist ratio
COMP_NUM = 1
CCA_model = CCA(n_components = COMP_NUM, max_iter=20000)
low = 5 / NYQ               #DIFF
high = 30 / NYQ
b, a = butter(4, [low, high], btype='band')

def plot_psd(epochs):
    f, axs = plt.subplots(3, 1, figsize=(10, 10))
    psd1, freq1 = psd_welch(epochs['Hz_15'], n_fft=750, n_per_seg = 250, picks='all')
    psd2, freq2 = psd_welch(epochs['Hz_10'], n_fft=750, n_per_seg = 250, picks='all')
    psd3, freq3 = psd_welch(epochs['Hz_6'], n_fft=750, n_per_seg = 250, picks='all')
    psd1 = 10 * np.log10(psd1)
    psd2 = 10 * np.log10(psd2)
    psd3 = 10 * np.log10(psd3)

    psd1_mean = psd1.mean(0)
    psd1_std = psd1.mean(0)
    psd2_mean = psd2.mean(0)
    psd2_std = psd2.mean(0)
    psd3_mean = psd3.mean(0)
    psd3_std = psd3.mean(0)

    #['P3', 'Pz', 'P4', 'Cz', 'O1', 'O2']
    axs[0].plot(freq1, psd1_mean[0, :], color='b', label='P3')
    axs[0].plot(freq1, psd1_mean[1, :], color='r', label='Pz')
    axs[0].plot(freq1, psd1_mean[2, :], color='g', label='P4')
    axs[0].plot(freq1, psd1_mean[3, :], color='c', label='Cz')
    axs[0].plot(freq1, psd1_mean[4, :], color='k', label='O1')
    axs[0].plot(freq1, psd1_mean[5, :], color='gray', label='O2')

    axs[1].plot(freq2, psd1_mean[0, :], color='b', label='P3')
    axs[1].plot(freq2, psd1_mean[1, :], color='r', label='Pz')
    axs[1].plot(freq2, psd1_mean[2, :], color='g', label='P4')
    axs[1].plot(freq2, psd1_mean[3, :], color='c', label='Cz')
    axs[1].plot(freq2, psd1_mean[4, :], color='k', label='O1')
    axs[1].plot(freq2, psd1_mean[5, :], color='gray', label='O2')

    axs[2].plot(freq3, psd3_mean[0, :], color='b', label='P3')
    axs[2].plot(freq3, psd3_mean[1, :], color='r', label='Pz')
    axs[2].plot(freq3, psd3_mean[2, :], color='g', label='P4')
    axs[2].plot(freq3, psd3_mean[3, :], color='c', label='Cz')
    axs[2].plot(freq3, psd3_mean[4, :], color='k', label='O1')
    axs[2].plot(freq3, psd3_mean[5, :], color='gray', label='O2')

    axs[0].set_title('Channels for 15 Hz')
    axs[1].set_title('Channels for 10 Hz')
    axs[2].set_title('Channels for 6 Hz')

    axs[0].set_ylabel('Power Spectral Density (dB)')
    axs[1].set_ylabel('Power Spectral Density (dB)')
    axs[2].set_ylabel('Power Spectral Density (dB)')

    axs[0].set_xlim((5, 16))
    axs[1].set_xlim((5, 16))
    axs[2].set_xlim((5, 16))

    axs[2].set_xlabel('Frequency (Hz)')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.show()

def validate_classifier(feature_set, label_set, CCA_mode, splits, filename):
    f = open(filename, 'a')

    kf = KFold(n_splits = splits, shuffle = True)
    accs = []
    count = 0

    print(f'\n\n\n\n\'feat_set\' arrary size: {feature_set.shape}')
    print(f'\'label_set\' arrary size: {label_set.shape}')
    f.write('\'feat_set\' arrary size: {}\n'.format(feature_set.shape))
    f.write('\'label_set\' arrary size: {}\n'. format(label_set.shape))

    for train_index, test_index in kf.split(feature_set):
        # print(train_index, test_index)
        new_y = []
        bin_y_test = []
        bin_y_train = []
        rgt = 0
        if CCA_mode:
            print("Training fold CCA", count, "\b...")
            f.write("Number of Components: {}\n".format(COMP_NUM))
            f.write("Training fold CCA {}...\n".format(count))
        else:
            print("Training fold LDA", count, "\b...")
            f.write("Training fold LDA {}...\n".format(count))

        X_train, X_test = feature_set[train_index], feature_set[test_index]
        y_train, y_test = label_set[train_index], label_set[test_index]

        if CCA_mode:
            model = CCA_model.fit(X_train, y_train)
        else:
            model = LDA_model.fit(X_train, y_train)
        y = model.predict(X_test)

        #Conditioning prediction
        y = np.array(y)

        if CCA_mode:
            y = abs(y)
            min_ = min(y)
            max_ = max(y)
            print('\n\nNormalized vals:\n', np.squeeze((y-min_)/(max_-min_)), '\n\n')
            new_y = np.round((y-min_)/(max_-min_)).astype(int)
            new_y = np.squeeze(new_y)

            ##Check
            for i in range(len(y)):
                print(y[i], ' vs. ', y_test[i])
                f.write("{} vs. {}\n".format(y[i], y_test[i]))
            print('\n')
        else:
            for yy in y:
                new_y.append(int(yy))
            bin_y_test = y_test
            #
        ##Check
        print(new_y)        #binary predicted
        print(y_test)   #vs. binary test lbls
        f.write("Pred: {}\n".format(new_y))
        f.write("Lbls: {}\n".format(y_test))

        #Calculating accuracy
        for i in range(len(new_y)):
            if new_y[i] == y_test[i]:
                rgt += 1
        print(rgt, "labels match")
        f.write("{} labels match\n".format(rgt))
        accs.append(rgt/len(y_test))
        if CCA_mode:
            print("ACC CCA = ", '{:.3f}'.format(accs[count]*100), "\b%")
            conf_mat = confusion_matrix(y_test, new_y)#, labels=[1, 2, 4])
            print("<<<<CCA Confusion Matrix\n", conf_mat, '\n\n')
            f.write("ACC CCA = {:.3f}%\n\n<<<<CCA Confusion Matrix\n{}\n\n".format(accs[count]*100, conf_mat))
        else:
            print("ACC LDA = ", '{:.3f}'.format(accs[count]*100), "\b%")
            conf_mat = confusion_matrix(y_test, new_y, labels=[0, 1])
            tn, fp, fn, tp = conf_mat.ravel()
            print("<<<<LDA Confusion Matrix\n", conf_mat)
            print("TN", tn, "FP", fp, "FN", fn, "TP", tp, '\n\n', sep = " : ")
            f.write("ACC LDA = {:.3f}%\n\n<<<<LDA Confusion Matrix\n{}\n\n".format(accs[count]*100, conf_mat))
            f.write("TN: {}, FP: {}, FN: {}, TP: {}\n\n".format(tn, fp, fn, tp))
        count += 1
    accs_mean = np.mean(accs)
    accs_devi = np.std(accs)

    # Rendering Confusion Matrix and Final ACC
    if CCA_mode:
        print("<<<<Average ACC CCA = ", '{:.3f}±{:.3f}'.format(accs_mean*100, accs_devi*100), "\b%")
        f.write("\nAverage ACC CCA = {:.3f}±{:.3f}%\n---------\n---------\n\n\n".format(accs_mean*100, accs_devi*100))
    else:
        print("<<<<Average ACC LDA = ", '{:.3f}±{:.3f}'.format(accs_mean*100, accs_devi*100), "\b%")
        f.write("\nAverage ACC LDA = {:.3f}±{:.3f}%\n---------\n---------\n\n\n".format(accs_mean*100, accs_devi*100))
    f.close()
    return accs, accs_mean


#2
filenames = filedialog.askopenfilenames(initialdir = "C:/Users/atech/Documents/GitHub/SSVEP_EyeGaze-py/Participants",
        title = "Select Trial", filetypes = (("Comma Separated Values", ["*.csv*", "*.xdf"]), ("XDF", "*.xdf"),("all files","*.*")))
all_data = np.array([])
all_stim = []
stim = 0
print('\tExtracting data...')
for file in filenames:
    end_str = file.rindex('/') + 1
    dir_name = file[0:end_str]

    print('\n\tReading file...', dir_name)
    streams, header = pyxdf.load_xdf(file)
    data = streams[0]["time_series"].T
    initial = 256*30
    data = data[0:11, initial:]
    stim_num = int(data.shape[1]/256)
    last = 256*stim_num
    data = np.array(data[:, :last])
    stims = [stim]*stim_num*2 # number of seconds * 2 (trial is every 0.5s)
    for stim in stims:
        all_stim.append(stim)
    stim  += 1

    if all_data.size > 0:
        all_data = np.hstack((all_data, data))
    else:
        all_data = data

all_stim = np.array(all_stim)
all_data = np.array(all_data, dtype=object)
print(">>>All_stims shape: ", all_stim.shape, sep = '')
print(">>>All_data shape: ", all_data.shape, sep = '')


#DIFF
print('\tCreating MNE object...')
montage = mne.channels.make_standard_montage('easycap-M1')
sfreq = float(streams[0]["info"]["nominal_srate"][0])
info = mne.create_info(montage.ch_names[0:11], sfreq, ["eeg"]*11)
raw = mne.io.RawArray(all_data, info)
raw.set_montage(montage)
raw_flt = raw.copy()
#3
raw_flt.load_data().filter(l_freq=5., h_freq=30)


for _ica in range(2):
    #DIFF
    if _ica:
        print('\tFiltering ICA components...')
        # raw_flt.plot_psd()
        ica_flt = ICA(n_components=11, method='fastica', max_iter = 50000) #len(ch_names)
        ica_flt.fit(raw_flt)
        ica_flt.detect_artifacts(raw_flt)
        ica_flt.apply(raw_flt, exclude=ica_flt.exclude)
        # raw_flt.plot_psd()
    print(montage.ch_names[0:11])

    ###### Extracting data from epochs
    print('\tExtracting conditioned data...')
    # Extract filtered epochs from O1, O2:
    filt_data = raw_flt.to_data_frame()
    print(filt_data)
    print(filt_data.shape)


    ssvep_data = []
    #looping through channel combinations
    for ch_one in range(10):
        for ch_two in range(ch_one+1, 11):
            ssvep_data = []
            if _ica:
                _save_path = dir_name + 'results_' + str(ch_one) + '+' + str(ch_two) + '_ICA_CCA_cls.txt'
            else:
                _save_path = dir_name + 'results_' + str(ch_one) + '+' + str(ch_two) + '_RAW_CCA_cls.txt'

            #Sectioning data in 0.5s
            ##USE 2 CHANNELS
            for x in range(128, filt_data.shape[0]+1, 128):
                ssvep_data.append([
                    [y for y in filt_data.iloc[x-128:x, ch_one].values],
                    [y for y in filt_data.iloc[x-128:x, ch_two].values]
                    ])

                ##USE ALL CHANNELS
                # ssvep_data.append([
                #     [y for y in filt_data.iloc[x-128:x, 0].values],
                #     [y for y in filt_data.iloc[x-128:x, 1].values],
                #     [y for y in filt_data.iloc[x-128:x, 2].values],
                #     [y for y in filt_data.iloc[x-128:x, 3].values],
                #     [y for y in filt_data.iloc[x-128:x, 4].values],
                #     [y for y in filt_data.iloc[x-128:x, 5].values],
                #     [y for y in filt_data.iloc[x-128:x, 6].values],
                #     [y for y in filt_data.iloc[x-128:x, 7].values],
                #     [y for y in filt_data.iloc[x-128:x, 8].values],
                #     [y for y in filt_data.iloc[x-128:x, 9].values],
                #     [y for y in filt_data.iloc[x-128:x, 10].values]
                #     ])
                '''
                [Epochs] = 376
                ssvep_data.shape = [Epochs x Ch x Samples] = [376 x 11 x 128]
                '''
            print(np.array(ssvep_data).shape)

            scaled_ssvep = mne_scaler.fit_transform(np.array(ssvep_data))
            new_dim = np.array(ssvep_data).shape[1]*128 #number of channels x 128 samples (0.5s)

            #4
            pp_ssvep_data = np.reshape(scaled_ssvep, (376, new_dim))

            # bin_lbl = []
            # for lbl in all_stim:
            #     bin_lbl.append(d2b(lbl))

            validate_classifier(pp_ssvep_data, np.array(all_stim), 1, 10, _save_path)

# extra = []
# print(plt.get_fignums())
# for i in plt.get_fignums():
#     plt.figure(i)
#     extra = '_ICA000%d_flt.png' % i
#     print(dir_name+extra)
#     plt.savefig(dir_name + extra)
# plt.close('all')

# raw.plot_psd(average=False)

# for epoch in custom_epochs:
#     for channel in epoch:
#         ica.plot_components(channel)
sys.exit()





















    # eog_evoked = create_eog_epochs(raw).average()
    # eog_evoked.apply_baseline(baseline=(None, -0.2))
    # eog_evoked.plot_joint()
    # sys.exit()


"""Create a mne epoch instance from csv file"""
# Add some more information
info['description'] = 'dataset from ' + filename

# Trials were cut from -1.5 to 1.5 seconds
tmin = -1.5

#and convert it to numpy array:
npdata = np.array(df)

#the first 4 columns of the data frame are the
#subject number... subNumber = npdata[:,0]
#trial number... trialNumber = npdata[:,1]
#condition number... conditions = npdata[:,2]
#and sample number (within a trial)... sampleNumber = npdata[:,3]

#sample 1537 is time 0, use that for the event 
onsets = np.array(np.where(npdata[:,3]==1537))
conditions = npdata[npdata[:,3]==1537,2]

#use these to make an events array for mne (middle column are zeros):
events = np.squeeze(np.dstack((onsets.flatten(), np.zeros(conditions.shape),conditions)))

#now we just need EEGdata in a 3D shape (n_epochs, n_channels, n_samples)
EEGdata = npdata.reshape(len(conditions),3072,74)
#remove the first 4 columns (non-eeg, described above):
EEGdata = EEGdata[:,:,4:]
EEGdata = np.swapaxes(EEGdata,1,2)

#create labels for the conditions, 1, 2, and 3:
event_id = dict(button_tone=1, playback_tone=2, button_alone=3)

# create raw object 
custom_epochs = EpochsArray(EEGdata, info=info, events=events.astype('int'), tmin=tmin, event_id=event_id)