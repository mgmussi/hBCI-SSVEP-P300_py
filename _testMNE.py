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
SPS = 250 #samples per second
NYQ = int(SPS*0.5) #Nyquist ratio
COMP_NUM = 2
CCA_model = CCA(n_components = COMP_NUM, max_iter=20000)
low = 5 / NYQ
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

######
def b2d(b_vect):
    str_format = ''
    for b in b_vect:
        str_format = str_format + str(int(b))
    return int(str_format,2)

def d2b(decimal):
    if decimal == '1': b_array = [0, 0, 1]
    elif decimal == '2': b_array = [0, 1, 0]
    elif decimal == '4': b_array = [1, 0, 0]
    return b_array
######

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
        if CCA_mode:
            for yy in y:
                yy = abs(yy)
                marg = np.argmax(yy)
                oarg = [x for x in range(len(yy)) if x != marg]
                yy[marg] = 1
                yy[oarg] = 0
                new_y.append(b2d(yy))
            #Conditioning labels
            for ll in y_test:
                bin_y_test.append(b2d(ll))
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
        print(bin_y_test)   #vs. binary test lbls
        f.write("Pred: {}\n".format(new_y))
        f.write("Lbls: {}\n".format(bin_y_test))

        #Calculating accuracy
        for i in range(len(new_y)):
            if new_y[i] == bin_y_test[i]:
                rgt += 1
        print(rgt, "labels match")
        f.write("{} labels match\n".format(rgt))
        accs.append(rgt/len(bin_y_test))
        if CCA_mode:
            print("ACC CCA = ", '{:.3f}'.format(accs[count]*100), "\b%")
            conf_mat = confusion_matrix(bin_y_test, new_y)#, labels=[1, 2, 4])
            print("<<<<CCA Confusion Matrix\n", conf_mat, '\n\n')
            f.write("ACC CCA = {:.3f}%\n\n<<<<CCA Confusion Matrix\n{}\n\n".format(accs[count]*100, conf_mat))
        else:
            print("ACC LDA = ", '{:.3f}'.format(accs[count]*100), "\b%")
            conf_mat = confusion_matrix(bin_y_test, new_y, labels=[0, 1])
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


filenames = filedialog.askopenfilenames(initialdir = "C:/Users/atech/Documents/GitHub/SSVEP_EyeGaze-py/Participants",
        title = "Select Trial", filetypes = (("Comma Separated Values", ["*.csv*", "*.xdf"]), ("XDF", "*.xdf"),("all files","*.*")))

for file in filenames:
#\\\\\\\\\\\\\\\\\\\\\\\\\
    eeg_data_all = []
    ini_str = file.rindex('/') + 1
    end_str = file.index('_', ini_str) + 6
    dir_name = file[0:end_str]
    print('\n\tReading file...', dir_name)

    # Read EEG file
    df = pd.read_csv(file, header = 0)

    # Select the relevant channels
    #CYTON
    # eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
    # eeg_channels = [x+1 for x in eeg_channels]
    # eeg_channels.pop()
    # eeg_channels.pop()
    # ch_names = ['P3', 'Pz', 'P4', 'Cz', 'O1', 'O2'] #BoardShim.get_eeg_names(BoardIds.CYTON_BOARD.value)
    # sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)

    #G.TEC
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD.value)
    eeg_channels = [x+1 for x in eeg_channels]
    ch_names = BoardShim.get_eeg_names(BoardIds.UNICORN_BOARD.value)
    sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
#\\\\\\\\\\\\\\\\\\\\\\\\

########
    # # Get Data in Chunks (relevant data only)
    # eeg_data = []
    # ts_data = []
    # idx2 = df.index[df['P300_ts'] == 2]
    # counter = 0
    # for j in idx2:
    #     ## Get the last 125 samples before ending timestamp
    #     eeg_data.append([
    #         [x[0] for x in df.loc[j-NYQ+1:j, ['CH1']].values],
    #         [x[0] for x in df.loc[j-NYQ+1:j, ['CH2']].values],
    #         [x[0] for x in df.loc[j-NYQ+1:j, ['CH3']].values],
    #         [x[0] for x in df.loc[j-NYQ+1:j, ['CH4']].values],
    #         [x[0] for x in df.loc[j-NYQ+1:j, ['CH5']].values],
    #         [x[0] for x in df.loc[j-NYQ+1:j, ['CH6']].values]])
    #     ts_data.append([t[0] for t in list(df.loc[j-NYQ+1:j, ['Timestamps']].values)])
    # eeg_data = np.array(eeg_data)
    # ts_data = np.array(ts_data)
########

#\\\\\\\\\\\\\\\\\\\\\\\\
    # Get all data
    eeg_data_all_raw = df.iloc[1:, eeg_channels].values
    eeg_data_all_raw = eeg_data_all_raw.transpose()
    eeg_data_all_raw = eeg_data_all_raw.reshape(len(eeg_channels),-1)
#\\\\\\\\\\\\\\\\\\\\\\\\

########
    # # Filter all data <shape = [6 x Samples]>
    # eeg_data_all_flt = np.array(lfilter(b, a, eeg_data_all, axis = 1)) #filter horizontally [axis = 1]

    # # Filter epochs <shape = [Epochs x 6 x Samples]>
    # eeg_data_flt = np.zeros(eeg_data.shape)
    # for x in range(len(eeg_data)):
    #     eeg_data_flt[x] = np.array(lfilter(b, a, eeg_data[x], axis = 1)) #filter horizontally [axis = 1]

    # # Convert to V for MNE
    # eeg_data = eeg_data / 1000000 # BrainFlow returns uV
    # eeg_data_flt = eeg_data_flt / 1000000 # BrainFlow returns uV
    # eeg_data_all_flt = eeg_data_all_flt / 1000000 # BrainFlow returns uV
########
    
#\\\\\\\\\\\\\\\\\\\\\\\\
    # Converting data from uV to V
    eeg_data_all_raw = eeg_data_all_raw / 1000000 # BrainFlow returns uV
#\\\\\\\\\\\\\\\\\\\\\\\\

    # Scaling data
    # eeg_data_all = scaler.fit_transform(eeg_data_all_raw.transpose())
    # eeg_data_all = eeg_data_all.transpose()

    # plt.figure()
    # plt.plot(eeg_data_all_raw.transpose(), 'k')
    # plt.title('Raw Data')
    # plt.figure()
    # plt.plot(eeg_data_all.transpose(), 'b')
    # plt.title('Scaled Data')

#\\\\\\\\\\\\\\\\\\\\\\\\
    print('\tCreating MNE object...')
    ## Creating MNE objects
    # Creating 'Events' array
    idx1 = df.index[df['P300_ts'] == 0]
    conditions = df.loc[idx1, ['SSVEP_labels']].values
    onsets = df.iloc[idx1, 0].values
    events = np.squeeze(np.dstack((onsets.flatten(), np.zeros(conditions.shape).flatten(), conditions.flatten())))
    ch_types = ['eeg'] * len(eeg_channels)
    n_channels = len(ch_names)
    info = mne.create_info(ch_names = ch_names, sfreq=250., ch_types='eeg')
    # info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types = ch_types, montage = montage)
    event_id = dict(Hz_15 = 1, Hz_10 = 2, Hz_6 = 4)
    montage = mne.channels.make_standard_montage('standard_1020')
#\\\\\\\\\\\\\\\\\\\\\\\\

    # streams, header = pyxdf.load_xdf(file)
    # data = streams[0]["time_series"].T
    # initial = 256*30
    # data = data[0:11, initial:]
    # print(data)
    # print(data.shape)
    # plt.plot(data[0:11].T)
    # plt.show()
    # sys.exit()
    # montage = mne.channels.make_standard_montage('easycap-M1')
    # sfreq = float(streams[0]["info"]["nominal_srate"][0])
    # info = mne.create_info(montage.ch_names[0:11], sfreq, ["eeg"]*11)
    # raw = mne.io.RawArray(data, info)
    # raw.set_montage(montage)
    # raw.plot(scalings=dict(eeg=100e-6), duration=2, start=100)
    # conditions = np.array([1, 2])
    # onsets = np.array([0, 2])
    # events = np.squeeze(np.dstack((onsets.flatten(), np.zeros(conditions.shape).flatten(), conditions.flatten())))

#////////////////////////
    # # Create RAW and Epochs
    raw = mne.io.RawArray(eeg_data_all_raw, info)
    custom_epochs = mne.Epochs(raw, events=events.astype('int'), event_id=event_id, tmin=0, tmax =0.5, baseline=(0, 0), preload=True)
    custom_epochs.set_montage(montage)
#///////////////////////


    # Create RAW_filtered and Epochs_filtered
    raw_flt = raw.copy()
    raw_flt.load_data().filter(l_freq=5., h_freq=30)
    raw.plot_psd()
    raw_flt.plot_psd()

    custom_epochs_flt = mne.Epochs(raw_flt, events=events.astype('int'), tmin=0, tmax =0.5, baseline=(0, 0), preload=True)
    custom_epochs_flt.set_montage(montage)

    
    # filtered = raw_flt.to_data_frame(picks = [0, 1, 2 ,3, 4, 5])
    # filtered_plot = filtered.iloc[0:, [1, 2 ,3, 4, 5, 6]].values
    # plt.figure()
    # plt.plot(filtered_plot, 'b')
    # plt.title('Filtered Data')
    
########
    # # Create Raw's and Epoch's using EpochsArrays
    # raw = mne.io.RawArray(eeg_data_all, info)
    # raw_flt = mne.io.RawArray(eeg_data_all_flt, info)
    # custom_epochs = mne.EpochsArray(eeg_data, info=info, events=events.astype('int'), tmin=0, event_id=event_id)
    # custom_epochs.set_montage(montage)
    # custom_epochs_flt = mne.EpochsArray(eeg_data_flt, info=info, events=events.astype('int'), tmin=0, event_id=event_id)
    # custom_epochs_flt.set_montage(montage)
########

    # ICA
    # ica = ICA(n_components=len(ch_names), method='fastica', max_iter = 5000)
    # ica.fit(custom_epochs)
    # custom_epochs_flt.plot(scalings = dict(eeg=50e-3))

    
    ica_flt = ICA(n_components=len(eeg_channels), method='fastica', max_iter = 50000) #len(ch_names)
    ica_flt.fit(custom_epochs_flt)
    ica_flt.detect_artifacts(raw_flt)
    print(ica_flt.exclude)
    # new_raw = raw_flt.copy()
    # ica_flt.apply(new_raw)

    # plot_psd(custom_epochs_flt)

    ica_flt.apply(custom_epochs_flt, exclude=ica_flt.exclude)
    ica_flt.apply(raw_flt, exclude=ica_flt.exclude)
    raw_flt.plot_psd()

    # plot_psd(custom_epochs_flt)
    # plot_psd(custom_epochs_comp)

    # custom_epochs_comp.plot(scalings = dict(eeg=50e-3))

    # wait = ''
    # while wait != 'q':
    #     wait = input("Press \'q\' to exit.")
    # sys.exit()
    # ica.plot_properties(raw, picks =[0, 1, 2, 3, 4, 5])
    # ica_flt.plot_properties(raw_flt, picks =[0, 1, 2, 3, 4, 5])

    # sys.exit()

    
###### Extracting data from epochs
    print('\tExtracting conditioned data...')
    # Extract filtered epochs from O1, O2:
    #CYTON
    # filt_data = custom_epochs_flt.to_data_frame(picks = [4, 5])

    #G.TEC
    filt_data = custom_epochs_flt.to_data_frame(picks = [5, 6, 7])
    
    # Epochs
    epoch_lst = filt_data.loc[:, ['epoch']].values
    epoch_lst = np.squeeze(epoch_lst)
    seen = set()
    seen_add = seen.add
    s_epoch_lst = [x for x in epoch_lst if not (x in seen or seen_add(x))]

    ssvep_data = []
    lbl_lst = []

    #CYTON
    # for epoch in s_epoch_lst:
    #     lbl_lst.append(list(set( list(np.squeeze(filt_data.loc[filt_data['epoch'] == epoch, ['condition']].values))))[0])
    #     ssvep_data.append([
    #         [x[0] for x in filt_data.loc[filt_data['epoch'] == epoch, ['O1']].values],
    #         [x[0] for x in filt_data.loc[filt_data['epoch'] == epoch, ['O2']].values]])

    #G.TEC 'PO7', 'Oz', 'PO8'
    for epoch in s_epoch_lst:
        lbl_lst.append(list(set( list(np.squeeze(filt_data.loc[filt_data['epoch'] == epoch, ['condition']].values))))[0])
        ssvep_data.append([
            [x[0] for x in filt_data.loc[filt_data['epoch'] == epoch, ['PO7']].values],
            [x[0] for x in filt_data.loc[filt_data['epoch'] == epoch, ['Oz']].values],
            [x[0] for x in filt_data.loc[filt_data['epoch'] == epoch, ['PO8']].values]])
    '''
    len(lbl_lst) = [Epochs] = 189
    len(epoch_lst) = [Epochs] = 189
    SSVEP_data.shape = [Epochs x Ch x Samples] = [189x2x126]
    '''

    scaled_ssvep = mne_scaler.fit_transform(np.array(ssvep_data))
    pp_ssvep_data = np.reshape(scaled_ssvep, (scaled_ssvep.shape[0], scaled_ssvep.shape[2]*scaled_ssvep.shape[1]))
    # plt.figure()
    # plt.plot(np.reshape(scaled_ssvep, (2,-1)).transpose(), 'r')
    # plt.title('Filtered & Scaled Data')
    # plt.show()

    bin_lbl = []
    for lbl in lbl_lst:
        bin_lbl.append(d2b(lbl))

    # # Pick EEG signal
    # picks = mne.pick_types(custom_epochs.info, eeg=True)
    
    _save_path = dir_name + '_cls.txt'
    validate_classifier(pp_ssvep_data, np.array(bin_lbl), 1, 10, _save_path)
    
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