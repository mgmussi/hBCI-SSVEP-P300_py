'''
ON THIS VERSION:
- Animating EEG signals
- Using different graphs for each EEG channel
'''

import sys
import os
from tkinter import filedialog
import pandas as pd
pd.options.display.float_format = '{:20.6f}'.format
import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
import matplotlib.pyplot as plt

# import scipy
from scipy.signal import butter, lfilter, kaiserord, firwin, iirnotch, freqz
from scipy import interpolate
from scipy.fft import fft, fftfreq

from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import FastICA
ica = FastICA(n_components=7, random_state=0, tol=0.05)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

import mne
from mne.decoding import Scaler
mne_scaler = Scaler(scalings = 'median')

from datetime import datetime
import time

import matplotlib.animation as ani
# animator = ani.FuncAnimation(fig, chartfunc, interval = 100)

import warnings
warnings.filterwarnings('ignore')

import pickle
import cProfile, pstats, io
import math
import winsound

#1
SPS = 250
NYQ = int(SPS*0.5)
win = 75 #padding
#
SLICE = NYQ
#
COMP_NUM = 2
LDA_model = LDA(solver = 'svd')
CCA_model = CCA(n_components = COMP_NUM, max_iter=20000)

#3
low = 5 / NYQ
high = 30 / NYQ

#Notch
b1, a1 = iirnotch(60, 1, SPS)
junk = 3

freq, h = freqz(b1, a1, fs=SPS)

##::BUTTERWORTH----------------------------
# b, a = butter(1, [low, high], btype='bandpass')#, fs = 250)#, analog=True)# # b, a = butter(order, [low, high], btype='band')
# delay = 0


##::FIR------------------------------------
# win = 75 #padding
# Eq. 1 method
# N = int((SPS/5)*(25/22)) #(fs/transition band width)*(attenuation (dB)/22)
# b = firwin(N, [low, high], pass_zero = 'bandpass', window = "hamming")

# #Kaiser window method
width = 2/NYQ #transition width
ripple_db = 11 #attenuation
N, beta = kaiserord(ripple_db, width)
b = firwin(N, [low, high], pass_zero = 'bandpass',  window=('kaiser', beta))
delay = int(0.5 * (N-1)) # / SPS
print('>>FIR Order: ', N)

print('>>Delay    : ', delay)

# # Frequency response
# freq, h = freqz(b, fs=SPS)
# # freq, h = freqz(b, a, fs=SPS)
# # Plot
# fig, axis = plt.subplots(2, 1, figsize=(8, 6))
# axis[0].plot(freq, 20*np.log10(abs(h)), color='blue')
# axis[0].set_title("Frequency Response")
# axis[0].set_ylabel("Amplitude (dB)", color='blue')
# # axis[0].set_xlim([0, 100])
# # axis[0].set_ylim([-25, 10])
# axis[0].grid()
# axis[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
# axis[1].set_ylabel("Angle (degrees)", color='green')
# axis[1].set_xlabel("Frequency (Hz)")
# # axis[1].set_xlim([0, 100])
# # axis[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
# # axis[1].set_ylim([-90, 90])
# axis[1].grid()
# plt.show()

def get_data_files():
	ch_data_125 = []
	ch_data_all = []
	tmstamp_all = []
	sampnum_all = []
	ts_data = []
	p300ll_data = []
	ssvepll_data = []
	ssvepll_data_all = []
	curr_id = 0

	##::Dealing with file location
	filename = filedialog.askopenfilename(initialdir = "C:/Users/atech/Documents/GitHub/SSVEP_EyeGaze-py/Participants",
		title = "Select Trial", filetypes = (("Comma Separated Values", "*.csv*"), ("all files","*.*")))
	dir_name = filename

	##::Getting data out of file
	print(filename)
	df = pd.read_csv(filename)

	## Get df indexes:
	idx1 = df.index[df['P300_ts'] == 0]
	idx2 = df.index[df['P300_ts'] == 2]
	ini_stim = df.loc[idx1[0], ['Timestamps']].values[0]
	ssvepll = df.loc[idx1[0], ['SSVEP_labels']].values[0]
	label_info = []
	print('\t\u0298 ', ini_stim)

	counter = 0

	ch_data_all = [[x[0] for x in df.loc[:, ['CH1']].values], #[chs x samples]
			[x[0] for x in df.loc[:, ['CH2']].values],
			[x[0] for x in df.loc[:, ['CH3']].values],
			[x[0] for x in df.loc[:, ['CH4']].values],
			[x[0] for x in df.loc[:, ['CH5']].values],
			[x[0] for x in df.loc[:, ['CH6']].values],
			[x[0] for x in df.loc[:, ['CH7']].values],
			[x[0] for x in df.loc[:, ['CH8']].values]]
	
	tmstamp_all = [x[0] for x in df.loc[:, ['Timestamps']].values]
	sampnum_all = [x[0] for x in df.loc[:, ['Samples']].values]

	counter = 0
	for i, j in zip(idx1, idx2):
		## Get start and end of SSVEP stimuli
		if counter % 20 == 0 and counter != 0:
			fin_stim = df.loc[j, ['Timestamps']].values[0]
			label_info.append([ini_stim, fin_stim, ssvepll])
		if counter % 21 == 0:
			ini_stim = df.loc[i, ['Timestamps']].values[0]
			ssvepll = df.loc[i, ['SSVEP_labels']].values[0]
		counter += 1

	# 	ts_data.append([t[0] for t in list(df.loc[j-NYQ+1:j, ['Timestamps']].values)])
	# 	p300 = df.loc[i, ['P300_labels']].values
	# 	p300ll_data.append(p300[0])
	# 	ssvep = df.loc[i, ['SSVEP_labels']].values
	# 	ssvepll_data.append(ssvep[0])

	# tmstamp_all = np.array(tmstamp_all)
	# ch_data_all = np.array(ch_data_all)
	# ts_data = np.array(ts_data)
	# p300ll_data = np.array(p300ll_data)
	# ssvepll_data = np.array(ssvepll_data)
	# ssvepll_data_all = np.array(ssvepll_data_all)
	label_info = np.array(label_info)
	# print(sampnum_all, end = '\n')
	return ch_data_all, tmstamp_all, sampnum_all, filename, label_info


def profile(fnc):
	def inner(*args, **kwargs):
		pr = cProfile.Profile()
		pr.enable()
		retval = fnc(*args, **kwargs)
		pr.disable()
		s = io.StringIO()
		sortby = 'cumulative'
		ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
		ps.print_stats()
		print(s.getvalue())
		return retval
	return inner


if __name__ == '__main__':
	##::Defining channels
	'''
	#Paradigm		#IDEAL CHs 				#CYTON 				#GTEC
	#P300 		 	#PO8, PO7, POz, CPz		#P3, Pz, P4, Cz		#C3, Cz, C4, Pz
	#SSVEP 			#O1, Oz, O2 			#O1, O2 			#O1, O2, Oz
	'''
	#CYTON
	# chs_p300 = [1,2,3,4]
	# chs_ssvep = [5,6]
	chs_p300 = [2,3,4,5]
	chs_ssvep = [6,7]
	chs_names = ['Empty', 'P3', 'Pz', 'P4', 'Cz', 'O1', 'O2', 'Empty']

	#G.TEC
	# chs_p300 = [2,3,4,5]
	# chs_ssvep = [6,7,8]
	# chs_names = ['C3', 'Cz', 'C4', 'Pz', 'O1', 'O2', 'Oz', 'Empty']

	chs_all = list(set(chs_p300 + chs_ssvep))
	_chs_p300 = list(range(len(chs_p300)))
	_chs_ssvep = list(range(_chs_p300[-1]+1, _chs_p300[-1]+1+len(chs_ssvep)))
	_chs_all = [x - 1 for x in chs_all]
	cca_lbls = []
	cca_lbls_all = []

	##::Receives ALL samples from .csv
	ch_data_all, tmstamp_all, sampnum_all, dir_, label_info = get_data_files()

	##::Check for pre-saved file
	dir__ = dir_.replace('.csv', '.pkl')
	##::Setting folder to save images
	new_folder = dir_[0:-4]
	if not os.path.exists(new_folder):
		os.makedirs(new_folder)

	if os.path.isfile(dir__):
		print('Opening file...')
		winsound.Beep(277, 150)
		winsound.Beep(391, 150)
		winsound.Beep(493, 150)
		with open(dir__, 'rb') as input:
			ch_data_sliced, tmstamp_sliced, samples_sliced, xfft, yfft, yfft_f, yfft_n, flt_data, norm_data = pickle.load(input)
	else:
		print('Extracting data from file...')
		##::Create data slices
		ch_data_sliced = []
		tmstamp_sliced = []
		samples_sliced = []
		# for idx in range(len(ch_data_all[0])-125):
		for idx in range(0, len(ch_data_all[0])-NYQ, 8):
			ch_data_sliced.append([ch_data_all[k][idx:idx+125] for k in range(len(ch_data_all))])
			tmstamp_sliced.append(tmstamp_all[idx:idx+125])
			samples_sliced.append(sampnum_all[idx:idx+125])

		##::Converting to array
		ch_data_all = np.array(ch_data_all)
		tmstamp_all = np.array(tmstamp_all)
		ch_data_sliced = np.array(ch_data_sliced)
		tmstamp_sliced = np.array(tmstamp_sliced)
		samples_sliced = np.array(samples_sliced)
		# print('ch_data_sliced size: ', ch_data_sliced.shape)

		##::Filtering and creating fft for each section
		xfft = fftfreq(SLICE, 1/SPS)
		# xfft = fftfreq(SPS, 1/SPS)
		yfft = np.empty(ch_data_sliced.shape, float)
		# yfft = np.empty((ch_data_sliced.shape[0], ch_data_sliced.shape[1], 250), float)
		#If using FIR, pre-location accounts for FIR induced delay; else, it will use delay = 0
		yfft_f = np.empty((ch_data_sliced.shape[0], ch_data_sliced.shape[1], ch_data_sliced.shape[2]-delay-junk), float)
		yfft_n = np.empty((ch_data_sliced.shape[0], ch_data_sliced.shape[1], ch_data_sliced.shape[2]-delay-junk), float)
		notch_data = np.empty((ch_data_sliced.shape[0], ch_data_sliced.shape[1], ch_data_sliced.shape[2]-junk), float)
		flt_data = np.empty((ch_data_sliced.shape[0], ch_data_sliced.shape[1], ch_data_sliced.shape[2]-delay-junk), float)
		norm_data = np.empty((ch_data_sliced.shape[0], ch_data_sliced.shape[1], ch_data_sliced.shape[2]-delay-junk), float)

		# print(">>pre-located space: ", flt_data.shape)
		

		for x in range(ch_data_sliced.shape[0]):

			if x % 125 == 0:
				msg = "Transforming slices |" + '\u2588' * math.floor( ((x/ch_data_sliced.shape[0]*100) % 100)/2 ) + '\u2591' * math.ceil(50-( ((x/ch_data_sliced.shape[0]*100) % 100)/2 )) + '| slice ' + str(x) + ' ~ '+'{:.2f}%'.format(x/ch_data_sliced.shape[0]*100)
				# msg = "Transforming slice " + str(x) + '/' + str(ch_data_sliced.shape[0])
				print(msg, end="\r")

			# plt.plot(ch_data_sliced[x, :, :].T)
			# plt.show()

			yfft[x,:,:] = np.abs(fft(ch_data_sliced[x, :, :], axis = 1))**2 #raw data FFT 
			# yfft[x,:,:] = np.abs(fft(np.pad(ch_data_sliced[x, :, :], [(0,0),(0, 250-ch_data_sliced[x, :, :].shape[1])], 'constant'), axis = 1))**2 #raw data FFT

			#notch filter - attempt 2
			# prefilt = np.hstack(( np.flip(ch_data_sliced[x, :, 0:win], axis = 1), ch_data_sliced[x, :, :], np.flip(ch_data_sliced[x, :, -win:], axis = 1) ))
			prefilt = np.pad(ch_data_sliced[x, :, :], [(0,0), (win, win)], 'reflect')
			b_notch = lfilter(b1, a1, prefilt, axis = 1)
			notch_data[x,:,:] = b_notch[:,win+junk:-win]

			# plt.plot(notch_data[x,:,:].T)
			# plt.show()

			yfft_notch = np.abs(fft(notch_data[x,:,:], axis = 1))**2
			# prefilt = np.hstack(( np.flip(s_notch[:, 0:win], axis = 1), s_notch[:, :], np.flip(s_notch[:, -win:], axis = 1) ))
			prefilt = np.pad(notch_data[x,:,:], [(0,0), (win, win)], 'reflect')
			#butterworth
			# posfilt = lfilter(b, a, prefilt, axis = 1)
			#or
			#FIR
			posfilt = lfilter(b, [1.0], prefilt, axis = 1)
			flt_data[x,:,:] = posfilt[:, win+delay:-win] #shape << [chs, samples]

			# plt.plot(flt_data[x,:,:].T)
			# plt.show()

			yfft_f[x,:,:] = np.abs(fft(flt_data[x,:,:], axis = 1))**2 #filtered data FFT


			##CHANGES
			>>>use np.pad() instead of np.flip() for padding
			>>>inverted notch_data with b_notch
			>>>added a delay compensation for notch filter (='junk')
			>>>using scaler instead of mne_scaler
			>>>tried padding data before fft (no success)
			>>>averaded 3 fft trials to smooth the animation


			# #temp plot 
			# if x > 3000:
			# 	use_frq1 = [i for i in range(len(xfft)) if xfft[i]>=5 and xfft[i]<50]
			# 	use_frq2 = [i for i in range(len(xfft_2)) if xfft_2[i]>=5 and xfft_2[i]<50]
			# 	t1 = xfft >= 0
			# 	t2 = xfft_2 >= 0

			# 	#temp
			# 	fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9,1, figsize=(16,9),gridspec_kw={'height_ratios': [1,1,1,1,1,1,1,1,2]}, constrained_layout=True)
			# 	count = 0
			# 	for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
			# 		#ch_data_sliced[x, count, :] /.vs.\ notch_data[x,count,:]
			# 		if count <= 7:
			# 			ax.set_ylim(min(flt_data[x,count,:]), max(flt_data[x,count,:]))
			# 			ax.set_xlim(tmstamp_sliced[x,0], tmstamp_sliced[x,-1])
			# 			ax.plot(tmstamp_sliced[x,:], ch_data_sliced[x,count,:], 'b', tmstamp_sliced[x,:], notch_data[x, count, win:-win].T, 'r', tmstamp_sliced[x,delay:], flt_data[x,count,:].T, 'g')
			# 		else:
			# 			max_y2 = np.amax(yfft[x, :, use_frq1])
			# 			min_y2 = np.amin(yfft[x, :, use_frq1])
			# 			diff_y = abs(max_y2 - min_y2)*0.1
			# 			max_new = max_y2 + diff_y
			# 			min_new = min_y2 - diff_y
			# 			ax.set_ylim(0, max_new)
			# 			ax.set_xlim(0, 65)
			# 			ax.plot(xfft[t1], yfft[x, :, t1], 'b') #, xfft[t1], yfft_notch[:, t1].T, 'r', xfft_2[t2], yfft_f[x,:,t2], 'g')
			# 			ax.plot()
			# 		count += 1
			# 	plt.show()

			# 	fig = plt.figure()

			# 	ax1 = plt.subplot(3,1,1)
			# 	plt.plot(xfft[t1], yfft[x,:,t1])
			# 	max_y1 = np.amax(yfft[x,:,use_frq1])
			# 	min_y1 = np.amin(yfft[x,:,use_frq1])
			# 	diff_y = abs(max_y1 - min_y1)*0.1
			# 	max_new = max_y1 + diff_y
			# 	min_new = min_y1 - diff_y
			# 	ax1.set_ylim(0, max_new)

			# 	ax2 = plt.subplot(3,1,2)
			# 	plt.plot(xfft[t1], yfft_notch[:, t1].T)
			# 	max_y2 = np.amax(yfft_notch[:, use_frq1])
			# 	min_y2 = np.amin(yfft_notch[:, use_frq1])
			# 	diff_y = abs(max_y2 - min_y2)*0.1
			# 	max_new = max_y2 + diff_y
			# 	min_new = min_y2 - diff_y
			# 	ax2.set_ylim(0, max_new)


			# 	ax3 = plt.subplot(3,1,3)
			# 	plt.plot(xfft_2[t2], yfft_f[x,:,t2])
			# 	max_y3 = np.amax(yfft_f[x,:,use_frq2])
			# 	min_y3 = np.amin(yfft_f[x,:,use_frq2])
			# 	diff_y = abs(max_y3 - min_y3)*0.1
			# 	max_new = max_y3 + diff_y
			# 	min_new = min_y3 - diff_y
			# 	ax3.set_ylim(0, max_new)

			# 	plt.show()

			# ##:: 2 - ICA (remove bad components?)
			# comps = ica.fit_transform(flt_data[x,:,:].T)
			# #plot the ICA components & save them
			# count = 0
			# fig = plt.figure()
			# plt.title("Epoch " + str(x))
			# for comp in comps.T:
			# 	count += 1
			# 	plt.subplot(2, 4, count)
			# 	plt.plot(comp) 
			# 	plt.title("Comp " + str(count))
			# _img = u'Epoch_%s.png' % x
			# _img_path = os.path.join(new_folder, _img)
			# plt.savefig(_img_path)
			# # plt.show()
			# #removes bad components
			# remove_indices = [0, 3, 4, 6]
			# comps[:, remove_indices] = 0
			# flt_data_rest[x,:,:] = ica.inverse_transform(comps).T
			
			# #plot chs to compare w/ and wo/ ica filter
			# fig = plt.figure()
			# plt.title("Epoch " + str(x))
			# for ch in range(len(_chs_all)):
			# 	plt.subplot(2, 4, ch+1)
			# 	plt.plot(flt_data[x, ch, 2:].T, 'b')
			# 	plt.plot(flt_data_rest[x, ch, 2:].T, 'r')
			# 	plt.title("Channel " + str(ch))
			# _img = u'Ch_comparison_%s.png' % x
			# _img_path = os.path.join(new_folder, _img)
			# plt.savefig(_img_path)


			##:: 3 - SCALE/NORMALIZE
			# norm_data[x,:,:] = scaler.fit_transform(flt_data[x,:,:])
			norm_data[x,:,:] = scaler.fit_transform(flt_data[x,:,:].T).T

			# plt.plot(norm_data[x,:,:].T)
			# plt.show()

			# norm_data[x,:,:] = np.squeeze(mne_scaler.fit_transform(flt_data[x,:,:]))
			yfft_n[x,:,:] = np.abs(fft(norm_data[x,:,:], axis = 1))**2 #normalized data FFT
		print('')
		##::Converting to array
		norm_data = np.array(norm_data)
		flt_data = np.array(flt_data)

		##::Save
		with open(dir__, 'wb') as output:
			pickle.dump([ch_data_sliced, tmstamp_sliced, samples_sliced, xfft, yfft, yfft_f, yfft_n, flt_data, norm_data], output, pickle.HIGHEST_PROTOCOL)
			print('Data saved \u00B6')
		winsound.Beep(277, 150)
		winsound.Beep(391, 150)
		winsound.Beep(493, 150)


	#plot timestamp distribution
	# oness = np.ones(len(tmstamp_all))
	# plt.plot(tmstamp_all, oness, 'x')
	# plt.show()
	# sys.exit()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---##::Plotting animation::##--------------------------------------------------------------------------------------------------------------------------------------------------------#
	# Figure with subplots
	fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9,1, figsize=(16,9),gridspec_kw={'height_ratios': [1,1,1,1,1,1,1,1,2]}, constrained_layout=True)

	# intialize two line objects (one in each axes)
	colors = ['#FF095F', '#FF3C15', '#FF552B', '#FF7A41', '#FFAE23', '#81BB3E', '#7C8F2F', '#C50000']
	line1, = ax1.plot([], [], lw=2, color = colors[0])
	line2, = ax2.plot([], [], lw=2, color = colors[1])
	line3, = ax3.plot([], [], lw=2, color = colors[2])
	line4, = ax4.plot([], [], lw=2, color = colors[3])
	line5, = ax5.plot([], [], lw=2, color = colors[4])
	line6, = ax6.plot([], [], lw=2, color = colors[5])
	line7, = ax7.plot([], [], lw=2, color = colors[6])
	line8, = ax8.plot([], [], lw=2, color = colors[7])

	line9, = ax9.plot([], [], lw=1, color = colors[0])
	line10, = ax9.plot([], [], lw=1, color = colors[1])
	line11, = ax9.plot([], [], lw=1, color = colors[2])
	line12, = ax9.plot([], [], lw=1, color = colors[3])
	line13, = ax9.plot([], [], lw=1, color = colors[4])
	line14, = ax9.plot([], [], lw=1, color = colors[5])
	line15, = ax9.plot([], [], lw=1, color = colors[6])
	line16, = ax9.plot([], [], lw=1, color = colors[7])

	line = [line1, line2, line3, line4, line5, line6, line7, line8,
			line9, line10, line11, line12, line13, line14, line15, line16]

	# axes initalization
	ch_count = 0
	for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
		if ch_count < len(chs_names):
			# title = 'CH' + str(ch_count)
			title = chs_names[ch_count]
			ax.set_ylabel(title)
			ax.set_ylim(-100, 100)
			ax.set_xlim(tmstamp_sliced[0,0], tmstamp_sliced[0,-1])
			# Turn off tick labels
			ax.set_yticklabels([])
			if not ch_count == len(chs_names)-1:
				ax.set_xticklabels([])
			else:
				ax.set_xlabel('Samples')
		else:
			ax.set_ylabel('FFT')
			ax.set_xticks([6, 10, 15, 20, 30, 60], minor = False)
			ax.set_xticks([12, 18, 24, 40, 50], minor = True)
			ax.set_xlim(0, 65)
			ax.set_ylim(-0.05, 1)
		ax.grid()
		ch_count += 1
	

#---##::Quick access variables-----
	
	eeg_data = flt_data  #ch_data_sliced, #norm_data, #
	tmst_data = tmstamp_sliced[:, delay+junk:] #tmstamp_sliced, #samples_sliced, #
	fft_data = yfft_f #yfft_n #yfft, #yfft_f
	xfft_2 = fftfreq(SLICE-delay-junk, 1/SPS)
	use_frq = [i for i in range(len(xfft_2)) if xfft_2[i]>=5 and xfft_2[i]<50]
	# print(use_frq)
	# use_frq = xfft_2>=3 and xfft_2<50
	init = 3000

	#Array size checker
	# print('ch_data_sliced size: ', ch_data_sliced.shape)
	# print('flt_data size: ', flt_data.shape)
	# print('norm_data size: ', norm_data.shape)
	# print('tmstamp_sliced size: ', tmstamp_sliced.shape)
	# print('samples_sliced size: ', samples_sliced.shape)
	# print('yfft size: ', yfft.shape)
	# print('yfft_f size: ', yfft_f.shape)
	# print('yfft_n size: ', yfft_n.shape)
	# print('xfft_2 size: ', xfft_2.shape)

	print('eeg_data size: ', eeg_data.shape)
	print('tmst_data size: ', tmst_data.shape)
	print('fft_data size: ', fft_data.shape)
	print('xfft_2 size: ', xfft_2.shape)

#--------------------------------

	##::Generate frames with plot
	##SAVE TO FOLDER: 'new_folder'
	FRAMES = ch_data_sliced.shape[0]
	k = 0
	vline = ax9.axvline(-1, linestyle ='--', color = 'c')
	for i in range(init, FRAMES): #<<skip frames
		counter = 0
		for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
			if counter <= len(chs_names)-1:

				#display frequency that was shown
				if tmstamp_sliced[i,-1] >= label_info[k,0] and not(tmstamp_sliced[i,0] >= label_info[k,1]):
					title = 'Frq. #' + str(int(label_info[k,2]))
					ax1.set_title(title)
					if int(label_info[k,2]) == 1:
						vline.set_xdata(15)
					elif int(label_info[k,2]) == 2:
						vline.set_xdata(10)
					elif int(label_info[k,2]) == 4:
						vline.set_xdata(6)
				else:
					ax1.set_title('')
					vline.set_xdata(-1)
				if tmstamp_sliced[i,0] >= label_info[k,1]:
					if k+1 < label_info.shape[0]:
						k += 1
					else:
						pass

				#setting x axis
				ax.set_xlim(tmst_data[i,0], tmst_data[i,-1] +0.008) #+2)

				#setting y axis
				max_y = max(eeg_data[i,counter,:])
				min_y = min(eeg_data[i,counter,:])				
			else:
				max_y = np.amax(fft_data[i,:,use_frq])
				min_y = np.amin(fft_data[i,:,use_frq])
				# print(min_y, max_y)
				# min_y, max_y = -0.05, 10000000
				pass
				
			diff_y = abs(max_y - min_y)*0.1
			max_new = max_y + diff_y
			min_new = min_y - diff_y
			ax.set_ylim(min_new, max_new)
			counter += 1

		# update the data of all line objects
		for ch in range(eeg_data.shape[1]):
			line[ch].set_data(tmst_data[i, :], eeg_data[i,ch,:])
			# ##Do not plot first and last channels
			smoothFFT = np.mean([fft_data[i-2,ch,:], fft_data[i-1,ch,:], fft_data[i,ch,:]], axis = 0)
			if ch < 1 or ch > 6:
				line[ch+8].set_data(xfft_2, np.zeros(xfft_2.shape))
			else:
				line[ch+8].set_data(xfft_2, smoothFFT)

			# ##Do not plot last two channels
			# if ch < 6:
			# 	line[ch+8].set_data(xfft_2, fft_data[i,ch,:])
			# else:
			# 	line[ch+8].set_data(xfft_2, np.zeros(xfft_2.shape))
		
		fig_frame = os.path.join(new_folder, 'Frame_'+'{:0>5}'.format(i))
		plt.savefig(fig_frame)
		# plt.show()

		#Progress bar
		# compare = "FTS: ", tmstamp_sliced[i,-1], 'vs. ILI: ', label_info[k,0], '|| ITS: ', tmstamp_sliced[i,0], 'vs. FLI: ', label_info[k,1], '\n'
		msg = "Working on graph |" + '\u2588' * math.floor( (((i-init)/(FRAMES-init)*100) % 100)/2 ) + '\u2591' * math.ceil(50-( (((i-init)/(FRAMES-init)*100) % 100)/2 )) + '| f' + str(i) + ' ~ '+'{:.2f}%'.format((i-init)/(FRAMES-init)*100)
		print(msg, end="\r")


	print('Saving Animation...')
	gif_save = dir_.replace('.csv', '.gif')
	print('\n')

	winsound.Beep(261, 150)
	winsound.Beep(329, 150)
	winsound.Beep(415, 150)






















	##::Generate gif with FuncAnimation
	# from datetime import datetime
	# from datetime import timedelta
	# FRAMES = 1000 #save_count=ch_data_sliced.shape[0])
	# def run(i=int):
	# 	# axis limits checking. Same as before, just for both axes
	# 	counter = 0
	# 	for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
	# 		if counter <= 7:
	# 			#setting x axis
	# 			ax.set_xlim(tmstamp_sliced[i,0], tmstamp_sliced[i,-1]+2)

	# 			#setting y axis
	# 			max_y = max(norm_data[i,counter,:])
	# 			min_y = min(norm_data[i,counter,:])
	# 			# ax_min_y, ax_max_y = ax.get_ylim()

	# 			if max_y > 0:
	# 				max_new = max_y * 1.1
	# 			else:
	# 				max_new = max_y * 0.9

	# 			if max_y > 0:
	# 				min_new = min_y * 0.9
	# 			else:
	# 				min_new = min_y * 1.1

	# 			ax.set_ylim(min_new, max_new)
				
	# 		# else:
	# 		# 	ax.set_ylim(-5, np.amax(yfft_n[i,:,:]))
	# 		counter += 1
	# 		ax.figure.canvas.draw()	
	# 	# update the data of both line objects
	# 	for ch in range(norm_data.shape[1]):
	# 		line[ch].set_data(tmstamp_sliced[i,:], norm_data[i,ch,:])
	# 		line[ch+8].set_data(xfft, yfft_n[i,ch,:]) #Use multiple lines for the FFT
	# 	#Progress bar
	# 	msg = "Working on graph |" + '\u2588' * math.floor( ((i/FRAMES*100) % 100)/2 ) + '\u2591' * math.ceil(50-( ((i/FRAMES*100) % 100)/2 )) + '|'
	# 	print(msg, end="\r")
	# 	return line

	# @profile
	# def save_animation(gif_save, ani):
	# 	ani.save(gif_save)
	# 	print('')


	# print('Generating Animation...')
	# ani = ani.FuncAnimation(fig, run, blit=True, interval=4, repeat=False, save_count=FRAMES)
	# print('Saving Animation...')
	# gif_save = dir_.replace('.csv', '.gif')
	# save_animation(gif_save, ani)
	# print('Preparing to display...')
	# plt.show()
	# print('\n')

	sys.exit()