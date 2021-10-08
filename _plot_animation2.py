'''
UPGRADES ON THIS VERSION:
- Trying to make animation faster
- Using all EEG graphs in one graph
- Potentially, reducing the frame rate (skipping every other frame?)
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
from scipy.signal import butter, lfilter, firwin
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

#1
SPS = 250
NYQ = int(SPS*0.5)
#
SLICE = NYQ
#
COMP_NUM = 2
LDA_model = LDA(solver = 'svd')
CCA_model = CCA(n_components = COMP_NUM, max_iter=20000)

#3
low = 5 / NYQ
high = 30 / NYQ
##BUTTERWORTH
# b, a = butter(4, [low, high], btype='band') # b, a = butter(order, [low, high], btype='band')
##FIR
N = int((SPS/60)*(16/22)) #(fs/transition band width)*(attenuation (dB)/22)
print('>>FIR Order: ', N)
b = firwin(N, [low, high], pass_zero = 'bandpass', window = "hamming")
def filtering(samples):
	# if len(samples[0]) > 1:
	if isinstance(samples[0], float):
	##BUTTERWORTH
		# filtsamples = np.array(lfilter(b, a, samples, axis = 0))
	##FIR
		filtsamples = np.array(lfilter(b, [1.0], samples, axis = 0))
	else:
	##BUTTERWORTH
		# filtsamples = np.array(lfilter(b, a, samples, axis = 1))
	##FIR
		filtsamples = np.array(lfilter(b, [1.0], samples, axis = 1))
	return filtsamples

def get_data_files():
	ch_data_125 = []
	ch_data_all = []
	tmstamp_all = []
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

	counter = 0

	ch_data_all = [[x[0] for x in df.loc[:, ['CH1']].values], #[chs x samples]
			[x[0] for x in df.loc[:, ['CH2']].values],
			[x[0] for x in df.loc[:, ['CH3']].values],
			[x[0] for x in df.loc[:, ['CH4']].values],
			[x[0] for x in df.loc[:, ['CH5']].values],
			[x[0] for x in df.loc[:, ['CH6']].values],
			[x[0] for x in df.loc[:, ['CH7']].values],
			[x[0] for x in df.loc[:, ['CH8']].values]]
	
	# tmstamp_all = [x[0] for x in df.loc[:, ['Timestamps']].values] #[samples]
	tmstamp_all = [x[0] for x in df.loc[:, ['Samples']].values] #[samples]
	

	# for i, j in zip(idx1, idx2):
	# 	## Get the last 125 samples
	# 	ch_data_125.append([
	# 		[x[0] for x in df.loc[j-SLICE+1:j, ['CH1']].values],
	# 		[x[0] for x in df.loc[j-SLICE+1:j, ['CH2']].values],
	# 		[x[0] for x in df.loc[j-SLICE+1:j, ['CH3']].values],
	# 		[x[0] for x in df.loc[j-SLICE+1:j, ['CH4']].values],
	# 		[x[0] for x in df.loc[j-SLICE+1:j, ['CH5']].values],
	# 		[x[0] for x in df.loc[j-SLICE+1:j, ['CH6']].values],
	# 		[x[0] for x in df.loc[j-SLICE+1:j, ['CH7']].values],
	# 		[x[0] for x in df.loc[j-SLICE+1:j, ['CH8']].values]])

	# 	## Get all samples for each frq. change
	# 	if counter % 21 == 0:
	# 		curr_id = i
	# 		ssvep_all = df.loc[i, ['SSVEP_labels']].values
	# 		ssvepll_data_all.append(ssvep_all[0])
	# 	counter += 1

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

	return ch_data_all, tmstamp_all, filename


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

	#G.TEC
	chs_p300 = [2,3,4,5]
	chs_ssvep = [6,7,8]

	chs_all = list(set(chs_p300 + chs_ssvep))
	_chs_p300 = list(range(len(chs_p300)))
	_chs_ssvep = list(range(_chs_p300[-1]+1, _chs_p300[-1]+1+len(chs_ssvep)))
	_chs_all = [x - 1 for x in chs_all]
	cca_lbls = []
	cca_lbls_all = []

	##::Receives ALL samples from .csv
	ch_data_all, tmstamp_all, dir_ = get_data_files()

	##::Check for pre-saved file
	dir__ = dir_.replace('.csv', '.pkl')
	##::Setting folder to save images
	new_folder = dir_[0:-4]
	if not os.path.exists(new_folder):
		os.makedirs(new_folder)

	if os.path.isfile(dir__):
		with open(dir__, 'rb') as input:
			ch_data_sliced, tmstamp_sliced, xfft, yfft, yfft_f, yfft_n, flt_data, norm_data = pickle.load(input)
	else:
		##::Create data slices
		ch_data_sliced = []
		tmstamp_sliced = []
		for idx in range(len(ch_data_all[0])-125):
			ch_data_sliced.append([ch_data_all[k][idx:idx+125] for k in range(len(ch_data_all))])
			tmstamp_sliced.append(tmstamp_all[idx:idx+125])

		##::Converting to array
		ch_data_all = np.array(ch_data_all)
		tmstamp_all = np.array(tmstamp_all)
		ch_data_sliced = np.array(ch_data_sliced)
		tmstamp_sliced = np.array(tmstamp_sliced)
		print('ch_data_sliced size: ', ch_data_sliced.shape)

		##::Filtering and creating fft for each section
		xfft = fftfreq(SLICE, 1/SPS)
		yfft = np.empty(ch_data_sliced.shape, float)
		yfft_f = np.empty(ch_data_sliced.shape, float)
		yfft_n = np.empty(ch_data_sliced.shape, float)
		flt_data = np.empty(ch_data_sliced.shape, float)
		norm_data = np.empty(ch_data_sliced.shape, float)
		for x in range(ch_data_sliced.shape[0]):

			if x % 125 == 0:
				msg = "Transforming slices |" + '\u2588' * math.floor( ((x/ch_data_sliced.shape[0]*100) % 100)/2 ) + '\u2591' * math.ceil(50-( ((x/ch_data_sliced.shape[0]*100) % 100)/2 )) + '|'
				# msg = "Transforming slice " + str(x) + '/' + str(ch_data_sliced.shape[0])
				print(msg, end="\r")

			yfft[x,:,:] = np.abs(fft(ch_data_sliced[x, :, :], axis = 1)) **2 #raw data FFT

			##:: 1 - FILTERING
			flt_data[x,:,:] = filtering(ch_data_sliced[x, :, :]) #shape << [chs, samples]

			yfft_f[x,:,:] = np.abs(fft(flt_data[x,:,:], axis = 1)) **2 #filtered data FFT

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
			norm_data[x,:,:] = scaler.fit_transform(flt_data[x,:,:])
			# norm_data[x,:,:] = np.squeeze(mne_scaler.fit_transform(flt_data[x,:,:]))
			yfft_n[x,:,:] = np.abs(fft(norm_data[x,:,:], axis = 1)) **2 #normalized data FFT
		print('')
		##::Converting to array
		norm_data = np.array(norm_data)
		flt_data = np.array(flt_data)

		##::Save
		with open(dir__, 'wb') as output:
			pickle.dump([ch_data_sliced, tmstamp_sliced, xfft, yfft, yfft_f, yfft_n, flt_data, norm_data], output, pickle.HIGHEST_PROTOCOL)



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---##::Plotting animation::##--------------------------------------------------------------------------------------------------------------------------------------------------------#
	# Figure with subplots
	fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,9),gridspec_kw={'height_ratios': [1,2]}, constrained_layout=True)

	# intialize two line objects (one in each axes)
	colors = ['#FF095F', '#FF3C15', '#FF552B', '#FF7A41', '#FFAE23', '#81BB3E', '#7C8F2F', '#C50000']
	line1, = ax1.plot([], [], lw=2, color = colors[0])
	line2, = ax1.plot([], [], lw=2, color = colors[1])
	line3, = ax1.plot([], [], lw=2, color = colors[2])
	line4, = ax1.plot([], [], lw=2, color = colors[3])
	line5, = ax1.plot([], [], lw=2, color = colors[4])
	line6, = ax1.plot([], [], lw=2, color = colors[5])
	line7, = ax1.plot([], [], lw=2, color = colors[6])
	line8, = ax1.plot([], [], lw=2, color = colors[7])

	line9, = ax2.plot([], [], lw=1, color = colors[0])
	line10, = ax2.plot([], [], lw=1, color = colors[1])
	line11, = ax2.plot([], [], lw=1, color = colors[2])
	line12, = ax2.plot([], [], lw=1, color = colors[3])
	line13, = ax2.plot([], [], lw=1, color = colors[4])
	line14, = ax2.plot([], [], lw=1, color = colors[5])
	line15, = ax2.plot([], [], lw=1, color = colors[6])
	line16, = ax2.plot([], [], lw=1, color = colors[7])

	line = [line1, line2, line3, line4, line5, line6, line7, line8,
			line9, line10, line11, line12, line13, line14, line15, line16]


	# axes initalization
	ax1.set_ylabel('Channels EEG')
	ax1.set_ylim(-100, 100)
	ax1.set_xlim(tmstamp_sliced[0,0], tmstamp_sliced[0,-1])
	# Turn off tick labels
	ax1.set_yticklabels([])
	ax1.set_xlabel('Samples')
	ax1.grid()

	ax2.set_ylabel('FFT')
	ax2.set_xlim(3, 65)
	ax2.set_ylim(-0.05, 1)
	ax2.grid()
	
	##::Quick access variables
	eeg_data = norm_data
	fft_data = yfft_n

	##::Generate frames with plot
	FRAMES = ch_data_sliced.shape[0]
	for i in range(3000, FRAMES): #<<skip frames can be included here (0, FRAMES, 2)
		#setting x axis
		ax1.set_xlim(tmstamp_sliced[i,0], tmstamp_sliced[i,-1]+2)

		#setting y axis
		max_y = np.amax(eeg_data[i,:,:])
		min_y = np.amin(eeg_data[i,:,:])
		diff_y = abs(max_y - min_y)*0.1
		max_new = max_y + diff_y
		min_new = min_y - diff_y
		ax1.set_ylim(min_new, max_new)
				
		# update the data of all line objects
		for ch in range(eeg_data.shape[1]):
			line[ch].set_data(tmstamp_sliced[i,:], eeg_data[i,ch,:])
			line[ch+8].set_data(xfft, fft_data[i,ch,:])
		
		fig_frame = os.path.join(new_folder, 'Frame_'+str(i))
		plt.savefig(fig_frame)
		# plt.show()

		#Progress bar
		msg = "Working on graph |" + '\u2588' * math.floor( ((i/FRAMES*100) % 100)/2 ) + '\u2591' * math.ceil(50-( ((i/FRAMES*100) % 100)/2 )) + '|'
		print(msg, end="\r")


	print('Saving Animation...')
	gif_save = dir_.replace('.csv', '.gif')
	print('\n')






















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