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
#
SLICE = NYQ #int(SPS*0.25)
#
COMP_NUM = 2
LDA_model = LDA(solver = 'svd')
CCA_model = CCA(n_components = COMP_NUM, max_iter=20000)

#3
low = 5 / NYQ
high = 30 / NYQ

#Notch
b1, a1 = iirnotch(60, 1, SPS)

##::FIR------------------------------------
# #Kaiser window method
win = 75 #padding
width = 2/NYQ #transition width
ripple_db = 11 #attenuation
N, beta = kaiserord(ripple_db, width)
b = firwin(N, [low, high], pass_zero = 'bandpass',  window=('kaiser', beta))
delay = int(0.5 * (N-1)) # / SPS
print('>>FIR Order: ', N)

print('>>Delay    : ', delay)
print('>>SPS      : ', SPS)

# # Frequency response
# freq, h = freqz(b, fs=SPS)
# # freq, h = freqz(b1, a1, fs=SPS)
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
# sys.exit()

def get_data_files(chs):
	ch_data_all = []
	ts_data = []

	##::Dealing with file location
	filename = filedialog.askopenfilename(initialdir = "C:/Users/atech/Documents/GitHub/SSVEP_EyeGaze-py/Participants",
		title = "Select Trial", filetypes = (("Tab delimited files", "*.txt*"), ("all files","*.*")))
	dir_name = filename

	##::Getting data out of file
	print(filename)
	df = pd.read_csv(filename, sep='\t', lineterminator='\r', header=None)

	##::Extracting data from file
	chs_idx = [ch-1 for ch in chs]
	ch_data_all = [[x for x in df.iloc[:, chs[0]].values], #[chs x samples]
			[x for x in df.iloc[:, chs[1]].values],
			[x for x in df.iloc[:, chs[2]].values],
			[x for x in df.iloc[:, chs[3]].values],
			[x for x in df.iloc[:, chs[4]].values],
			[x for x in df.iloc[:, chs[5]].values],
			[x for x in df.iloc[:, chs[6]].values],
			[x for x in df.iloc[:, chs[7]].values]]
	samples_all = df.index.values

	return ch_data_all, samples_all, filename


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
	#EGI
	chs_p300 = [28,34,42,4]
	chs_ssvep = [35,39,37,36]
	chs_names = ['P3', 'Pz', 'P4', 'FCz', 'O1', 'O2', 'OZ', 'POz']

	chs_all = chs_p300 + chs_ssvep
	_chs_p300 = list(range(len(chs_p300)))
	_chs_ssvep = list(range(_chs_p300[-1]+1, _chs_p300[-1]+1+len(chs_ssvep)))
	_chs_all = _chs_p300 + _chs_ssvep
	cca_lbls = []
	cca_lbls_all = []

	##::Receives ALL samples from .csv
	ch_data_all, samples_all, dir_ = get_data_files(chs_all)

	print('ch_data_all size: ', len(ch_data_all), len(ch_data_all[0]))
	print('samples_all size: ', len(samples_all))

	##::Check for pre-saved file
	dir__ = dir_.replace('.txt', '.pkl')
	##::Setting folder to save images
	new_folder = dir_[0:-4]
	if not os.path.exists(new_folder):
		os.makedirs(new_folder)

	if os.path.isfile(dir__):
		print('Opening file...')
		with open(dir__, 'rb') as input:
			ch_data_sliced, samples_sliced, xfft, yfft, yfft_f, yfft_n, flt_data, norm_data = pickle.load(input)
	else:
		##::Create data slices
		ch_data_sliced = []
		samples_sliced = []
		# for idx in range(len(ch_data_all[0])-NYQ):
		for idx in range(0, len(ch_data_all[0])-NYQ, 8):		
			ch_data_sliced.append([ch_data_all[k][idx:idx+NYQ] for k in range(len(ch_data_all))]) #for each channel, chop the data in NYQ-sample-segments
			samples_sliced.append(samples_all[idx:idx+NYQ])
			msg = 'Extracting data from file...{:5.2f}%'.format((idx/(len(ch_data_all[0])-NYQ))*100)
			print(msg, end="\r")
		print('')

		print('Converting to arrays...')
		##::Converting to array
		ch_data_all = np.array(ch_data_all)
		ch_data_sliced = np.array(ch_data_sliced)
		samples_all = np.array(samples_all)
		samples_sliced = np.array(samples_sliced)

		#Array size & memory checker
		print('\t> ch_data_all size: ', ch_data_all.shape, '[', len(pickle.dumps(ch_data_all, -1)), 'b]')
		print('\t> ch_data_sliced size: ', ch_data_sliced.shape, '[', len(pickle.dumps(ch_data_sliced, -1)), 'b]')
		print('\t> samples_all size: ', samples_all.shape, '[', len(pickle.dumps(samples_all, -1)), 'b]')
		print('\t> samples_sliced size: ', samples_sliced.shape, '[', len(pickle.dumps(samples_sliced, -1)), 'b]')

		print('Pre-locating arrays...')
		##::Filtering and creating fft for each section
		xfft = fftfreq(SLICE, 1/SPS)
		xfft_2 = fftfreq(SLICE-delay, 1/SPS)
		yfft = np.empty(ch_data_sliced.shape, float)
		#If using FIR, pre-location accounts for FIR induced delay; else, it will use delay = 0
		yfft_f = np.empty((ch_data_sliced.shape[0], ch_data_sliced.shape[1], ch_data_sliced.shape[2]-delay), float)
		yfft_n = np.empty((ch_data_sliced.shape[0], ch_data_sliced.shape[1], ch_data_sliced.shape[2]-delay), float)
		flt_data = np.empty((ch_data_sliced.shape[0], ch_data_sliced.shape[1], ch_data_sliced.shape[2]-delay), float)
		notch_data = np.empty((ch_data_sliced.shape[0], ch_data_sliced.shape[1], ch_data_sliced.shape[2]+2*win), float)
		norm_data = np.empty((ch_data_sliced.shape[0], ch_data_sliced.shape[1], ch_data_sliced.shape[2]-delay), float)
		

		print('Starting slices...\r')
		for x in range(ch_data_sliced.shape[0]):
		# for x in range(0, ch_data_sliced.shape[0], 8):
			if x % 128 == 0:
				msg = "Transforming slices |" + '\u2588' * math.floor( ((x/ch_data_sliced.shape[0]*100) % 100)/2 ) + '\u2591' * math.ceil(50-( ((x/ch_data_sliced.shape[0]*100) % 100)/2 )) + '| slice ' + str(x) + ' ~ '+'{:.2f}%'.format(x/ch_data_sliced.shape[0]*100)
				# msg = "Transforming slice " + str(x) + '/' + str(ch_data_sliced.shape[0])
				print(msg, end="\r")

			yfft[x,:,:] = np.abs(fft(ch_data_sliced[x, :, :], axis = 1))**2 #raw data FFT

			##:: 1 - FILTERING FIR
			#notch filter
			prefilt = np.hstack(( np.flip(ch_data_sliced[x, :, 0:win], axis = 1), ch_data_sliced[x, :, :], np.flip(ch_data_sliced[x, :, -win:], axis = 1) ))
			notch_data[x,:,:] = lfilter(b1, a1, prefilt, axis = 1)
			yfft_notch = np.abs(fft(notch_data[x, :, win:-win], axis = 1))**2
			s_notch = notch_data[x,:,win:-win]
			prefilt = np.hstack(( np.flip(s_notch[:, 0:win], axis = 1), s_notch[:, :], np.flip(s_notch[:, -win:], axis = 1) ))
			#FIR
			posfilt = lfilter(b, [1.0], prefilt, axis = 1)
			flt_data[x,:,:] = posfilt[:, win+delay:-win] #shape << [chs, samples]
			yfft_f[x,:,:] = np.abs(fft(flt_data[x,:,:], axis = 1))**2 #filtered data FFT

			##:: 3 - SCALE/NORMALIZE
			# norm_data[x,:,:] = scaler.fit_transform(flt_data[x,:,:])
			norm_data[x,:,:] = np.squeeze(mne_scaler.fit_transform(flt_data[x,:,:]))
			yfft_n[x,:,:] = np.abs(fft(norm_data[x,:,:], axis = 1))**2 #normalized data FFT
		print('')
		##::Converting to array
		norm_data = np.array(norm_data)
		flt_data = np.array(flt_data)

		##::Save
		print('Final alocation: ', len(pickle.dumps([ch_data_sliced, samples_sliced, xfft, yfft, yfft_f, yfft_n, flt_data, norm_data], -1)), 'b')
		with open(dir__, 'wb') as output:
			pickle.dump([ch_data_sliced, samples_sliced, xfft, yfft, yfft_f, yfft_n, flt_data, norm_data], output, pickle.HIGHEST_PROTOCOL)
			print('Data saved \u00B6')
		winsound.Beep(277, 50)
		winsound.Beep(391, 50)
		winsound.Beep(493, 50)

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
			ax.set_xlim(samples_sliced[0,0], samples_sliced[0,-1])
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
	

#--------------------------
	##::Quick access variables
	eeg_data =  norm_data  #ch_data_sliced, #, #flt_data
	tmst_data = samples_sliced[:, delay:] #tmstamp_sliced, #, #tmstamp_sliced[:, delay:]
	fft_data =  yfft_n #yfft, #, #, #yfft_f
	xfft_2 = fftfreq(SLICE-delay, 1/SPS)
	use_frq = [i for i in range(len(xfft_2)) if xfft_2[i]>=5 and xfft_2[i]<50]
	# print(use_frq)
	# use_frq = xfft_2>=3 and xfft_2<50
	init = 3000
#--------------------------	

	##::Generate frames with plot
	##SAVE TO FOLDER: 'new_folder'
	FRAMES = ch_data_sliced.shape[0]
	k = 0
	for i in range(init, FRAMES): #<<skip frames
		counter = 0
		for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:

			# EEG PLOTS
			if counter <= len(chs_names)-1:

				#setting x axis
				ax.set_xlim(tmst_data[i,0], tmst_data[i,-1] +0.008) #+2)

				#setting y axis
				max_y = max(eeg_data[i,counter,:])
				min_y = min(eeg_data[i,counter,:])

			# FFT PLOTS
			else:
				max_y = np.amax(fft_data[i,4:,use_frq])
				min_y = np.amin(fft_data[i,4:,use_frq])
				
			diff_y = abs(max_y - min_y)*0.1
			max_new = max_y + diff_y
			min_new = min_y - diff_y
			ax.set_ylim(min_new, max_new)
			counter += 1

		# update the data of all line objects
		for ch in range(eeg_data.shape[1]):
			# EEG PLOTS
			line[ch].set_data(tmst_data[i, :], eeg_data[i,ch,:])
			
			# FFT PLOTS
			if ch < 4: #do not plot FIRST THREE channels
				line[ch+8].set_data(xfft_2, np.zeros(xfft_2.shape))
			else:
				line[ch+8].set_data(xfft_2, fft_data[i,ch,:])
				
		
		fig_frame = os.path.join(new_folder, 'Frame_'+'{:0>5}'.format(i))
		plt.savefig(fig_frame)
		# plt.show()

		#Progress bar
		# compare = "FTS: ", tmstamp_sliced[i,-1], 'vs. ILI: ', label_info[k,0], '|| ITS: ', tmstamp_sliced[i,0], 'vs. FLI: ', label_info[k,1], '\n'
		msg = "Working on graph |" + '\u2588' * math.floor( (((i-init)/(FRAMES-init)*100) % 100)/2 ) + '\u2591' * math.ceil(50-( (((i-init)/(FRAMES-init)*100) % 100)/2 )) + '| f' + str(i) + ' ~ '+'{:.2f}%'.format((i-init)/(FRAMES-init)*100)
		print(msg, end="\r")


	print('Saving Animation...')
	# gif_save = dir_.replace('.csv', '.gif')
	print('\n')

	winsound.Beep(261, 50)
	winsound.Beep(329, 50)
	winsound.Beep(415, 50)



















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