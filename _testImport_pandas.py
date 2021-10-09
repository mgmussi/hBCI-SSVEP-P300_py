import pandas as pd
pd.options.display.float_format = '{:20.6f}'.format
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

SPS = 250
NYQ = int(SPS*0.5)

# una_pred = np.zeros((3,1))
# lda_pred = np.zeros((3,1))
# cca_pred = np.zeros((3,1))
# tot_pred = np.zeros((3,1))
# print(una_pred, lda_pred, cca_pred, sep='\n', end='\n--\n')

def b2d(b_vect):
	str_format = ''
	for b in b_vect:
		str_format = str_format + str(int(b))
	# str_format = str(int(b_vect[0])) + str(int(b_vect[1])) + str(int(b_vect[2]))
	return int(str_format,2)

def CCA2d_single(Y):
	print('---___---___---')
	dec_y = []
	oarg = []
	marg = []
	if isinstance(Y, list):
		Y = np.array(Y)
	Y = abs(Y)
	print(Y)
	marg = np.argmax(Y)
	oarg = [x for x in range(len(Y)) if x != marg]
	Y[marg] = 1
	Y[oarg] = 0
	print(Y)
	dec_y.append(b2d(Y))
	return(dec_y)

if __name__ == '__main__':
	##::IMPORT LIBRARIES
	import numpy as np
	import mne
	from mne.preprocessing import ICA, create_ecg_epochs
	from mne.datasets import sample
	import pandas as pd
	pd.options.display.float_format = '{:20.6f}'.format
	import os
	import csv
	import numpy as np
	import matplotlib.pyplot as plt
	from tkinter import filedialog

	##::DEFINE CONSTANTS
	SPS = 250 #samples per second
	NYQ = int(SPS*0.5)

	##::EXTRACT DATA FROM .CSV
	ch_data_125 = []
	ts_data = []
	filename = filedialog.askopenfilename(initialdir = "C:/Users/atech/Documents/GitHub/SSVEP_EyeGaze-py/Participants",
			title = "Select Trial", filetypes = (("Comma Separated Values", "*.csv*"), ("all files","*.*")))
	df = pd.read_csv(filename)
	## Get df indexes:
	idx1 = df.index[df['P300_ts'] == 0]
	idx2 = df.index[df['P300_ts'] == 2]
	counter = 0
	for i, j in zip(idx1, idx2):
		## Get the last 125 samples before ending timestamp
		ch_data_125.append([
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH1']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH2']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH3']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH4']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH5']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH6']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH7']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH8']].values]])
	ts_data.append([t[0] for t in list(df.loc[j-NYQ+1:j, ['Timestamps']].values)])
	ch_data_125 = np.array(ch_data_125)
	ts_data = np.array(ts_data)

	##::RESHAPE DATA ARRAY
	ch_data = ch_data_125.reshape([-1, 8], order='C')
	ch_data = ch_data.transpose()
	print(ch_data.shape)

	##::NAME CHANNELS
	ch_names = ['CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8'] 

	# Create the info structure needed by MNE
	info = mne.create_info(ch_names, SPS)

	# Finally, create the Raw object
	raw = mne.io.RawArray(ch_data, info)
	raw.pick_types(meg=False, eeg=True, stim=False).load_data()

	##::MANUALLY CREATING EVENTS ARRAY (based on output specified in https://mne.tools/stable/generated/mne.find_events.html#mne.find_events)
	first_col = [i for i in range(0, 23624, 125)]
	third_col = [0 for _ in range(0, 23624, 125)]
	second_col = [-1 for _ in range(0, 23624, 125)]
	events = []
	events.append(first_col)
	events.append(second_col)
	events.append(third_col)
	events = np.array(events)
	events = events.transpose()

	##::CREATING EPOCHS (here is where I should fix the picks)
	epochs = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=0.5, baseline=(0, 0), picks = ['CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6']) #np.array([0,1,2,3,4,5]))

	##::PLOT RAW OBJECT
	raw.plot()

	##::FIT ICA
	ica = ICA(n_components=0.95, method='fastica').fit(epochs)
	ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5)
	ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, threshold='auto')

	##::PLOT ICA
	ica.plot_components(ecg_inds)


	# from sklearn.cross_decomposition import CCA
	# CCA_model = CCA(n_components = 3, max_iter=20000)

	# input_arr = [[[k*-1+j*-i*-1 for k in range(125)] for j in range(2)] for i in range(189)]
	# input_arr = np.array(input_arr)
	# print("INPUT SHAPE:", input_arr.shape)
	# input_lbl = [[(-(-1+(-1)**(1+k+j)))/2 for k in range(3)] for j in range(189)]
	# input_lbl = np.array(input_lbl)
	# print("LABEL SHAPE:", input_lbl.shape)

	# model = CCA_model.fit(input_arr, input_lbl)