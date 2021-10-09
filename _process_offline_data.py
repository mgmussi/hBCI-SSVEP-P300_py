import sys
from tkinter import filedialog
import pandas as pd
pd.options.display.float_format = '{:20.6f}'.format
import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
import matplotlib.pyplot as plt

# import scipy
from scipy.signal import butter, lfilter
from scipy import interpolate
from scipy.fft import fft, fftfreq

from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

from datetime import datetime
import time

SPS = 250
NYQ = int(SPS*0.5)
COMP_NUM = 2
LDA_model = LDA(solver = 'svd')
CCA_model = CCA(n_components = COMP_NUM, max_iter=20000)

######
def b2d(b_vect):
	str_format = ''
	for b in b_vect:
		str_format = str_format + str(int(b))
	return int(str_format,2)

def d2b(decimal):
	b_array = []
	str_format = "{0:03b}".format(int(decimal))#, int(decimal))
	# print(str_format)
	for s in str_format:
		b_array.append(int(s))
	# print(b_array)
	return b_array
######


def normalization(matrix, x_max = 1, x_min = -1):
	try:
		avgsamples = scaler.fit_transform(matrix)
	except RuntimeWarning:
			print('Caution::may have RAILED channel(s)')
			print('|||||Caution::may have RAILED channel(s)')
	return avgsamples

def filtering(samples):
	low = 5 / NYQ
	high = 30 / NYQ

	##BUTTERWORTH
	b, a = butter(4, [low, high], btype='band') # b, a = butter(order, [low, high], btype='band')

	# if len(samples[0]) > 1:
	if isinstance(samples[0], float):
		filtsamples = np.array(lfilter(b, a, samples, axis = 0))
	else:
		filtsamples = np.array(lfilter(b, a, samples, axis = 1))

	# print("FLT_SAMPS:",filtsamples.shape)
	return filtsamples

def get_data():
	ch_data_125 = []
	ch_data_250 = []
	ch_data_375 = []
	ch_data_all = []
	ts_data = []
	p300ll_data = []
	ssvepll_data = []
	ssvepll_data_all = []
	curr_id = 0

	filename = filedialog.askopenfilename(initialdir = "C:/Users/atech/Documents/GitHub/SSVEP_EyeGaze-py/Participants",
		title = "Select Trial", filetypes = (("Comma Separated Values", "*.csv*"), ("all files","*.*")))

	df = pd.read_csv(filename)

	## Get df rows:
	# idx1 = df.loc[df['P300_ts'] == 0]
	# idx2 = df.loc[df['P300_ts'] == 2]

	## Get df indexes:
	idx1 = df.index[df['P300_ts'] == 0]
	idx2 = df.index[df['P300_ts'] == 2]

	counter = 0
	for i, j in zip(idx1, idx2):

		## Get the last 125 samples
		ch_data_125.append([
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH1']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH2']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH3']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH4']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH5']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH6']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH7']].values],
			[x[0] for x in df.loc[j-NYQ+1:j, ['CH8']].values]])

		## Get the last 250 samples
		ch_data_250.append([
			[x[0] for x in df.loc[j-SPS+1:j, ['CH1']].values], # 
			[x[0] for x in df.loc[j-SPS+1:j, ['CH2']].values], # 
			[x[0] for x in df.loc[j-SPS+1:j, ['CH3']].values], # 
			[x[0] for x in df.loc[j-SPS+1:j, ['CH4']].values], # 
			[x[0] for x in df.loc[j-SPS+1:j, ['CH5']].values], # 
			[x[0] for x in df.loc[j-SPS+1:j, ['CH6']].values], # 
			[x[0] for x in df.loc[j-SPS+1:j, ['CH7']].values], # 
			[x[0] for x in df.loc[j-SPS+1:j, ['CH8']].values]]) # 
		
		## Get the last 375 samples
		ch_data_375.append([
			[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH1']].values], # 
			[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH2']].values], # 
			[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH3']].values], # 
			[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH4']].values], # 
			[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH5']].values], # 
			[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH6']].values], # 
			[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH7']].values], # 
			[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH8']].values]]) # 

		## Get all samples for each frq. change
		if counter % 21 == 0:
			# print(f'Cycle {counter}: current ID = {curr_id}')
			curr_id = i
			ssvep_all = df.loc[i, ['SSVEP_labels']].values
			ssvepll_data_all.append(ssvep_all[0])
		if counter % 20 == 0 and counter != 0:
			# print(f'\tCycle {counter}: sample len = {i+125-curr_id}')
			ch_data_all.append([
				[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH1']].values],
				[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH2']].values],
				[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH3']].values],
				[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH4']].values],
				[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH5']].values],
				[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH6']].values],
				[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH7']].values],
				[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH8']].values]])
		counter += 1

		ts_data.append([t[0] for t in list(df.loc[j-NYQ+1:j, ['Timestamps']].values)])
		p300 = df.loc[i, ['P300_labels']].values
		p300ll_data.append(p300[0])
		ssvep = df.loc[i, ['SSVEP_labels']].values
		ssvepll_data.append(ssvep[0])
		# print(f'SSVEP LBL DATA SIZE #{counter}: {len(ssvepll_data)}')

	# ##::PRINT ARRAY SIZES::
	# ##CHANNELS << [n. slices]x[chanels]x[samples]
	# print('CH_DATA:\t['+str(len(ch_data)), len(ch_data[0]), len(ch_data[0][0]), sep = ']x[', end = ']\n')
	# ##TIMESTAMPS << [n. slices]x[timestamps]
	# print('TS_DATA:\t['+str(len(ts_data)), len(ts_data[0]), sep = ']x[', end = ']\n')
	# ##P300 LABELS << [n. slices]
	# print('P300_LBL:\t['+str(len(p300ll_data)), sep = ']x[', end = ']\n')
	# ##SSVEP LABELS << [n. slices]
	# print('SSVEP_LBL:\t['+str(len(ssvepll_data)), sep = ']x[', end = ']\n')

	ch_data_125 = np.array(ch_data_125)
	ch_data_250 = np.array(ch_data_250)
	ch_data_375 = np.array(ch_data_375)
	ch_data_all = np.array(ch_data_all)
	ts_data = np.array(ts_data)
	p300ll_data = np.array(p300ll_data)
	ssvepll_data = np.array(ssvepll_data)
	ssvepll_data_all = np.array(ssvepll_data_all)
	print(f'CHAN DATA SIZE FINAL: {ch_data_250.shape}')

	return ch_data_125, ch_data_250, ch_data_375, ch_data_all, ts_data, p300ll_data, ssvepll_data, ssvepll_data_all, filename

def average_across_trials(data):
	count_ch = 0
	count_seg = 0
	if data.ndim == 2:
		print('2D INPUT')
		for ch_data in data:
			avg = np.mean(ch_data)
			print(f'Data average in ch{count_ch} = {avg}')
			count_ch  += 1
	elif data.ndim == 3:
		print('3D INPUT')
		for data_seg in data:
			count_ch = 0
			for ch_data in data_seg:
				avg = np.mean(ch_data)
				print(f'Trial{count_seg}: Data average in ch{count_ch} = {avg}')
				count_ch += 1
			print('\n\n')
			count_seg += 1


def get_data_files():
	ch_data_125 = []
	ch_data_250 = []
	ch_data_375 = []
	ch_data_all = []
	ts_data = []
	p300ll_data = []
	ssvepll_data = []
	ssvepll_data_all = []
	curr_id = 0

	filename = filedialog.askopenfilenames(initialdir = "C:/Users/atech/Documents/GitHub/SSVEP_EyeGaze-py/Participants",
		title = "Select Trial", filetypes = (("Comma Separated Values", "*.csv*"), ("all files","*.*")))
	if len(filename) > 1:
		ini_str = filename[0].rindex('/') + 1
		end_str = filename[0].index('_', ini_str)
		part_name = filename[0][ini_str:end_str]
		dir_name = part_name+"__all_saveddata"
	else:
		dir_name = filename[0]
	for file in filename:
		print(file)

# C:/Users/atech/Documents/GitHub/SSVEP_EyeGaze-py/Participants/_All/Erfan_00_02_saveddata.csv

		df = pd.read_csv(file)

		## Get df rows:
		# idx1 = df.loc[df['P300_ts'] == 0]
		# idx2 = df.loc[df['P300_ts'] == 2]

		## Get df indexes:
		idx1 = df.index[df['P300_ts'] == 0]
		idx2 = df.index[df['P300_ts'] == 2]

		counter = 0
		for i, j in zip(idx1, idx2):

			## Get the last 125 samples
			ch_data_125.append([
				[x[0] for x in df.loc[j-NYQ+1:j, ['CH1']].values],
				[x[0] for x in df.loc[j-NYQ+1:j, ['CH2']].values],
				[x[0] for x in df.loc[j-NYQ+1:j, ['CH3']].values],
				[x[0] for x in df.loc[j-NYQ+1:j, ['CH4']].values],
				[x[0] for x in df.loc[j-NYQ+1:j, ['CH5']].values],
				[x[0] for x in df.loc[j-NYQ+1:j, ['CH6']].values],
				[x[0] for x in df.loc[j-NYQ+1:j, ['CH7']].values],
				[x[0] for x in df.loc[j-NYQ+1:j, ['CH8']].values]])

			## Get the last 250 samples
			ch_data_250.append([
				[x[0] for x in df.loc[j-SPS+1:j, ['CH1']].values], # 
				[x[0] for x in df.loc[j-SPS+1:j, ['CH2']].values], # 
				[x[0] for x in df.loc[j-SPS+1:j, ['CH3']].values], # 
				[x[0] for x in df.loc[j-SPS+1:j, ['CH4']].values], # 
				[x[0] for x in df.loc[j-SPS+1:j, ['CH5']].values], # 
				[x[0] for x in df.loc[j-SPS+1:j, ['CH6']].values], # 
				[x[0] for x in df.loc[j-SPS+1:j, ['CH7']].values], # 
				[x[0] for x in df.loc[j-SPS+1:j, ['CH8']].values]]) # 
			
			## Get the last 375 samples
			ch_data_375.append([
				[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH1']].values], # 
				[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH2']].values], # 
				[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH3']].values], # 
				[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH4']].values], # 
				[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH5']].values], # 
				[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH6']].values], # 
				[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH7']].values], # 
				[x[0] for x in df.loc[j-NYQ-SPS+1:j, ['CH8']].values]]) # 

			## Get all samples for each frq. change
			if counter % 21 == 0:
				# print(f'Cycle {counter}: current ID = {curr_id}')
				curr_id = i
				ssvep_all = df.loc[i, ['SSVEP_labels']].values
				ssvepll_data_all.append(ssvep_all[0])
			if counter % 20 == 0 and counter != 0:
				# print(f'\tCycle {counter}: sample len = {i+125-curr_id}')
				ch_data_all.append([
					[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH1']].values],
					[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH2']].values],
					[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH3']].values],
					[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH4']].values],
					[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH5']].values],
					[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH6']].values],
					[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH7']].values],
					[x[0] for x in df.loc[curr_id:curr_id+2625+1, ['CH8']].values]])
			counter += 1

			ts_data.append([t[0] for t in list(df.loc[j-NYQ+1:j, ['Timestamps']].values)])
			p300 = df.loc[i, ['P300_labels']].values
			p300ll_data.append(p300[0])
			ssvep = df.loc[i, ['SSVEP_labels']].values
			ssvepll_data.append(ssvep[0])
			# print(f'SSVEP LBL DATA SIZE #{counter}: {len(ssvepll_data)}')

	# ##::PRINT ARRAY SIZES::
	# ##CHANNELS << [n. slices]x[chanels]x[samples]
	# print('CH_DATA:\t['+str(len(ch_data_125)), len(ch_data_125[0]), len(ch_data_125[0][0]), sep = ']x[', end = ']\n')
	# print('CH_DATA:\t['+str(len(ch_data_250)), len(ch_data_250[0]), len(ch_data_250[0][0]), sep = ']x[', end = ']\n')
	# print('CH_DATA:\t['+str(len(ch_data_375)), len(ch_data_375[0]), len(ch_data_375[0][0]), sep = ']x[', end = ']\n')
	# print('CH_DATA:\t['+str(len(ch_data_all)), len(ch_data_all[0]), len(ch_data_all[0][0]), sep = ']x[', end = ']\n')
	# ##TIMESTAMPS << [n. slices]x[timestamps]
	# print('TS_DATA:\t['+str(len(ts_data)), len(ts_data[0]), sep = ']x[', end = ']\n')
	# ##P300 LABELS << [n. slices]
	# print('P300_LBL:\t['+str(len(p300ll_data)), sep = ']x[', end = ']\n')
	# ##SSVEP LABELS << [n. slices]
	# print('SSVEP_LBL:\t['+str(len(ssvepll_data)), sep = ']x[', end = ']\n')
	# sys.exit()

	ch_data_125 = np.array(ch_data_125)
	ch_data_250 = np.array(ch_data_250)
	ch_data_375 = np.array(ch_data_375)
	ch_data_all = np.array(ch_data_all)
	ts_data = np.array(ts_data)
	p300ll_data = np.array(p300ll_data)
	ssvepll_data = np.array(ssvepll_data)
	ssvepll_data_all = np.array(ssvepll_data_all)
	print(f'CHAN DATA SIZE FINAL: {ch_data_250.shape}')

	return ch_data_125, ch_data_250, ch_data_375, ch_data_all, ts_data, p300ll_data, ssvepll_data, ssvepll_data_all, dir_name

def P300fun(data, timestamps):
	data_DwnSmp = []
	tmst_DwnSmp = []
	for datum, tmst in zip(data, timestamps):
		try:
			mean_data = np.mean(datum, 0)
		except RuntimeWarning:
			print('<<<Attempting mean with empty array',ss)
		##::Downsample == Features 63 sample?
		data_DwnSmp.append(mean_data[::6])
		tmst_DwnSmp.append(tmst[::6])

		##ADD FILTER TO SIGNAL?
	return data_DwnSmp, tmst_DwnSmp

def SSVEPfun(data):
	feat_set = []
	concat = []

	low1 = 5 / NYQ
	high1 = 7 / NYQ
	#
	low2 = 9 / NYQ
	high2 = 11 / NYQ
	#
	low3 = 14 / NYQ
	high3 = 16 / NYQ

	b1, a1 = butter(2, [low1, high1], btype='band')
	b2, a2 = butter(2, [low2, high2], btype='band')
	b3, a3 = butter(2, [low3, high3], btype='band')
	#
	B, A = butter(2, [low1, high3], btype='band')


	for datum in data:
		# ##::Average data together
		# try:
		# 	mean_data = np.mean(datum, 0)
		# 	mean_data = np.pad(mean_data, (20,), 'edge')
		# except RuntimeWarning:
		# 	print('<<<Attempting mean with empty array',ss)

		# ##::Concatenate data togehter
		# concat = []
		# for ch in datum:
		# 	ch_pad = np.pad(ch, (10,), 'edge')
		# 	f1 = np.array(lfilter(b1, a1, ch_pad, axis = -1))
		# 	f2 = np.array(lfilter(b2, a2, ch_pad, axis = -1))
		# 	f3 = np.array(lfilter(b3, a3, ch_pad, axis = -1))
		# 	feat = np.concatenate([f1,f2,f3], 0)
		# 	concat = np.concatenate((concat, feat))
		# # concat = np.concatenate((concat, np.pad(ch, (10,), 'edge')))

		# f1 = np.array(lfilter(b1, a1, mean_data, axis = -1))
		# f2 = np.array(lfilter(b2, a2, mean_data, axis = -1))
		# f3 = np.array(lfilter(b3, a3, mean_data, axis = -1))
		# #
		# F = np.array(lfilter(B, A, concat, axis = -1))

		# ## SUM METHOD
		# feat = np.sum([f1,f2,f3], 0)

		# ## CONCAT METHOD
		# # feat = np.concatenate((f1.T, f2.T, f3.T))

		# # print(feat.shape)
		# # print(F.shape)

		# ## PSD
		# # yf = fft(F)
		# # xf = fftfreq(len(concat), 1/SPS) #change first term to NYQ if 125 or SPS if 250 or 1.5*SPS if 375
		# # i = xf >= 0
		# # xf, yf = xf[i], np.abs(yf[i])

		####
		both_chan = datum.reshape(-1)
		f1 = np.array(lfilter(b1, a1, both_chan, axis = -1))
		f2 = np.array(lfilter(b2, a2, both_chan, axis = -1))
		f3 = np.array(lfilter(b3, a3, both_chan, axis = -1))
		# feat = np.sum([f1,f2,f3], 0)
		# feat = np.concatenate((f1.T, f2.T, f3.T))
		feat = np.array([f1, f2, f3]).reshape(-1)
		feat_set.append(feat)
	return feat_set


######################

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
			# #
			# for ll in y_train:
			# 	bin_lbl = str(int(ll[0])) + str(int(ll[1])) + str(int(ll[2]))
			# 	bin_y_train.append(int(bin_lbl,2))
			# y_train = bin_y_train
			# print(len(y_train))
			# print(X_train.shape)
			# #

			model = LDA_model.fit(X_train, y_train)

		# #
		# print("Starting labeling")
		# for ll in y_test:
		# 	bin_lbl = str(int(ll[0])) + str(int(ll[1])) + str(int(ll[2]))
		# 	bin_y_test.append(int(bin_lbl,2))
		# y_test = bin_y_test
		# print(len(y_test))
		# print("End labeling")
		# #

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
			# for ll in y_test:
			# 	# print(ll)
			# 	bin_y_test.append(int(ll))
			#
			bin_y_test = y_test
			#
		##Check
		print(new_y)		#binary predicted
		print(bin_y_test)	#vs. binary test lbls
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




#################################################
#################################################
#################################################
#################################################




if __name__ == '__main__':
	chs_p300 = [1,2,3,4]	#available channels [1 to 8] 		#PO8, PO7, POz, CPz		#P3, Pz, P4, Cz
	chs_ssvep = [5,6]		#available channels [1 to 8]		#O1, Oz, O2 			#O1, O2
	chs_all = list(set(chs_p300 + chs_ssvep))
	_chs_p300 = [x - 1 for x in chs_p300]
	_chs_ssvep = [x - 1 for x in chs_ssvep]
	_chs_all = [x - 1 for x in chs_all]
	# avg_data = []
	# avg_data = np.empty([])
	# flt_data = []
	# flt_data = np.empty([])
	cca_lbls = []
	cca_lbls_all = []

	## Receives ALL samples from .csv
	# eeg_data, eeg_data_2, eeg_data_3, eeg_data_all, timestamps, P300_lbls, SSVEP_lbls, SSVEP_lbls_all, dir_ = get_data() #all numpy arrays
	eeg_data, eeg_data_2, eeg_data_3, eeg_data_all, timestamps, P300_lbls, SSVEP_lbls, SSVEP_lbls_all, dir_ = get_data_files() #all numpy arrays

	# average_across_trials(eeg_data)
	# sys.exit()

	## Creating Empty arrays
	avg_data = np.empty((eeg_data.shape[0], len(_chs_all), eeg_data.shape[2]), float)
	flt_data = np.empty((eeg_data.shape[0], len(_chs_all), eeg_data.shape[2]), float)

	avg_data_2 = np.empty((eeg_data_2.shape[0], len(_chs_all), eeg_data_2.shape[2]), float)
	flt_data_2 = np.empty((eeg_data_2.shape[0], len(_chs_all), eeg_data_2.shape[2]), float)

	avg_data_3 = np.empty((eeg_data_3.shape[0], len(_chs_all), eeg_data_3.shape[2]), float)
	flt_data_3 = np.empty((eeg_data_3.shape[0], len(_chs_all), eeg_data_3.shape[2]), float)

	avg_data_all = np.empty((eeg_data_all.shape[0], len(_chs_all), eeg_data_all.shape[2]), float)
	flt_data_all = np.empty((eeg_data_all.shape[0], len(_chs_all), eeg_data_all.shape[2]), float)

	## Conditioning Data
	# c = 0
	# for slices in eeg_data:
	# 	samples = slices[_chs_all] #shape << [n_chs]x[samples]
	# 	avg = normalization(samples)
	# 	avg_data.append(avg)
	# 	flt_data.append(filtering(avg))
	# 	# print(len(avg), 'x', len(avg[0]))
	# 	plt.plot(timestamps[c].T, filtering(avg),'b')
	# 	plt.show()

	print('EEG_DATA.shape', eeg_data.shape)
	print('EEG_DATA_2.shape', eeg_data_2.shape)
	print('EEG_DATA_ALL.shape', eeg_data_all.shape)

	for x in range(eeg_data.shape[0]):
		samples = eeg_data[x, _chs_all, :] #shape << [1, n_chs, samples]
		avg = normalization(samples)
		avg_data[x, :, :] = avg
		flt_data[x,:,:] = filtering(avg)

		samples_2 = eeg_data_2[x, _chs_all, :] #shape << [1, n_chs, samples]
		avg_2 = normalization(samples_2)
		avg_data_2[x, :, :] = avg_2
		flt_data_2[x, :, :] = filtering(avg_2)

		samples_3 = eeg_data_3[x, _chs_all, :] #shape << [1, n_chs, samples]
		avg_3 = normalization(samples_3)
		avg_data_3[x, :, :] = avg_3
		flt_data_3[x, :, :] = filtering(avg_3)

	for x in range(eeg_data_all.shape[0]):
		samples = eeg_data_all[x, _chs_all, :] #shape << [1, n_chs, samples]
		avg = normalization(samples)
		avg_data_all[x, :, :] = avg
		flt_data_all[x,:,:] = filtering(avg)

	avg_data = np.array(avg_data)
	flt_data = np.array(flt_data)
	avg_data_2 = np.array(avg_data_2)
	flt_data_2 = np.array(flt_data_2)
	avg_data_3 = np.array(avg_data_3)
	flt_data_3 = np.array(flt_data_3)
	avg_data_all = np.array(avg_data_all)
	flt_data_all = np.array(flt_data_all)

	## Plotting
	# tst_continuum = np.reshape(timestamps, (-1, 1))
	# raw_continuum = np.reshape(eeg_data[:, _chs_all, :], (len(_chs_all), -1))
	# avg_continuum = np.reshape(avg_data, (len(_chs_all), -1))
	# flt_continuum = np.reshape(flt_data, (len(_chs_all), -1))
	# plt.plot(tst_continuum, raw_continuum.T,'k')
	# plt.figure()
	# plt.plot(tst_continuum, avg_continuum.T,'b')
	# plt.figure()
	# plt.plot(tst_continuum, flt_continuum.T,'r')
	# plt.show()
	# sys.exit()
	# plt.show()

	## Conditioning SSVEP Labels
	for S_lbl in SSVEP_lbls:
		cca_lbls.append(d2b(S_lbl))
	cca_lbls = np.array(cca_lbls)

	for S_lbl_all in SSVEP_lbls_all:
		cca_lbls_all.append(d2b(S_lbl_all))
	cca_lbls_all = np.array(cca_lbls_all)

	## Processing for P300
	feat_set_P300, tmst_set_P300 = P300fun(flt_data[:, _chs_p300, :], timestamps)

	for c in range(3):
		print(f"\n\n\nTEST #{c}:\n")
		## Processing for SSVEP
		if c == 0:
			feat_set_SSVEP = SSVEPfun(flt_data[:, _chs_ssvep, :])
		elif c == 1:
			feat_set_SSVEP = SSVEPfun(flt_data_2[:, _chs_ssvep, :])
		elif c == 2:
			feat_set_SSVEP = SSVEPfun(flt_data_3[:, _chs_ssvep, :])

		feat_set_P300 = np.array(feat_set_P300)
		tmst_set_P300 = np.array(tmst_set_P300)
		feat_set_SSVEP = np.array(feat_set_SSVEP)

		print(f'LDA Feature Arrary size: {feat_set_P300.shape}')
		print(f'CCA Feature Arrary size: {feat_set_SSVEP.shape}')

		_now_ = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #isoformat(timespec='minutes')
		print(dir_)
		dir__ = dir_.replace('.csv', '')
		filename = dir__+"_"+'validation_log_' + _now_ + '.txt'
		CCA_model = CCA(n_components = COMP_NUM, max_iter = 20000)

		## Validation P300
		# accs, accs_mean = validate_classifier(feat_set_P300, P300_lbls, 0, 10, filename)
		
		# ## Validation SSPVE
		accs, accs_mean = validate_classifier(feat_set_SSVEP, cca_lbls, 1, 10, filename)
		time.sleep(1)























	#############################################################
	########################### SAVES ###########################
	#############################################################
	##>Normalization function
	# print("Normalizing:")
	# print(matrix.shape)
	# mean = np.mean(matrix, 1)
	# # print(mean.shape)
	# # matrix = np.array(matrix)
	# avgsamples = np.subtract(matrix.transpose(), mean).transpose()
	# # print(avgsamples.shape)
	# maxx = np.max(avgsamples, 1)
	# minn = np.min(avgsamples, 1)
	# nom = (np.subtract(avgsamples.transpose(), minn))*(x_max - x_min)
	# denom = np.subtract(maxx, minn)
	# denom[denom==0] = 1
	
		# # avgsamples = np.interp(matrix, (minn, maxx), (-1,1))
		# avgsamples = x_min + nom/denom #(avg-min)/(max-min)
	# print("AVG_SAMPS:",avgsamples.shape)
	# print("End_Normalizing\n\n")
	# print(avgsamples.shape)



	##>Put data into array
	# avg = np.expand_dims(normalization(samples), axis=0)
	# avg_data = np.append(avg_data, avg, axis=0)
	# flt_data = np.append(flt_data, filtering(avg), axis=0)
	# print('SAMPLES.shape', samples.shape)
	# print('AVG.shape', avg.shape)
	# print('AVG_DATA.shape', avg_data.shape)
	# print('FLT_DATA.shape', avg_data.shape)
	# print(avg.shape, avg_data.shape, flt_data.shape)
	# plt.plot(timestamps[x], filtering(avg).T,'b')
	# plt.show()