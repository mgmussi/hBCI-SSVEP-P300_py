'''
UPGRADES ON THIS VERSION:
- Using Fan et al., 2015 method for SSVEP
	Using frq. bins for SSVEP features
	Using LDA for SSVEP
- Using Fan et al., 2015 method for P300
	Using downsampling
	Using PCA for dimension reducion
'''

import sys
import os
import math
from tkinter import filedialog
import pandas as pd
pd.options.display.float_format = '{:20.6f}'.format
import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
import matplotlib.pyplot as plt

import scipy
from scipy.signal import butter, lfilter, kaiserord, firwin, iirnotch, decimate
from scipy import interpolate
from scipy.fft import fft, fftfreq

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import FastICA

rng = np.random.RandomState(None)
ica = FastICA(n_components=6, random_state=0, tol=0.005)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

import mne
from mne.decoding import Scaler
mne_scaler = Scaler(scalings = 'median')

from datetime import datetime
import time

#1
SPS = 250
NYQ = int(SPS*0.5)
#
SLICE = NYQ
#
COMP_NUM = 2
LDA_model = LDA(solver = 'svd')
#new
LDA_model2 = LDA(solver = 'svd')
CCA_model = CCA(n_components = COMP_NUM, max_iter=20000)
#new
PCA_model = PCA(n_components = 2)

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

def filtering(samples):
	#create padding
	pf = np.hstack(( np.flip(samples[:, 0:win], axis = 1), samples[:, :], np.flip(samples[:, -win:], axis = 1) ))
	notch = lfilter(b1, a1, pf, axis = 1)
	s_notch = notch[:,win:-win]
	pf2 = np.hstack(( np.flip(s_notch[:, 0:win], axis = 1), s_notch[:, :], np.flip(s_notch[:, -win:], axis = 1) ))
	#BUTTERWORTH
	# posfilt = lfilter(b, a, pf2, axis = 1)
	#FIR
	posfilt = lfilter(b, [1.0], pf2, axis = 1)
	filtsamples = posfilt[:, win+delay:-win]
	return filtsamples

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
	ch_data_all = []
	ts_data = []
	p300ll_data = []
	p300pos_data = []
	ssvepll_data = []
	ssvepll_data_all = []
	curr_id = 0

	#2
	filename = filedialog.askopenfilenames(initialdir = "C:/Users/atech/Documents/GitHub/SSVEP_P300-py/Participants",
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
		df = pd.read_csv(file)

		#plot all the data
		the_data = np.array([
				[x[0] for x in df.loc[:, ['CH1']].values],
				[x[0] for x in df.loc[:, ['CH2']].values],
				[x[0] for x in df.loc[:, ['CH3']].values],
				[x[0] for x in df.loc[:, ['CH4']].values],
				[x[0] for x in df.loc[:, ['CH5']].values],
				[x[0] for x in df.loc[:, ['CH6']].values],
				[x[0] for x in df.loc[:, ['CH7']].values],
				[x[0] for x in df.loc[:, ['CH8']].values]])
		the_tstm = [t[0] for t in list(df.loc[:, ['Timestamps']].values)]
		plt.plot(the_tstm, the_data.T, 'r')
		plt.show()

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
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH1']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH2']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH3']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH4']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH5']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH6']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH7']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH8']].values]])

			## Get all samples for each frq. change (TL)
			if counter % 21 == 0:
				curr_id = i
				ssvep_all = df.loc[i, ['SSVEP_labels']].values
				ssvepll_data_all.append(ssvep_all[0])

			if counter % 20 == 0 and counter != 0:
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

			## Get labels for each sample
			ts_data.append([t[0] for t in list(df.loc[j-NYQ+1:j, ['Timestamps']].values)])
			p300 = df.loc[i, ['P300_labels']].values
			p300ll_data.append(p300[0])
			p300pos = df.loc[i, ['P300_position']].values
			p300pos_data.append(p300pos[0])
			ssvep = df.loc[i, ['SSVEP_labels']].values
			ssvepll_data.append(ssvep[0])

	ch_data_125 = np.array(ch_data_125)
	ch_data_all = np.array(ch_data_all)
	ts_data = np.array(ts_data)
	p300ll_data = np.array(p300ll_data)
	p300pos_data = np.array(p300pos_data)
	ssvepll_data = np.array(ssvepll_data)
	ssvepll_data_all = np.array(ssvepll_data_all)

	return ch_data_125, ch_data_all, ts_data, p300ll_data, p300pos_data, ssvepll_data, ssvepll_data_all, dir_name

def P300fun(data, timestamps):
	feat_set = []

	for datum, tmst in zip(data, timestamps):
		# try:
		# 	mean_data = np.mean(datum, 0)
		# except RuntimeWarning:
		# 	print('<<<Attempting mean with empty array',ss)
		##::Downsample == Features 63 sample?
		# data_DwnSmp.append(mean_data[::6])
		# tmst_DwnSmp.append(tmst[::6])

		#new
		# Downsample by factor 2
		datum_DwnSmp = decimate(datum, 15, axis = 1)

		# Filter with a bandpass from 0.5 to 15Hz ?

		# PCA dimensionality reduction (from 4 [chs] to 2 components)
		# data_PCA = PCA_model.fit_transform(datum_DwnSmp)
		# print(PCA_model.explained_variance_ratio_)
		# feat_set.append(data_PCA.reshape(-1))

		feat_set.append(datum_DwnSmp.reshape(-1))
	feat_set = np.array(feat_set)
	return feat_set

def SSVEPfun(data):
	feat_set = []
	for datum in data:
		feat = []
		# both_chan = datum.reshape(-1)
		# feat_set.append(both_chan)

		##::Get PSD bins sum for each target frq and its double frq (+-0.5Hz)
		##total of features: chs*2*frqs = 2*2*3 = 12 features
		xfft = fftfreq(SPS*2, 1/SPS) #precision of 0.5Hz
		for s_ch in datum:
			# print(s_ch.shape)
			s_ch_pad = np.pad(s_ch, (0, SPS*2-len(s_ch) ), 'constant')
			yfft = np.abs(fft(s_ch_pad))**2 #PSD
			for frq in [15, 30, 10, 20, 6, 12]: #for each frq and its double frq +-0.5Hz
				idx = [i for i in range(len(xfft)) if xfft[i] >= (frq-0.5) and xfft[i] <= (frq+0.5)]
				feat.append(np.sum(yfft[idx]))
		feat_set.append(feat)
	return feat_set


######################

def validate_classifier(feature_set, label_set, SSVEP_mode, splits, filename):
	f = open(filename, 'a')

	kf = KFold(n_splits = splits, random_state = 42, shuffle = True)
	accs = []
	count = 0
	ini_acc = 0
	the_model = []

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
		if SSVEP_mode:
			print("Training fold SSVEP", count, "\b...")
			# f.write("Number of Components: {}\n".format(COMP_NUM))
			f.write("Training fold LDA {}...\n".format(count))
		else:
			print("Training fold P300", count, "\b...")
			f.write("Training fold LDA {}...\n".format(count))

		X_train, X_test = feature_set[train_index], feature_set[test_index]
		y_train, y_test = label_set[train_index], label_set[test_index]

		if SSVEP_mode:
			##for older files, where it was saving 1, 2 and 4
			##updated files save 0, 1 and 2, so this is unnecessary
			# y_train = np.array([t-2 if t == 4 else t-1 for t in y_train])
			# y_test = np.array([t-2 if t == 4 else t-1 for t in y_test])
			model = LDA_model2.fit(X_train, y_train)
		else:
			model = LDA_model.fit(X_train, y_train)

		##Predict results using model
		y = model.predict(X_test)

		#Conditioning prediction
		new_y = [int(yy) for yy in y]
		bin_y_test = [int(x) for x in y_test]

		##Check
		print(new_y)		#predicted
		print(bin_y_test)	#vs. lbls
		f.write("Pred: {}\n".format(new_y))
		f.write("Lbls: {}\n".format(bin_y_test))

		#Calculating accuracy
		for i in range(len(new_y)):
			if new_y[i] == bin_y_test[i]:
				rgt += 1
		print(rgt, "labels match")
		f.write("{} labels match\n".format(rgt))

		#Choosing best model
		acc = rgt/len(bin_y_test)
		if acc > ini_acc:
			ini_acc = acc
			the_model = model

		accs.append(acc)
		if SSVEP_mode:
			print("ACC LDA SSVEP = ", '{:.3f}'.format(accs[count]*100), "\b%")
			conf_mat = confusion_matrix(bin_y_test, new_y, labels=[0,1,2])
			print("<<<<Confusion Matrix\n", conf_mat, '\n\n')
			f.write("ACC LDA SSVEP = {:.3f}%\n\n<<<<Confusion Matrix\n{}\n\n".format(accs[count]*100, conf_mat))
		else:
			print("ACC LDA P300 = ", '{:.3f}'.format(accs[count]*100), "\b%")
			conf_mat = confusion_matrix(bin_y_test, new_y, labels=[0, 1])
			tn, fp, fn, tp = conf_mat.ravel()
			print("<<<<Confusion Matrix\n", conf_mat)
			print("TN", tn, "FP", fp, "FN", fn, "TP", tp, '\n\n', sep = " : ")
			f.write("ACC LDA P300 = {:.3f}%\n\n<<<<Confusion Matrix\n{}\n\n".format(accs[count]*100, conf_mat))
			f.write("TN: {}, FP: {}, FN: {}, TP: {}\n\n".format(tn, fp, fn, tp))
		count += 1
	accs_mean = np.mean(accs)
	accs_devi = np.std(accs)

	# Rendering Confusion Matrix and Final ACC
	if SSVEP_mode:
		print("<<<<Average ACC SSVEP LDA = ", '{:.3f}±{:.3f}'.format(accs_mean*100, accs_devi*100), "\b%")
		f.write("\nAverage ACC SSVEP LDA = {:.3f}±{:.3f}%\n---------\n---------\n\n\n".format(accs_mean*100, accs_devi*100))
	else:
		print("<<<<Average ACC P300 LDA = ", '{:.3f}±{:.3f}'.format(accs_mean*100, accs_devi*100), "\b%")
		f.write("\nAverage ACC P300 LDA = {:.3f}±{:.3f}%\n---------\n---------\n\n\n".format(accs_mean*100, accs_devi*100))
	f.close()
	return accs, accs_mean, the_model



def fusion(M_P300_CLASSIFIER, M_SSVEP_CLASSIFIER, acc_LDA, acc_CCA, PL_LDA, SSVEP_feat, P300_feat, CCA_lbls, filename):
	f = open(filename, 'a')
	# rounds = 3 #numbers of results that will be fused before giving the final answer
	goal = 3
	timeout = 20
	correct = 0
	predict_arr = []
	lbl_arr = []
	votes_log = []

	print(CCA_lbls.shape, SSVEP_feat.shape)

	for trial in range(0, CCA_lbls.shape[0], 21):
		una_pred = np.zeros((3,1))
		lda_pred = np.zeros((3,1))
		cca_pred = np.zeros((3,1))
		tot_pred = np.zeros((3,1))
		pred_counter = 0
		
		for trial in range(trial, SSVEP_feat.shape[0]): #loop over each trial
			##::Predicting based on trials
			y_CCA  = np.squeeze(M_SSVEP_CLASSIFIER.predict(SSVEP_feat[trial,:].reshape(1,-1)))
			y_LDA  = np.squeeze(int(M_P300_CLASSIFIER.predict(P300_feat[trial,:].reshape(1,-1))))
			pl_LDA = int(PL_LDA[trial])
			##for older files, where it was saving 1, 2 and 4
			##updated files save 0, 1 and 2, so this is unnecessary
			# lbl_d123 = [CCA_lbls[trial]-2 if CCA_lbls[trial] == 4 else CCA_lbls[trial]-1][0]
			lbl_d123 = CCA_lbls[trial]

			##::Decimal Transformation
			# yy_CCA = [y_CCA-2 if y_CCA == 4 else y_CCA-1] #setting values to 0, 1, 2 instead of 1, 2, 4
			yy_CCA = int(y_CCA)

			if y_LDA == 1:
				if pl_LDA == yy_CCA: #if position and CCA selection are the same
					una_pred[yy_CCA] += 2
				else:
					lda_pred[yy_CCA] += 1*acc_LDA
					cca_pred[yy_CCA] += 1*acc_CCA
			elif y_LDA == 0:
				lda_pred[pl_LDA] -= 0.5*acc_LDA
				cca_pred[yy_CCA] += 1*acc_CCA
			pred_counter += 1
			tot_pred = np.sum([una_pred, lda_pred, cca_pred], 0)

			# ##::Print prediction arrays
			# print(np.array([una_pred, lda_pred, cca_pred, tot_pred]).reshape(4,3, order = 'F').T)
			# print("\n\n")

			if any(tot_pred[tot_pred >= goal]) or pred_counter > timeout:
				voted_ans = np.argmax(tot_pred)
				threshold = tot_pred[voted_ans]
				predict_arr.append(voted_ans)
				lbl_arr.append(int(lbl_d123))
				votes_log.append(pred_counter)
				if lbl_d123 == voted_ans:
					correct += 1
				# print("Voted Answer  = {} ({})\t||\tRequired votes = {}".format(voted_ans, threshold, pred_counter))
				# print("True Answer   = {}".format(lbl_d123))
				break


	acc = correct/(CCA_lbls.shape[0]/21)
	conf_mat = confusion_matrix(np.array(lbl_arr), np.array(predict_arr))#, labels=[0, 1, 2])
	print("\n\nPred:\t", predict_arr)	#decimal predicted
	print("Lbls:\t", lbl_arr)  #vs. decimal true lbls
	print("#Votes:\t", votes_log) #votes needed for decision
	print("Fusion ACC: {:.3f}% | SSVEP LDA ACC: {:.3f}% | P300 LDA ACC: {:.3f}%\nCCA Confusion Matrix\n{}".format(acc*100, acc_CCA*100, acc_LDA*100, conf_mat))
	f.write("\n<<<Fusion\nPred:\t{}".format(predict_arr))
	f.write("\nLbls:\t{}".format(lbl_arr))
	f.write("\n#Votes:\t{}".format(votes_log))
	f.write("\nFusion ACC: {:.3f}% | SSVEP LDA ACC: {:.3f}% | P300 LDA ACC: {:.3f}%\n\nCCA Confusion Matrix\n{}s".format(acc*100, acc_CCA*100, acc_LDA*100, conf_mat))
	f.close()



#################################################
#################################################
#################################################
#################################################




if __name__ == '__main__':
	'''
	#Paradigm		#IDEAL CHs 				#CYTON 				#GTEC
	#P300 		 	#PO8, PO7, POz, CPz		#P3, Pz, P4, Cz		#C3, Cz, C4, Pz
	#SSVEP 			#O1, Oz, O2 			#O1, O2 			#O1, O2, Oz
	'''
	#CYTON
	chs_p300 = [2,3,4,5]
	chs_ssvep = [6,7]

	#G.TEC
	# chs_p300 = [2,3,4,5]
	# chs_ssvep = [6,7,8]


	chs_all = list(set(chs_p300 + chs_ssvep))
	_chs_p300 = list(range(len(chs_p300)))
	_chs_ssvep = list(range(_chs_p300[-1]+1, _chs_p300[-1]+1+len(chs_ssvep)))
	_chs_all = [x - 1 for x in chs_all]
	# avg_data = []
	# avg_data = np.empty([])
	# flt_data = []
	# flt_data = np.empty([])
	cca_lbls = []
	cca_lbls_all = []

	## Receives ALL samples from .csv
	eeg_data, eeg_data_all, timestamps, P300_lbls, P300_pos, SSVEP_lbls, SSVEP_lbls_all, dir_ = get_data_files() #all numpy arrays

	# average_across_trials(eeg_data)
	# sys.exit()

	## Creating Empty arrays
	avg_data = np.empty((eeg_data.shape[0], len(_chs_all), eeg_data.shape[2]-delay), float)
	flt_data = np.empty((eeg_data.shape[0], len(_chs_all), eeg_data.shape[2]-delay), float)
	flt_data_ICA = np.empty((eeg_data.shape[0], len(_chs_all), eeg_data.shape[2]-delay), float)
	avg_data_all = np.empty((eeg_data_all.shape[0], len(_chs_all), eeg_data_all.shape[2]-delay), float)
	flt_data_all = np.empty((eeg_data_all.shape[0], len(_chs_all), eeg_data_all.shape[2]-delay), float)

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
	print('EEG_DATA_ALL.shape', eeg_data_all.shape)

	#setting up folder to save images
	new_folder = dir_[0:-4]
	if not os.path.exists(new_folder):
		os.makedirs(new_folder)

	for x in range(eeg_data.shape[0]):
		##:: 1 - FILTERING
		flt_data[x,:,:] = filtering(eeg_data[x, _chs_all, :]) #shape << [n_chs, samples]

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
		# remove_indices = [0, 5]
		# comps[:, remove_indices] = 0
		# flt_data_ICA[x,:,:] = ica.inverse_transform(comps).T
		# #plot chs to compare w/ and wo/ ica filter
		# fig = plt.figure()
		# plt.title("Epoch " + str(x))
		# for ch in range(len(_chs_all)):
		# 	plt.subplot(2, 4, ch+1)
		# 	plt.plot(flt_data[x, ch, 2:].T, 'b')
		# 	plt.plot(flt_data_ICA[x, ch, 2:].T, 'r')
		# 	plt.title("Channel " + str(ch))
		# _img = u'Ch_comparison_%s.png' % x
		# _img_path = os.path.join(new_folder, _img)
		# plt.savefig(_img_path)

		##:: 3 - SCALE/NORMALIZE
		# avg_data[x,:,:] = scaler.fit_transform(flt_data[x,:,:])
		avg_data[x,:,:] = scaler.fit_transform(flt_data[x,:,:].T).T
		# avg_data[x,:,:] = np.squeeze(mne_scaler.fit_transform(flt_data[x,:,:]))
		# avg_data[x,:,:] = np.squeeze(mne_scaler.fit_transform(flt_data_ICA[x,:,:]))

		msg = "Processing... |" + '\u2588' * math.floor( (( (x/eeg_data.shape[0])*100) % 100)/2 ) + '\u2591' * math.ceil(50-( ( (x/eeg_data.shape[0])*100) % 100)/2 ) + '| x' + str(x) + ' ~ '+'{:.2f}%'.format((x/eeg_data.shape[0])*100)
		print(msg, end="\r")
	print('')

	#process all SSVEP data (instead of chunks)
	for x in range(eeg_data_all.shape[0]):
		flt_data_all[x,:,:] = filtering(eeg_data_all[x, _chs_all, :])
		avg_data_all[x,:,:] = scaler.fit_transform(flt_data_all[x,:,:].T).T
		# avg_data_all[x,:,:] = np.squeeze(mne_scaler.fit_transform(flt_data_all[x,:,:]))

	avg_data = np.array(avg_data)
	flt_data = np.array(flt_data)
	avg_data_all = np.array(avg_data_all)
	flt_data_all = np.array(flt_data_all)

	# Conditioning SSVEP Labels to Binary
	# for S_lbl in SSVEP_lbls:
	# 	cca_lbls.append(d2b(S_lbl))
	cca_lbls = np.array(SSVEP_lbls)

	# for S_lbl_all in SSVEP_lbls_all:
	# 	cca_lbls_all.append(d2b(S_lbl_all))
	cca_lbls_all = np.array(SSVEP_lbls_all)

	## Processing for P300
	feat_set_P300 = P300fun(avg_data[:, _chs_p300, :], timestamps)
	feat_set_P300 = np.array(feat_set_P300)

	## Processing for SSVEP
	feat_set_SSVEP = SSVEPfun(avg_data[:, _chs_ssvep, :])
	feat_set_SSVEP = np.array(feat_set_SSVEP)
	print(feat_set_SSVEP.shape)

	# feat_set_SSVEP_all = SSVEPfun(avg_data_all[:, _chs_ssvep, :])
	# feat_set_SSVEP_all = np.array(feat_set_SSVEP_all)

	print(f'LDA Feature Arrary size: {feat_set_P300.shape}')
	print(f'CCA Feature Arrary size: {feat_set_SSVEP.shape}')
	# print(f'CCA Feature Arrary All size: {feat_set_SSVEP_all.shape}')
	print(f'P300 Position Arrary size: {P300_pos.shape}')
	print(f'P300 Position Arrary size: {P300_pos.shape}')

	_now_ = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #isoformat(timespec='minutes')
	print(dir_)
	dir__ = dir_.replace('.csv', '')
	filename = dir__+'_validation_log_' + _now_ + '.txt'

	## Validation P300
	accs_LDA, accs_mLDA, M_P300_CLASSIFIER = validate_classifier(feat_set_P300, P300_lbls, 0, 10, filename)
	
	# ## Validation SSPVE
	accs_CCA, accs_mCCA, M_SSVEP_CLASSIFIER = validate_classifier(feat_set_SSVEP, SSVEP_lbls, 1, 10, filename)
	# accs, accs_mean = validate_classifier(feat_set_SSVEP_all, cca_lbls_all, 1, 5, filename)


	##::ONLINE CLASSIFICATION SIMULATION: (LDA + LDA) => FUSION
	fusion(M_P300_CLASSIFIER, M_SSVEP_CLASSIFIER, accs_mLDA, accs_mCCA, P300_pos, feat_set_SSVEP, feat_set_P300, cca_lbls, filename)
  # fusion(M_P300_CLASSIFIER, M_SSVEP_CLASSIFIER, acc_LDA,   acc_CCA,   PL_LDA,   SSVEP_feat,     P300_feat,     CCA_lbls)


















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