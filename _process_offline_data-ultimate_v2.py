'''
ON THIS VERSION:
- Simulating an increasing window size for SSVEP
- Used to (tentatively) increase FFT's precision
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
ica = FastICA(n_components=6, random_state=42, tol=0.5)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

import mne
from mne.decoding import Scaler
mne_scaler = Scaler(scalings = 'median')

from datetime import datetime
import time

###DEFINITIONS
SPS = 250
NYQ = int(SPS*0.5)
#
SLICE = NYQ
#
COMP_NUM = 2
LDA_model = LDA(solver = 'svd')
LDA_model2 = LDA(solver = 'svd')
CCA_filter = CCA(n_components = 1)
PCA_model = PCA(n_components = 2)
paradigmSSVEP = True
paradigmP300 = True


###FILTERING
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
print('\n')

def conditioning(samples):
	#create padding
	pf = np.hstack(( np.flip(samples[:, 0:win], axis = 1), samples[:, :], np.flip(samples[:, -win:], axis = 1) ))
	notch = lfilter(b1, a1, pf, axis = 1)
	conditioned = notch[:,win:-win]# s_notch = notch[:,win:-win]
	# ## BUTTERWORTH
	# # posfilt = lfilter(b, a, pf2, axis = 1)
	# ## FIR
	# pf2 = np.hstack(( np.flip(s_notch[:, 0:win], axis = 1), s_notch[:, :], np.flip(s_notch[:, -win:], axis = 1) ))
	# posfilt = lfilter(b, [1.0], pf2, axis = 1)
	# conditioned = scaler.fit_transform(posfilt[:, win+delay:-win].T).T
	return conditioned
	
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

def get_data_files(filename):
	ch_data = []
	ts_data = []
	p300ll_data = []
	p300pos_data = []
	ssvepll_data = []
	curr_id = 0

	if np.array(filename).shape[0] > 1:
		print("\n[[Validation Data]]")
		valid_ch_data = []
		valid_ts_data = []
		valid_p300ll_data = []
		valid_p300pos_data = []
		valid_ssvepll_data = []
	else:
		print("\n[[Training Data]]")

	for file in filename:
		print("[Getting data from]", file)
		df = pd.read_csv(file)

		## Get df indexes:
		idx1 = df.index[df['P300_ts'] == 0]
		idx2 = df.index[df['P300_ts'] == 2]

		for i, j in zip(idx1, idx2):
			## Get the last 125 samples
			ch_data.append([
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH1']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH2']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH3']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH4']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH5']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH6']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH7']].values],
				[x[0] for x in df.loc[j-SLICE+1:j, ['CH8']].values]])

			## Get labels for each sample
			ts_data.append([t[0] for t in list(df.loc[j-NYQ+1:j, ['Timestamps']].values)])
			p300ll_data.append(df.loc[i, ['P300_labels']].values[0])
			p300pos_data.append(df.loc[i, ['P300_position']].values[0])
			ssvepll_data.append(df.loc[i, ['SSVEP_labels']].values[0])

		if np.array(filename).shape[0] > 1:
			valid_ch_data.append(ch_data)
			valid_ts_data.append(ts_data)
			valid_p300ll_data.append(p300ll_data)
			valid_p300pos_data.append(p300pos_data)
			valid_ssvepll_data.append(ssvepll_data)
			ch_data = []
			ts_data = []
			p300ll_data = []
			p300pos_data = []
			ssvepll_data = []

	if np.array(filename).shape[0] > 1:
		ch_data = valid_ch_data
		ts_data = valid_ts_data
		p300ll_data = valid_p300ll_data
		p300pos_data = valid_p300pos_data
		ssvepll_data = valid_ssvepll_data
		ch_data = np.array(ch_data, dtype = object)
		ts_data = np.array(ts_data, dtype = object)
		p300ll_data = np.array(p300ll_data, dtype = object)
		p300pos_data = np.array(p300pos_data, dtype = object)
		ssvepll_data = np.array(ssvepll_data, dtype = object)
	else:
		ch_data = np.array(ch_data)
		ts_data = np.array(ts_data)
		p300ll_data = np.array(p300ll_data)
		p300pos_data = np.array(p300pos_data)
		ssvepll_data = np.array(ssvepll_data)

	# if not(np.array(filename).shape[0] > 1):
	# 	ch_data = np.expand_dims(ch_data, axis=0)
	# 	ts_data = np.expand_dims(ts_data, axis=0)
	# 	p300ll_data = np.expand_dims(p300ll_data, axis=0)
	# 	p300pos_data = np.expand_dims(p300pos_data, axis=0)
	# 	ssvepll_data = np.expand_dims(ssvepll_data, axis=0)

	return ch_data, ts_data, p300ll_data, p300pos_data, ssvepll_data

def P300fun(data):
	feat_set = []

	for datum in data:

		# Downsample
		datum_DwnSmp = decimate(datum, 15, axis = 1)

		# Filter with a bandpass from 0.5 to 15Hz ?

		# PCA dimensionality reduction (from 4 [chs] to 2 components)
		# data_PCA = PCA_model.fit_transform(datum_DwnSmp)
		# print(PCA_model.explained_variance_ratio_)
		# feat_set.append(data_PCA.reshape(-1))

		feat_set.append(datum_DwnSmp.reshape(-1))
	feat_set = np.array(feat_set)
	return feat_set

def SSVEPfun(data, validation, perf_file):
	print("\n[FEAT_EXTRACT SSVEP]")
	print(data.shape)

	#new
	###FILTER BANK
	lows = [x/NYQ for x in range(5, 105, 10)]
	high = 105/NYQ

	#new
	##Creating index-change arrays
	#training
	if not validation:
		change_idx = []
		for i in range(0,316,21):
			change_idx.append(i)
		# print(change_idx)
	#validation
	else:
		#Extracting max num of votes
		with open(perf_file) as f:
			lines = f.readlines()
			num_votes = lines[-6].replace('\t>#Vts: ', '')
			num_votes = num_votes.replace(',', '')
			num_votes = num_votes.replace('[', '')
			num_votes = num_votes.replace(']', '')
			num_votes = num_votes.split()
			num_votes = [int(x) for x in num_votes]
			print('Max votes: ',num_votes)
			f.close()
		#Creating change array
		change_idx = [0]
		for i in range(len(num_votes)):
			change_idx.append(change_idx[-1] + num_votes[i])
		# print('Change idx: ',change_idx)


	#new
	#calculating reference waves for CCA
	ref_bank = []
	for frq in [15, 10, 6]:
		ref = [ ]
		ks = np.arange(SPS/2-delay) / SPS
		for i in range(3):
			ref.append(np.sin(2 * np.pi * (i+1) * frq * ks))
			ref.append(np.cos(2 * np.pi * (i+1) * frq * ks))
		ref_bank.append(np.array(ref).T)
	ref_bank = np.array(ref_bank) #[frqs x samples x harm*(sin, cos)] = [3, 112, 4]
#delete
	print(ref_bank.shape)

	###WEIGHT
	a = 1.25
	b = 0.25
	w = [((x+1)**-a) + b for x in range(ref_bank.shape[2])]

	#new
	feat_set = []
	pred_sz_set = []
	for i in range(len(change_idx)-1): #per stimulus
		inc_signal = np.empty((data[change_idx[i],:,:].shape[0],0)) #creating empty array
		finc = 0
		for j in range(change_idx[i], change_idx[i+1]): #loop over each trial within stimulus
			feat = []
			p = np.empty((10, ref_bank.shape[0]))
			finc += 1

			# incrementally append signals
			if finc >= 1:
				inc_signal = np.empty((data[change_idx[i],:,:].shape[0],0)) #creating empty array
				finc = 1
			inc_signal = np.append(inc_signal, data[j,:,:], axis = 1)
			# print(inc_signal.shape)
			# print("::", SPS*2*finc)

			# '''-------------
			# Get PSD bins sum for each target frq and its double frq (+-0.5Hz)
			# Total of features: chs*2*frqs = 5*2*3 = 30 features
			# -------------'''
			# xfft = fftfreq(SPS*2*finc, 1/SPS) #precision of 0.5Hz (double the number of samples)
			# for s_ch in inc_signal:
			# 	s_ch_pad = np.hstack((s_ch, np.zeros((SPS*2*finc-len(s_ch))))) #zero padding at the end
			# 	yfft = np.abs(fft(s_ch_pad))**2 #PSD
			# 	for frq in [15, 30, 10, 20, 6, 12]: #for each frq and its double frq +-0.5Hz
			# 		idx = [i for i in range(len(xfft)) if xfft[i] >= (frq-0.5) and xfft[i] <= (frq+0.5)]
			# 		feat.append(np.sum(yfft[idx]))


			for n, low in enumerate(lows):
				b = firwin(N, [low, high], pass_zero = 'bandpass',  window=('kaiser', beta))
				pf2 = np.hstack(( np.flip(inc_signal[:, 0:win], axis = 1), inc_signal[:, :], np.flip(inc_signal[:, -win:], axis = 1) ))
				posfilt = lfilter(b, [1.0], pf2, axis = 1)
				conditioned = scaler.fit_transform(posfilt[:, win+delay:-win].T).T

				for k, y in enumerate(ref_bank):
					# print(conditioned.T.shape, y.shape)
					x_, y_ = CCA_filter.fit_transform(conditioned.T, y)
					r = np.corrcoef(x_.T, y_.T)[0, 1]
					p[n,k] = r

			for k in range(len(lows)):
				feat.append(sum([w[n]*(p[k,n]**2) for n in range(ref_bank.shape[0])]))
			feat_set.append(feat)
			pred_sz_set.append(inc_signal.shape[1])
	feat_set = np.array(feat_set)
	print('Feature set = ', feat_set.shape)
	return feat_set, pred_sz_set

def P300_pred(the_model, feat_set, label_set, pl_set, filename):
	pred_set = []
	rgt = 0
	for feat in feat_set:
		y = the_model.predict(feat.reshape(1,-1))
		pred_set.append(y)

	##::Conditioning
	int_pred = [int(x) for x in pred_set]
	int_lbls = [int(x) for x in label_set]
	int_pl = [int(x) for x in pl_set]
	conf_mat = confusion_matrix(int_lbls, int_pred)

	##::Calculating accuracy
	for i in range(len(int_pred)):
		if int_pred[i] == int_lbls[i]:
			rgt += 1
	acc = rgt/len(int_lbls)

	f = open(filename, 'w')
	f.write('[LIVE RESULTS FOR P300 CLASSIFICATION]')
	f.write('\n\t[PREDICTION RESULTS]\n\t>Pred: {}\n'.format(int_pred))
	f.write('\t>Lbls: {}\n'.format(int_lbls))
	f.write('\t>PosL: {}\n'.format(int_pl))
	f.write("\t>Correct: {}/{}\n\t>ACC LDA P300 = {:.3%}\n\t>Confusion Matrix:\n{}\n\n".format(rgt,len(int_lbls),acc, conf_mat))
	f.close()
	return np.array(int_pred)

def SSVEP_pred(the_model, feat_set, label_set, filename, train_pred_sz_set):
	pred_set = []
	rgt = 0
	for feat in feat_set:
		# y = the_model.predict(feat.reshape(1,-1))
		
		pred_set.append(y)

	int_pred = [int(x) for x in pred_set]
	int_lbls = [int(x) for x in label_set]
	conf_mat = confusion_matrix(int_lbls, int_pred)

	##::Calculating accuracy
	for i in range(len(int_pred)):
		if int_pred[i] == int_lbls[i]:
			rgt += 1
	acc = rgt/len(int_lbls)

	f = open(filename, 'a')
	f.write('[LIVE RESULTS FOR SSVEP CLASSIFICATION]')
	f.write('\n\t[PREDICTION RESULTS]\n\t>Pred: {}\n'.format(int_pred))
	f.write('\t>Lbls: {}\n'.format(int_lbls))
	# f.write('\t>PrSz: {}\n'.format(train_pred_sz_set))
	f.write("\t>Correct: {}/{}\n\t>ACC LDA SSVEP = {:.3%}\n\t>Confusion Matrix:\n{}\n".format(rgt,len(int_lbls),acc, conf_mat))
	f.close()
	return np.array(int_pred)


######################

def validate_classifier(feature_set, label_set, SSVEP_mode, splits, filename):
	if SSVEP_mode:
		f = open(filename, 'a')
	else:
		f = open(filename, 'w')

	kf = KFold(n_splits = splits, random_state = 42, shuffle = True)
	accs = []
	count = 0
	ini_acc = 0
	the_model = []

	final_ans = np.array([])

	print(f'\n\n\n\n\'feat_set\' arrary size: {feature_set.shape}')
	print(f'\'label_set\' arrary size: {label_set.shape}')
	f.write('\'feat_set\' arrary size: {}\n'.format(feature_set.shape))
	f.write('\'label_set\' arrary size: {}\n'. format(label_set.shape))

	if SSVEP_mode:
		rgt = 0
		final_ans = feature_set.argmax(axis = 1)
		for x, _ in enumerate(final_ans):
			# print(final_ans[x], 'vs', label_set[x])
			if final_ans[x] == int(label_set[x]):
				rgt += 1
			count += 1
		#Choosing best model
		acc = rgt/len(label_set)
		conf_mat = confusion_matrix(label_set, final_ans, labels=[0,1,2])
		print("ACC LDA SSVEP = ", '{:.3%}'.format(acc))
		print("<<<<Confusion Matrix\n", conf_mat, '\n\n')
		sys.exit()

	else:
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

	if SSVEP_mode:
		the_model = LDA_model2.fit(feature_set, label_set)
	else:
		the_model = LDA_model.fit(feature_set, label_set)

	# Rendering Confusion Matrix and Final ACC
	if SSVEP_mode:
		print("<<<<Average ACC SSVEP LDA = ", '{:.3f}±{:.3f}'.format(accs_mean*100, accs_devi*100), "\b%")
		f.write("\nAverage ACC SSVEP LDA = {:.3f}±{:.3f}%\n---------\n---------\n\n\n".format(accs_mean*100, accs_devi*100))
	else:
		print("<<<<Average ACC P300 LDA = ", '{:.3f}±{:.3f}'.format(accs_mean*100, accs_devi*100), "\b%")
		f.write("\nAverage ACC P300 LDA = {:.3f}±{:.3f}%\n---------\n---------\n\n\n".format(accs_mean*100, accs_devi*100))
	f.close()
	return accs, accs_mean, the_model



def fusion(acc_LDA_P300, acc_LDA_SSVEP, PL_LDA, Y_SSVEP, Y_P300, True_lbls, filename, perf_file):
	goal = 3
	correct = 0
	predict_arr = []
	lbl_arr = []
	votes_log = []

	# print(len(True_lbls), Y_SSVEP.shape, Y_P300.shape)

	#Extracting max num of votes
	with open(perf_file) as f:
		lines = f.readlines()
		num_votes = lines[-6].replace('\t>#Vts: ', '')
		num_votes = num_votes.replace(',', '')
		num_votes = num_votes.replace('[', '')
		num_votes = num_votes.replace(']', '')
		num_votes = num_votes.split()
		num_votes = [int(x) for x in num_votes]
		print('Max votes: ',num_votes)
		f.close()

	#Creating change array
	change_idx = [0]
	for i in range(len(num_votes)):
		change_idx.append(change_idx[-1] + num_votes[i])
	print('Change idx: ',change_idx)

	required_samples = 0
	for i in range(len(change_idx)-1):
		una_pred = np.zeros((3,1))
		p300_lda_pred = np.zeros((3,1))
		ssvep_lda_pred = np.zeros((3,1))
		tot_pred = np.zeros((3,1))
		pred_counter = 0
		cumul_clas = 1
		prev_clas = -1
		prev_count = 1
		for j in range(change_idx[i], change_idx[i+1]): #loop over each trial
			timeout = num_votes[i]-1
			pl_LDA = int(PL_LDA[j])
			tru_lbl_d = int(True_lbls[j])
			y_SSVEP = int(Y_SSVEP[j])

			if prev_clas == y_SSVEP:
				cumul_clas = cumul_clas*1.1
				prev_count += 1 			#new
			else:
				prev_count = 1
				prev_clas = y_SSVEP
				cumul_clas = 1

			if paradigmP300 and paradigmSSVEP:
				if Y_P300[j] == 1:
					if pl_LDA == y_SSVEP: #if position and CCA selection are the same
						una_pred[y_SSVEP] += 1*acc_LDA_P300 + 1*acc_LDA_SSVEP #new
					else:
						p300_lda_pred[y_SSVEP] += 1*acc_LDA_P300
						ssvep_lda_pred[y_SSVEP] += 1*acc_LDA_SSVEP*cumul_clas #new
						ssvep_lda_pred[~(np.arange(len(ssvep_lda_pred)) == y_SSVEP)] -= 0.5*acc_LDA_SSVEP #new
				elif Y_P300[j] == 0:
					p300_lda_pred[pl_LDA] -= 0.5*acc_LDA_P300
					p300_lda_pred[~(np.arange(len(p300_lda_pred)) == pl_LDA)] += 0.25*acc_LDA_P300 #new
					ssvep_lda_pred[y_SSVEP] += 1*acc_LDA_SSVEP*cumul_clas #new
					ssvep_lda_pred[~(np.arange(len(ssvep_lda_pred)) == y_SSVEP)] -= 0.5*acc_LDA_SSVEP #new
				pred_counter += 1
				tot_pred = np.sum([una_pred, p300_lda_pred, ssvep_lda_pred], 0)

				##::Print prediction arrays
				print(np.array([una_pred, p300_lda_pred, ssvep_lda_pred, tot_pred]).reshape(4,3, order = 'F').T)
				print("\n")

			##P300
			elif paradigmP300 and not paradigmSSVEP:
				if Y_P300[j] == 1:
					tot_pred[pl_LDA] += 2*acc_LDA_P300
				else:
					tot_pred[pl_LDA] -= 0.5*acc_LDA_P300
					tot_pred[~(np.arange(len(tot_pred)) == pl_LDA)] += 0.25*acc_LDA_P300 #new
				pred_counter += 1

				# ##::Print prediction arrays
				print(np.array(tot_pred).T)
				print("\n")
			
			##SSVEP
			elif paradigmSSVEP and not paradigmP300:
				# if prev_count >= 4:
				# 	tot_pred[y_SSVEP] *= 1.5
				# tot_pred[y_SSVEP] += acc_LDA_SSVEP #*cumul_clas #new
				# tot_pred[~(np.arange(len(tot_pred)) == y_SSVEP)] -= 0.5*acc_LDA_SSVEP
				# pred_counter += 1

				tot_pred[y_SSVEP] += 2*acc_LDA_SSVEP*cumul_clas #new
				tot_pred[~(np.arange(len(tot_pred)) == y_SSVEP)] -= 0.5*acc_LDA_SSVEP
				pred_counter += 1

				# ##::Print prediction arrays
				print(np.array(tot_pred).T)
				print("\n")

			if any(tot_pred[tot_pred >= goal]) or pred_counter > timeout:
				prev_count = 1
				prev_clas = -1
				cumul_clas = 1
				if pred_counter > timeout:
					print('Timeout!')
				voted_ans = np.argmax(tot_pred)
				threshold = tot_pred[voted_ans]
				predict_arr.append(voted_ans)
				lbl_arr.append(tru_lbl_d)
				votes_log.append(pred_counter)
				if tru_lbl_d == voted_ans:
					correct += 1
				required_samples = sum(votes_log)
				print("Voted Answer  = {} ({})\t||\tRequired votes = {}".format(voted_ans, threshold, pred_counter))
				print("True Answer   = {}".format(tru_lbl_d))
				print('# Correct:', correct)
				print('# Rq.Samp:', required_samples)
				break

	acc = correct/15
	conf_mat = confusion_matrix(np.array(lbl_arr), np.array(predict_arr))#, labels=[0, 1, 2])
	f = open(filename, 'a')
	print("\n\nPred:\t", predict_arr)	#decimal predicted
	print("Lbls:\t", lbl_arr)  #vs. decimal true lbls
	print("#Votes:\t", votes_log) #votes needed for decision
	print("Fusion ACC: {:.3f}% | SSVEP LDA ACC: {:.3f}% | P300 LDA ACC: {:.3f}%\nCCA Confusion Matrix\n{}".format(acc*100, acc_LDA_SSVEP*100, acc_LDA_P300*100, conf_mat))
	f.write("\n<<<Fusion\nPred:\t{}".format(predict_arr))
	f.write("\nLbls:\t{}".format(lbl_arr))
	f.write("\n#Votes:\t{}".format(votes_log))
	f.write("\nFusion ACC: {:.3f}% | SSVEP LDA ACC: {:.3f}% | P300 LDA ACC: {:.3f}%\n#Req. Samples:\t{}\n\nCCA Confusion Matrix\n{}".format(acc*100, acc_LDA_SSVEP*100, acc_LDA_P300*100, required_samples, conf_mat))
	f.close()



#################################################
#################################################
#################################################
#################################################


if __name__ == '__main__':

	## Selecting Files
	#EX>: C:\Users\atech\Documents\GitHub\SSVEP_P300-py\Participants\P02\SESS01s\P02_01_01_saveddata.csv
	filename = filedialog.askopenfilenames(initialdir = "C:/Users/atech/Documents/GitHub/SSVEP_P300-py/Participants",
		title = "Select Trial", filetypes = (("Comma Separated Values", "*.csv*"), ("all files","*.*")))
	fidx = filename[0].rindex('_', 0, filename[0].rindex('_')) + 1
	filename_1 = filename[0][:fidx]
	filename_2 = filename[0][fidx+2:]
	train_file = [filename_1 + '01' + filename_2]
	valid_file = []
	perf_file = []
	for i in range(2,6):
		valid_file.append(filename_1 + '{:02}'.format(i) + filename_2)
		perf_file.append(filename_1 + '{:02}'.format(i) + '_performance.txt')

	## Checking paradigm
	fidx = filename[0].rindex('/')
	char = filename[0][fidx-1:fidx]
	if char == 'h':
		paradigmSSVEP = True
		paradigmP300 = True
	elif char == 'p':
		paradigmSSVEP = False
		paradigmP300 = True
	elif char == 's':
		paradigmSSVEP = True
		paradigmP300 = False

	## Checking participant
	P = filename[0][fidx+2:fidx+4] #from 'P02' takes only '02'
	P_num = int(P)
	'''
	Paradigm		IDEAL CHs 				CYTON 				GTEC
	[P300] 		 	[PO8, PO7, POz, CPz]	[P3, Pz, P4, Cz]	[C3, Cz, C4, Pz]
	[SSVEP] 		[O1, Oz, O2]			[Pz, O1, O2] 		[O1, O2, Oz]
	'''
	#CYTON
	chs_p300 = [2,3,4,5]
	if P_num < 4:
		chs_ssvep = [3,6,7]
	elif P_num > 3:
		chs_ssvep = [1,3,8,6,7]

	#G.TEC
	# chs_p300 = [2,3,4,5]
	# chs_ssvep = [6,7,8]

	chs_all = list(set(chs_p300 + chs_ssvep))
	_chs_all = [x - 1 for x in chs_all]
	_chs_p300 = [x-min(chs_all) for x in chs_p300]
	_chs_ssvep = [x-min(chs_all) for x in chs_ssvep]


	## Receives training data
	train_eeg, train_tmst, train_P300_lbls, train_P300_pos, train_SSVEP_lbls = get_data_files(train_file)

	## Receives validation data
	valid_eeg, valid_tmst, valid_P300_lbls, valid_P300_pos, valid_SSVEP_lbls = get_data_files(valid_file)


	'''------------------
	REMINDER
	'valid_eeg', 'valid_tmst', 'valid_P300_lbls', 'valid_P300_pos' and
	'valid_SSVEP_lbls' are a dtype=object array to access individual
	sets as arrays, use 'np.array(valid_SSVEP_lbls[i])'
	------------------'''

	## Conditioning Training Data
	train_flt_data = []
	for x in range(train_eeg.shape[0]):
		train_flt_data.append(conditioning(train_eeg[x, _chs_all, :]))
		msg = "Conditioning Train Data |" + '\u2588' * math.floor( (( (x/train_eeg.shape[0])*100) % 100)/2 ) + '\u2591' * math.ceil(50-( ( (x/train_eeg.shape[0])*100) % 100)/2 ) + '| x' + str(x) + ' ~ '+'{:.2f}%'.format((x/train_eeg.shape[0])*100)
		print(msg, end="\r")
	print('')
	print('train_flt_data.shape', np.array(train_flt_data).shape)
	train_flt_data = np.array(train_flt_data)

	## Conditioning Validation Data
	valid_flt_data = []
	for y in range(valid_eeg.shape[0]):
		valid_bin = []
		_valid_eeg = np.array(valid_eeg[y])
		for x in range(_valid_eeg.shape[0]):
			valid_bin.append(conditioning(_valid_eeg[x, _chs_all, :]))
			msg = "Conditioning Valid Data |" + '\u2588' * math.floor( (( (x/_valid_eeg.shape[0])*100) % 100)/2 ) + '\u2591' * math.ceil(50-( ( (x/_valid_eeg.shape[0])*100) % 100)/2 ) + '| x' + str(x) + ' ~ '+'{:.2f}%'.format((x/_valid_eeg.shape[0])*100)
			print(msg, end="\r")
		print('')
		print('valid_bin.shape', np.array(valid_bin).shape)
		valid_flt_data.append(np.array(valid_bin))

	## Creating txt files for saving
	dir__ = train_file[0].replace('.csv', '')
	train_file_txt = dir__+'_offline_processing.txt'

	valid_file_txt = []
	for file in valid_file:
		dir__ = file.replace('.csv', '')
		valid_file_txt.append(dir__+'_offline_processing.txt')

	## TRAINING SET
	#Feature Extraction
	train_feat_set_P300 = P300fun(train_flt_data[:, _chs_p300, :])
	train_feat_set_SSVEP, train_pred_sz_set = SSVEPfun(train_flt_data[:, _chs_ssvep, :], 0, '')

	#10-Fold Validation with Training data
	accs_LDA_P300, accs_mLDA_P300, M_P300_CLASSIFIER = validate_classifier(train_feat_set_P300, train_P300_lbls, 0, 10, train_file_txt)
	accs_LDA_SSVEP, accs_mLDA_SSVEP, M_SSVEP_CLASSIFIER = validate_classifier(train_feat_set_SSVEP, train_SSVEP_lbls, 1, 10, train_file_txt)

	## VALIDATION SET
	# valid_feat_set_P300 = []
	# valid_feat_set_SSVEP = []
	for i in range(len(valid_flt_data)):
		_valid_flt = valid_flt_data[i]

		#Feature Extraction
		P300_set = P300fun(_valid_flt[:, _chs_p300, :])
		SSVEP_set, valid_pred_sz_set = SSVEPfun(_valid_flt[:, _chs_ssvep, :], 1, perf_file[i])

		#Classification
		Y_P300 = P300_pred(M_P300_CLASSIFIER, P300_set, valid_P300_lbls[i], valid_P300_pos[i], valid_file_txt[i])
		Y_SSVEP = SSVEP_pred(M_SSVEP_CLASSIFIER, SSVEP_set, valid_SSVEP_lbls[i], valid_file_txt[i], valid_pred_sz_set)

		#Fusion
		fusion(accs_mLDA_P300, accs_mLDA_SSVEP, valid_P300_pos[i], Y_SSVEP, Y_P300, valid_SSVEP_lbls[i], valid_file_txt[i], perf_file[i])



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


		# #Finding labels change
	# shift_lbls = np.roll(True_lbls, 1)
	# change_lbls = np.abs(True_lbls-shift_lbls)
	# change_lbls[0] = 0
	# change_idx = np.where(change_lbls > 0)[0]
	# print(change_idx)