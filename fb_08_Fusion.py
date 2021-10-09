#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Used to create flashing boxes for the SSVEP paradigm
For 60Hz monitor:
MonFr/ DesFr = NumFr (NumFr/2 >> ON, NumFr/2 >> OFF)
60/30 = 2 (1 on, 1 off)
60/15 = 4 (2 on, 2 off)
60/10 = 6 ...
60/7.5 = 8
60/6 = 10
60/5 = 12

To control frames, use if(NFrame % NumFr >= NumFr/2) which will make cycle 50-50% ON-OFF
"""
import os
import pickle
import csv
from psychopy import core, visual, gui
import math
import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
import random
import time
from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import queue
from queue import Empty
from datetime import datetime
import pandas as pd
pd.options.display.float_format = '{:20.6f}'.format
import scipy
from scipy.signal import butter, lfilter
import cProfile, pstats, io
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes#, AggOperation
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from scipy.fft import fft, fftfreq

import warnings
warnings.filterwarnings("error")
warnings.filterwarnings('ignore', 'Solver terminated early.*') #for when "sklearn.exceptions.ConvergenceWarning: "Maximum number of iterations reached
warnings.filterwarnings('ignore', 'Maximum number of iterations reached')

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

##CLASSES
class FlickeringBoxes: #using monitor @ 120Hz = 12, 15, 20 or 6, 10, 15
	def __init__(self, position):
		#Definitions
		HOLD.clear() #print('h')
		self.H = 800 #window hight (to fit side of the window)
		self.W = 1920 #window width

		self.sS = 0.125			#square size
		self.pS = self.sS*1.35	#padding size
		self.p3S = self.sS*1.75	#p300 size

		self.p =[[-0.5, -0.5],[0, 0.5],[0.5, -0.5]] #from the centre; 1 is for 100%, 0 is for 50% and -1 is for 0% of screen
		# vert = [[1,1],[1,-1],[-1,-1],[-1,1]]
		xp = (2-(self.W/self.H))/1
		yp = (2*(self.H/self.W))/1
		#maybe use Pythagoras?
		vert = [[xp, yp],[xp, -yp],[-xp, -yp],[-xp, yp]]

		##::Create  and show the Window
		self.win = visual.Window([self.W,self.H], position, monitor = 'SSVEP Paradigm', color = 'black')
		self.win.flip()

		##::Create Squares
		self.SqP = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.p3S, pos = self.p[2], autoDraw = True)
		self.St0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = self.p[0], autoDraw = True)
		self.St1 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = self.p[1], autoDraw = True)
		self.St2 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = self.p[2], autoDraw = True)
		self.Sq0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = self.p[0], autoDraw = True)
		self.Sq1 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = self.p[1], autoDraw = True)
		self.Sq2 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = self.p[2], autoDraw = True)


	def start(self):
		eachclock = core.MonotonicClock()
		genclock = core.MonotonicClock()
		ISI = core.StaticPeriod(screenHz = 240) #change depending on screen refresh rate
		sq_colors = ['#59FF00', '#B80117']
		
		self.SqP.setOpacity(0)

		self.message = visual.TextStim(self.win, text = 'Flickering routine\n\nReady?')
		self.message.draw()
		self.win.flip()
		core.wait(1.5)

		self.message.setText('3')
		self.message.draw()
		self.win.flip()
		core.wait(1.0)

		self.message.setText('2')
		self.message.draw()
		self.win.flip()
		core.wait(1.0)

		self.message.setText('1')
		self.message.draw()
		self.win.flip()
		core.wait(1.0)

		print("|||\t\t>>>>>Initializing Flickering>>>>>") #print('0')
		# T0 = genclock.getTime()
		for TL in TrialLabels:
			

#-----------##::Showing Cue
			self.message2 = visual.TextStim(self.win, text = 'Focus on the orange:', pos = [0, 0.75])
			if TL == 0:
				self.Sq0.setColor('#FF6C00')
				self.Sq1.setColor('white')
				self.Sq2.setColor('white')
			if TL == 1:
				self.Sq0.setColor('white')
				self.Sq1.setColor('#FF6C00')
				self.Sq2.setColor('white')
			if TL == 2:
				self.Sq0.setColor('white')
				self.Sq1.setColor('white')
				self.Sq2.setColor('#FF6C00')
			self.message2.draw()
			self.win.flip()
			core.wait(3) ## more time from 3 to 5
			HOLD.set()
#-----------##

			##::Setting SSVEP Label
			if TL == 0:
				ES1Label.set()
			elif TL == 1:
				ES2Label.set()
			elif TL == 2:
				ES3Label.set()

			##::Creating P300 sequence
			sequenceP300 = []
			for m in range (0,3):
				for k in range(0,P3NUM): #number of blinks per square
					sequenceP300.append(m)
			random.shuffle(sequenceP300)

			self.Sq0.setColor('white')
			self.Sq1.setColor('white')
			self.Sq2.setColor('white')

			##::Starting Flickering
			# T0_ = eachclock.getTime()
			for sel in sequenceP300:
				COLLECTTRIG.clear() #print('\nc_')
				P300TRIG.set() #print('T')
				self.SqP.setPos(self.p[sel]) #setting position for P300

#---------------##::Setting P300 Labels
				#:Target x Non-Target Label
				if TL == sel:
					EPLabel.set()
				else:
					EPLabel.clear()
				#:Position Label
				if sel == 0:
					EPos1Label.set()
				elif sel == 1:
					EPos2Label.set()
				elif sel == 2:
					EPos3Label.set()
#-------------------
				
				sq0_flag = 1
				sq1_flag = 1
				sq2_flag = 1

				# t0 = eachclock.getTime()
				for cyc in range(1, 61, 1): #half a second (60frames @ 120Hz)

#-------------------##P300 stimuli
					if cyc > 48: #three fourths of cycle
						if not COLLECTTRIG.is_set():
							COLLECTTRIG.set() #print('C')
						self.SqP.setOpacity(0)
					else:
						self.SqP.setOpacity(1)
						pass

					##P300 stimulus: turn on or off per cycle
					# if cyc > 6 and cyc < 48:
					# 	self.SqP.setOpacity(1)
					# else:
					# 	self.SqP.setOpacity(0)
#-----------------------
					
					

#-------------------##SSVEP stimuli
					# if cyc % 4 == 0:							#for 20Hz use % 3 		#for 15Hz use % 4		#for 30Hz use % 2
					# 	sq0_flag = not sq0_flag
					# 	self.Sq0.setColor(sq_colors[sq0_flag])
					# if cyc % 6 == 0:							#for 15Hz use % 4 		#for 10Hz use % 6		#for 15Hz use % 4
					# 	sq1_flag = not sq1_flag
					# 	self.Sq1.setColor(sq_colors[sq1_flag])
					# if cyc % 10 == 0:							#for 12Hz use % 5 		#for 6Hz use % 10		#for 7.5Hz use % 8
					# 	sq2_flag = not sq2_flag
					# 	self.Sq2.setColor(sq_colors[sq2_flag])
#-----------------------###maybe use the same color as the cue?

					self.win.flip()
					ISI.start(0.0081)# ISI.start(0.01245) <<- If stim starts to lag, use this value before 'if cyc > 48', instead
					
					# print(ISI.complete())
					ISI.complete()
				# t1 = eachclock.getTime()
				# print('Half a second? = ',t1-t0,'s')
			# T1_ = eachclock.getTime()
			# print('All P300 =',T1_-T0_,'s\nEach cycle =',(T1_-T0_)/len(sequenceP300))
			HOLD.clear() #print('h')

		# T1 = genclock.getTime()
		# print('||Flickering took', round(T1-T0,3), 's')
		E.set() #print('1')

		self.Sq0.setOpacity(0)
		self.Sq1.setOpacity(0)
		self.Sq2.setOpacity(0)
		self.SqP.setOpacity(0)
		self.message.setText('Experiment is ending...')
		self.message.draw()
		self.win.flip()

	def stop(self):
		self.win.close()
		core.quit()
		# time.sleep(1)

class AcqThread(threading.Thread):
	def __init__(self, dataOutQ1, dataOutQ2, saveQ):
		threading.Thread.__init__(self)
		self.dataOutQ2 = dataOutQ2
		self.dataOutQ1 = dataOutQ1
		self.saveQ = saveQ

	def run(self):
		Acquisition(self.dataOutQ1, self.dataOutQ2, self.saveQ) #place where function is called

class P300Thread(threading.Thread):
	def __init__(self, dataInQ, featureQ):
		threading.Thread.__init__(self)
		self.dataInQ = dataInQ
		self.featureQ = featureQ

	def run(self):
		P300fun(self.dataInQ, self.featureQ)

class SSVEPThread(threading.Thread):
	def __init__(self, dataInQ, featureQ, saveQ):
		threading.Thread.__init__(self)
		self.dataInQ = dataInQ
		self.featureQ = featureQ
		self.saveQ = saveQ

	def run(self):
		SSVEPfun(self.dataInQ, self.featureQ, self.saveQ)

class CCA_Thread(threading.Thread):
	def __init__(self, featureQ, modelQ, classOutQ, dataclassOutQ, mode, saveQ):
		threading.Thread.__init__(self)
		self.featureQ = featureQ
		self.modelQ = modelQ
		self.mode = mode
		self.classOutQ = classOutQ
		self.dataclassOutQ = dataclassOutQ
		self.saveQ = saveQ

	def run(self):
		if self.mode == 1:
			CCA_classifier_train(self.featureQ, self.modelQ, self.saveQ)
		elif self.mode == 2:
			CCA_classifier_predict(self.featureQ, self.modelQ, self.classOutQ, self.dataclassOutQ)

class LDA_Thread(threading.Thread):
	def __init__(self, featureQ, modelQ, classOutQ, dataclassOutQ, mode):
		threading.Thread.__init__(self)
		self.featureQ = featureQ
		self.modelQ = modelQ
		self.mode = mode
		self.classOutQ = classOutQ
		self.dataclassOutQ = dataclassOutQ

	def run(self):
		if self.mode == 1:
			LDA_classifier_train(self.featureQ, self.modelQ)
		elif self.mode == 2:
			LDA_classifier_predict(self.featureQ, self.modelQ, self.classOutQ, self.dataclassOutQ)

class Fusion_Thread(threading.Thread):
	def __init__(self, predict_CCA, predict_LDA, accs_LDA, accs_CCA):
		threading.Thread.__init__(self)
		self.predict_CCA = predict_CCA
		self.predict_LDA = predict_LDA
		self.accs_LDA = accs_LDA
		self.accs_CCA = accs_CCA

	def run(self):
		fusion(predict_LDA, predict_CCA, accs_LDA, accs_CCA)

##FUNCTIONS
def init_dialog():
	global _currDir
	global _partDir
	global _csvpractice
	global _imgpractice
	global _csvsession
	global _imgsession
	global _classifile
	global _classioutCCA
	global _classioutLDA
	global session_mode
	global headset
	global TRIALNUM
	global P3NUM

	_currDir = os.path.dirname(os.path.abspath(__file__))
	session_info = {'ParticipantID':'00', 'SessionNumber':'01', 'Version': 8.0, 'Headset':['','cyton', 'gtec']}

	infoDlg1 = gui.DlgFromDict(dictionary=session_info, 
								title='Hybrid BCI system', 
								fixed=['Version'], 
								order = ['Version','ParticipantID','SessionNumber','Headset'])
	if not infoDlg1.OK:
		print('Operation Cancelled')
		core.quit()
	else:
		headset = session_info['Headset']
		if headset == '':
			print('Headset not selected. Quitting application...')
			core.quit()
		#create dir
		_partDir = os.path.join(_currDir, 'Participants', session_info['ParticipantID'], 'SESS'+session_info['SessionNumber'])
		if not os.path.exists(_partDir):
			os.makedirs(_partDir)

		set_info = {'ParticipantID':session_info['ParticipantID'], 'SessionNumber': session_info['SessionNumber'],
					'SetNum':'01', 'Mode': ['Training', 'Practice', 'Live']}
		infoDlg2 = gui.DlgFromDict(dictionary=set_info, 
									title='Hybrid BCI system', 
									fixed=['ParticipantID', 'SessionNumber'],
									order = ['ParticipantID','SessionNumber','SetNum','Mode'])
		if not infoDlg2.OK:
			print('Operation Cancelled')
			core.quit()
		else:
			if set_info['Mode'] == 'Practice':
				_csvpractice = u'%s_%s_practice.csv' % (session_info['ParticipantID'],session_info['SessionNumber'])
				_imgpractice = u'%s_%s_practice.png' % (session_info['ParticipantID'],session_info['SessionNumber'])
				TRIALNUM = 3
				P3NUM = 7
				session_mode = 0

			else:
				_classifile = u'%s_%s_classifier_models.plk' % (session_info['ParticipantID'],session_info['SessionNumber'])
				_csvsession = u'%s_%s_%s_saveddata.csv' % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])
				_imgsession = u'%s_%s_%s_saveddata.png' % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])

				if set_info['Mode'] == 'Training':
					TRIALNUM = 3
					P3NUM = 7
					session_mode = 1
				elif set_info['Mode'] == 'Live':
					_classioutCCA = u'%s_%s_%s_CCAresults.csv' % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])
					_classioutLDA = u'%s_%s_%s_LDAresults.csv' % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])
					TRIALNUM = 3
					P3NUM = 3
					session_mode = 2

			print('SessMode:',session_mode, '\nHeadset:', headset)

def normalization(matrix):
	## normalization
	mean = np.mean(matrix, 1)
	avgsamples = np.subtract(matrix.transpose(), mean).transpose()
	maxx = np.max(avgsamples, 1)
	minn = np.min(avgsamples, 1)
	try:

		avgsamples = (np.subtract(avgsamples.transpose(), minn)/np.subtract(maxx, minn)).transpose() #(avg-min)/(max-min)
	except RuntimeWarning:
			print('|||||Caution::may have RAILED channel(s)')
	return avgsamples

def Acquisition(dataOutQ1, dataOutQ2, saveQ):	
	'''
	#AVAILABILITY					#IDEAL CHs 				#CYTON 				#GTEC
	#available channels [1 to 8] 	#PO8, PO7, POz, CPz		#P3, Pz, P4, Cz		#C3, Cz, C4, Pz
	#available channels [1 to 8]	#O1, Oz, O2 			#O1, O2 			#O1, O2, Oz
	'''
	if headset == 'cyton':
		chs_p300 = [1,2,3,4]
		chs_ssvep = [5,6]
		ts_idx = 22
	elif headset == 'gtec':
		chs_p300 = [2,3,4,5]
		chs_ssvep = [6,7,8]
		ts_idx = 17

	chs_all = list(set(chs_p300 + chs_ssvep))
	_chs_p300 = list(range(len(chs_p300)))
	# _chs_p300 = [x - 1 for x in chs_p300]
	_chs_ssvep = list(range(_chs_p300[-1]+1, _chs_p300[-1]+1+len(chs_ssvep)))
	# _chs_ssvep = [x - 1 for x in chs_ssvep]

	fs = SPS	#Hz 
	low = 5 / NYQ
	high = 35 / NYQ
	b, a = butter(4, [low, high], btype='band') # b, a = butter(order, [low, high], btype='band')

	thesamples = np.array([])
	thefilterd = np.array([])
	p_avgsamples = np.array([])
	p_filtsamples = np.array([])
	s_avgsamples = np.array([])
	s_filtsamples = np.array([])
	thetimestamps = np.array([])
	tstampsP300 = np.array([])

	i = 0
	start = time.perf_counter() #time.time() #<for older python versions
	while not E.is_set():#while i<1250:
		HOLD.wait() #intro
		P300TRIG.wait(timeout = 10) #function called when P300 starts

		##::Save stimuli beginning timestamp
		current_data = board.get_current_board_data(10)
		ini_stim_ts = current_data[ts_idx,-1]

		if not E.is_set():
			##::Getting Labels
			if EPLabel.is_set():
				ll = 1
			else:
				ll = 0

			if EPos1Label.is_set():
				pl = 0
			elif EPos2Label.is_set():
				pl = 1
			elif EPos3Label.is_set():
				pl = 2

			if ES1Label.is_set():
				TL = [1,0,0]
			elif ES2Label.is_set():
				TL = [0,1,0]
			elif ES3Label.is_set():
				TL = [0,0,1]
			# print('|Labels ll and TL =', ll, TL)

			EPos1Label.clear()
			EPos2Label.clear()
			EPos3Label.clear()
			ES1Label.clear()
			ES2Label.clear()
			ES3Label.clear()

			##::Getting Signal
			COLLECTTRIG.wait() #function called when P300 ends
			P300TRIG.clear() #print('t')
			# print("|Latest samples: ", board.get_board_data_count() ) #last data
			while board.get_board_data_count() < NYQ:
				time.sleep(0.001)
			data = board.get_board_data()

			##::Save stimuli ending timestamp
			end_stim_ts = data[ts_idx,-1]

			ss = np.array(data[chs_all,:])
			tt = np.array(data[ts_idx,:])

			avgsamples = normalization(ss)
			filtsamples = np.array(lfilter(b, a, avgsamples, axis = 1))

			## SAVE Raw data
			if not thesamples.size == 0:
				thesamples = np.hstack((thesamples, avgsamples))
			else:
				thesamples = np.array(avgsamples)

			## SAVE Filtered data
			if not thefilterd.size == 0:
				thefilterd = np.hstack((thefilterd, filtsamples))
			else:
				thefilterd = np.array(filtsamples)
			if not thetimestamps.size == 0:
				thetimestamps = np.append(thetimestamps, tt)
			else:
				thetimestamps = np.array(tt)

			int_TL = int(str(int(TL[0])) + str(int(TL[1])) + str(int(TL[2])), 2)

			if not tstampsP300.size == 0:
				tstampsP300 = np.vstack((tstampsP300, [ini_stim_ts, end_stim_ts, ll, int_TL]))
			else:
				tstampsP300 = np.array([ini_stim_ts, end_stim_ts, ll, int_TL])


			threadLock.acquire()				#queue locked to prevent other threads to access it
			dataOutQ1.put([filtsamples[_chs_p300,-NYQ:], tt[-NYQ:], ll, pl]) 	#data is put in the queue
			dataOutQ2.put([filtsamples[_chs_ssvep,-NYQ:], tt[-NYQ:], TL])		
			# print('|Samples sent: AVG [',avgsamples.shape,'], Filt\'d [',filtsamples.shape,'], tt [',tt.shape,']' ) #for numpy type
			threadLock.release()				#queue unlocked with info

			##::(Re)set Events
			EP300.set() # print('P')
			ESSVEP.set() # print('S')
			i += 1
		else:
			break

	threadLock.acquire()
	saveQ.put([thesamples[:,:], thefilterd[:,:], thetimestamps[:], tstampsP300])
	threadLock.release()
	###
	print('\n\n|||Ending acquisition')
	P300Th.join()
	SSVEPTh.join()
	finish = time.perf_counter()
	# print('___\n|Final Score: Samples [',thesamples.shape,'], Timestamp [',thetimestamps.shape,']') #for numpy type
	print('|Streamed in',round(finish-start,3),'s!\n---')

######################
######################

def P300fun(dataInQ, featureQ):
	labels = []
	ss = np.array([])
	tt = np.array([])
	while not E.is_set():
		EP300.wait(timeout = 10)
		if not E.is_set():
			##::Collect Samples from Queue
			while dataInQ.qsize(): # if not dataqueue.empty():
				try:
					ss, tt, ll, pl = dataInQ.get(0)
					# print('|P300 ss [', ss.shape, ']\n|P300 tt [', tt.shape, ']\n|ll [',ll,']') #for list type
				except Empty:
					return

			##::Average all samples together
			labels.append(ll)
			# with warnings.catch_warnings():
			# 	warnings.simplefilter("ignore", category=RuntimeWarning)
			try:
				mean_ss = np.mean(ss, 0)
			except RuntimeWarning:
				print('<<<Attempting mean with empty dataque',ss)
			# print("\tShape means P300", mean_ss.shape)

			##::Donwsample == Features ~125sample?
			ss_ds = mean_ss[::2]
			tt_ds = tt[::2]

			threadLock.acquire()
			featureQ.put([ss_ds,  ll, pl]) #tt_ds, ll])
			threadLock.release()

			##::END OF PROCESSING
			ELDA.set() #print('LDA')
			EP300.clear() #print('p')
			ss = np.array([])
			tt = np.array([])
		else:
			break
	print('|||P300 labels (ll) {',len(labels),'}')


def SSVEPfun(dataInQ, featureQ, saveQ):
	filter1_set = np.array([])
	filter2_set = np.array([])
	filter3_set = np.array([])
	means_set = np.array([])
	feature_set = np.array([])
	tstamp_set = np.array([])
	label_set = np.array([])

	xf_set = np.array([])
	yf_set = np.array([])
	yff1_set = np.array([])
	yff2_set = np.array([])
	yff3_set = np.array([])

	labels = []
	ss = np.array([])
	tt = np.array([])
	# counter = 0

	# #consider doing 12, 15 and 20 || 6, 10 and 15
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

	while not E.is_set():
		ESSVEP.wait(timeout = 10)
		if not E.is_set():
			##::Collect Samples from Queue
			while dataInQ.qsize():
				try:
					ss, tt, ll = dataInQ.get(0)
					# print('|SSVEP ss [',ss.shape,']\n|SSVEP tt [',tt.shape,']\n|SSVEP ll [',ll,']')
				except Empty:
					return

			##::Average all samples together
			labels.append(ll)
			# with warnings.catch_warnings():
			# 	warnings.simplefilter("ignore", category=RuntimeWarning)

			try:
				mean_ss = np.mean(ss, 0) #add in the future: add weights to different channels
			except RuntimeWarning:
				print('<<<Attempting mean with empty dataque',ss)

			# print("\tShape means SSVEP", mean_ss.shape)
			# filtsamples = np.array(lfilter(b, a, avgsamples, axis = 1))
			f1 = np.array(lfilter(b1, a1, mean_ss, axis = -1))
			f2 = np.array(lfilter(b2, a2, mean_ss, axis = -1))
			f3 = np.array(lfilter(b3, a3, mean_ss, axis = -1))
			# feat = np.sum([[f1],[f2],[f3]], 0)

			# print(f1.shape, f2.shape, f3.shape)
			feat = np.concatenate((f1.T, f2.T, f3.T))
			# print(feat.shape)

			N = int(SPS*0.5)	# N = SAMPLE_RATE * DURATION
			yf = fft(mean_ss)
			xf = fftfreq(N, 1/SPS)
## i = xf >= 0
## xf[i] and yf[i]

			yff1 = fft(f1)
			xff1 = fftfreq(N, 1/SPS)

			yff2 = fft(f2)
			xff2 = fftfreq(N, 1/SPS)

			yff3 = fft(f3)
			xff3 = fftfreq(N, 1/SPS)
			#NOTE: if it's not working, instead of summing, concatenate them, or use [f1, f2 and f3] as a feature matrix
			# print('|SSVEP feat [',feat.shape,']\n|ll [',ll,']')

			# print("Ft. array ", counter," is sparse = ",scipy.sparse.issparse(feat))
			# print("Ft. array ", counter," n. of 0's = ", np.sum(np.array(feat) == 0.))
			# counter += 1

			## SAVING VARIABLES
			if not feature_set.size == 0: feature_set = np.vstack((feature_set, feat))
			else: feature_set = np.array(feat)
			if not means_set.size == 0: means_set = np.vstack((means_set, mean_ss))
			else: means_set = np.array(mean_ss)

			if not filter1_set.size == 0: filter1_set = np.vstack((filter1_set, xff1))
			else: filter1_set = np.array(xff1)
			if not filter2_set.size == 0: filter2_set = np.vstack((filter2_set, xff2))
			else: filter2_set = np.array(xff2)
			if not filter3_set.size == 0: filter3_set = np.vstack((filter3_set, xff3))
			else: filter3_set = np.array(xff3)

			if not yff1_set.size == 0: yff1_set = np.vstack((yff1_set, yff1))
			else: yff1_set = np.array(yff1)
			if not yff2_set.size == 0: yff2_set = np.vstack((yff2_set, yff2))
			else: yff2_set = np.array(yff2)
			if not yff3_set.size == 0: yff3_set = np.vstack((yff3_set, yff3))
			else: yff3_set = np.array(yff3)
 
			if not label_set.size == 0: label_set = np.vstack((label_set, ll))
			else: label_set = np.array(ll)
			if not xf_set.size == 0: xf_set = np.vstack((xf_set, xf))
			else: xf_set = np.array(xf)
			if not yf_set.size == 0: yf_set = np.vstack((yf_set, yf))
			else: yf_set = np.array(yf)

			threadLock.acquire()
			featureQ.put([feat, ll]) #tt, ll])
			saveQ.put([filter1_set, filter2_set, filter3_set, feature_set, means_set, label_set, xf_set, yf_set, yff1_set, yff2_set, yff3_set])
			threadLock.release()

			##::END OF PROCESSING
			ECCA.set() #print('CCA')
			ESSVEP.clear() #print('s')
			ss = np.array([])
			tt = np.array([])
		else:
			break
	print('|||SSVEP labels (TL) {',len(labels),'}')

######################

def b2d(b_vect):
	str_format = str(int(b_vect[0])) + str(int(b_vect[1])) + str(int(b_vect[2]))
	return int(str_format,2)

def CCA2d(Y):
	dec_y = []
	for yy in Y:
		yy = abs(yy)
		marg = np.argmax(yy)
		oarg = [x for x in range(len(yy)) if x != marg]
		yy[marg] = 1
		yy[oarg] = 0
		dec_y.append(b2d(yy))
	return(dec_y)

def CCA2d_single(Y):
	dec_y = []
	oarg = []
	marg = []
	if isinstance(Y, list):
		Y = np.array(Y)
	Y = abs(Y)
	marg = np.argmax(Y)
	oarg = [x for x in range(len(Y)) if x != marg]
	Y[marg] = 1
	Y[oarg] = 0
	dec_y.append(b2d(Y))
	return(dec_y)


def validate_classifier(feature_set, label_set, CCA_mode, splits):
	##Validation
	kf = KFold(n_splits = splits)
	accs = []
	count = 0
	for train_index, test_index in kf.split(feature_set):
		new_y = []
		bin_y_test = []
		rgt = 0
		if CCA_mode:
			print("\n\nTraining fold CCA", count, "\b...")
		else:
			print("\n\nTraining fold LDA", count, "\b...")
		X_train, X_test = feature_set[train_index], feature_set[test_index]
		y_train, y_test = label_set[train_index], label_set[test_index]

		if CCA_mode:
			model = CCA_model.fit(X_train, y_train)
		else:
			model = LDA_model.fit(X_train, y_train)

		y = model.predict(X_test)
		#Conditioning prediction
		if CCA_mode:
			new_y = CCA2d(y)
			#Conditioning labels
			for ll in y_test:
				bin_y_test.append(b2d(ll))
			##Check
			for i in range(len(y)):
				print(y[i], ' vs. ', y_test[i])
		else:
			new_y = y
			bin_y_test = y_test
		##Check
		print(new_y)		#binary predicted
		print(bin_y_test)	#vs. binary test lbls

		#Calculating accuracy
		for i in range(len(new_y)):
			if new_y[i] == bin_y_test[i]:
				rgt += 1
		print(rgt, "labels match")
		accs.append(rgt/len(bin_y_test))
		if CCA_mode:
			print("ACC CCA = ", '{:.3f}'.format(accs[count]*100), "\b%")
		else:
			print("ACC LDA = ", '{:.3f}'.format(accs[count]*100), "\b%")
		count += 1
	accs_mean = np.mean(accs)
	if CCA_mode:
		print("Average ACC CCA = ", '{:.3f}'.format(accs_mean*100), "\b%")
	else:
		print("Average ACC LDA = ", '{:.3f}'.format(accs_mean*100), "\b%")
	return accs, accs_mean

######################

def LDA_classifier_train(featureQ, modelQ):
	feature_set = np.array([])
	label_set = np.array([])
	while not E.is_set():
		ELDA.wait(timeout = 10)
		if not E.is_set():
			while featureQ.qsize():
					try:
						feat, ll, pl = featureQ.get(0)  #pl is not used for training, only for prediction
						# print('|LDA feat [',feat.shape,']\n|ll [',ll,']')
					except Empty:
						return
			if not feature_set.size == 0: feature_set = np.vstack((feature_set, feat))
			else: feature_set = np.array(feat)

			if not label_set.size == 0: label_set = np.hstack((label_set, ll))
			else: label_set = np.array(ll)
			# print('|LDA feat [',feature_set.shape,']\n|ll [',label_set.shape,']')
			ELDA.clear() #print('lda')
		else:
			break
	print('Training LDA...')
	print('|||LDA feature_set [',feature_set.shape,']\n|||LDA label_set [',label_set.shape,']')

	##Validation
	accs, accs_mean = validate_classifier(feature_set, label_set, 0, 10)

	model = LDA_model.fit(feature_set, label_set)
	threadLock.acquire()
	modelQ.put([model, accs_mean])
	threadLock.release()
	print('|||Training finished:', model)


def CCA_classifier_train(featureQ, modelQ, saveQ):
	feature_set = np.array([])
	#
	print_feat1 = np.array([])
	print_feat2 = np.array([])
	print_feat4 = np.array([])
	#
	label_set = np.array([])
	while not E.is_set():
		ECCA.wait(timeout = 10)
		if not E.is_set():
			while featureQ.qsize():
					try:
						feat, ll = featureQ.get(0)
						bin_lbl = str(ll[0]) + str(ll[1]) + str(ll[2])
						# print('|CCA feat [',feat.shape,']\n|ll [',ll,']')
					except Empty:
						return

			#Creating plot array for specific labels
			if int(bin_lbl,2) == 1:
				if not print_feat1.size == 0: print_feat1 = np.vstack((print_feat1, feat))
				else: print_feat1 = np.array(feat)
			elif int(bin_lbl,2) == 2:
				if not print_feat2.size == 0: print_feat2 = np.vstack((print_feat2, feat))
				else: print_feat2 = np.array(feat)
			elif int(bin_lbl,2) == 4:
				if not print_feat4.size == 0: print_feat4 = np.vstack((print_feat4, feat))
				else: print_feat4 = np.array(feat)

			#Stacking feature set for training
			if not feature_set.size == 0: feature_set = np.vstack((feature_set, feat))
			else: feature_set = np.array(feat)

			#Stacking label set for training
			if not label_set.size == 0: label_set = np.vstack((label_set, ll))
			else: label_set = np.array(ll)

			# print('|CCA feat [',feature_set.shape,']\n|ll [',label_set.shape,']')

			ECCA.clear() #print('cca')
		else:
			break

	# ## Testing for Sparsity
	# print("Full ft. array is sparse = ",scipy.sparse.issparse(feature_set))
	# print("Full lb. array is sparse = ",scipy.sparse.issparse(label_set))
	# print("Total Zeros Ft. = ", np.sum(np.array(feature_set) == 0.))
	# print("Total Zeros Lb. = ", np.sum(np.array(label_set) == 0.))

	threadLock.acquire()
	saveQ.put([print_feat1, print_feat2, print_feat4])
	threadLock.release()

	print('Training CCA...')
	print('|||CCA feature_set [',feature_set.shape,']\n|||CCA label_set [',label_set.shape,']')

	##Validation
	accs, accs_mean = validate_classifier(feature_set, label_set, 1, 10)

	model = CCA_model.fit(feature_set, label_set)
	threadLock.acquire()
	modelQ.put([model, accs_mean])
	threadLock.release()
	print('|||Training finished:', model)

######################

def LDA_classifier_predict(featureQ, modelQ, classOutQ, dataclassOutQ):
	while not E.is_set():
		ELDA.wait(timeout = 10)
		if not E.is_set():
			while featureQ.qsize():
					try:
						feat, ll, pl = featureQ.get(0)
						# print('|LDA feat [',feat.shape,']\n|ll [',ll,']')
					except Empty:
						return
			feat = feat.reshape(1,-1)
			y = modelQ.predict(feat)
			print('|||||LDA label', ll, 'vs.', y,' LDA predicted')
			threadLock.acquire()
			dataclassOutQ.put([feat, ll, pl, y])
			threadLock.release()
			ELDA.clear() #print('lda')
		else:
			break
	print('||End of LDA prediction...')


def CCA_classifier_predict(featureQ, modelQ, classOutQ, dataclassOutQ):
	feature_set = np.array([])
	label_set = np.array([])
	y_set = np.array([])
	while not E.is_set():
		ECCA.wait(timeout = 10)
		if not E.is_set():
			while featureQ.qsize():
					try:
						feat, ll = featureQ.get(0)
						# print('|CCA feat [',feat.shape,']\n|ll [',ll,']')
					except Empty:
						return
			y = modelQ.predict(feat)
			print('|||||CCA label', ll, 'vs.', y, ' CCA predicted')
			threadLock.acquire()
			dataclassOutQ.put([feat, ll, y])
			threadLock.release()
			ECCA.clear() #print('cca')
		else:
			break
	print('||End of CCA prediction...')

######################

def fusion(Data_LDAQ, Data_CCAQ, accs_LDA, accs_CCA):
	feature_set = np.array([])
	label_set = np.array([])
	y_set = np.array([])
	una_pred = np.zeros((3,1))
	lda_pred = np.zeros((3,1))
	cca_pred = np.zeros((3,1))
	tot_pred = np.zeros((3,1))
	pred_counter = 0
	answer = []


	while not E.is_set():
		ELDAFUS.wait(timeout = 10)
		ECCAFUS.wait(timeout = 10)
		if not E.is_set():
			while Data_LDAQ.qsize():
					try: #feat, ll, pl, y
						feat_LDA, ll_LDA, pl_LDA, y_LDA= Data_LDAQ.get(0)
					except Empty:
						return

			while Data_CCAQ.qsize():
					try: #feat, ll, y
						feat_CCA, ll_CCA, y_CCA = Data_CCAQ.get(0)
					except Empty:
						return

			##::Decimal Transformation (remember, it will be a single value)
			y_CCA_d = CCA2d_single(y_CCA) #decimal output
			y_CCA_d = [x-2 if x == 4 else x-1 for x in y_CCA_d] #setting values to 0, 1, 2 instead of 1, 2, 4
			yy_CCA = y_CCA_d[0]

			#Stack features and Labels
			if not feature_set_LDA.size == 0:
				feature_set_LDA = np.vstack((feature_set_LDA, feat_LDA))
			else:
				feature_set_LDA = np.array(feat_LDA)
			if not label_set_LDA.size == 0:
				label_set_LDA = np.vstack((label_set_LDA, ll_LDA))
			else:
				label_set_LDA = np.array(ll_LDA)

			if not feature_set_CCA.size == 0:
				feature_set_CCA = np.vstack((feature_set_CCA, feat_CCA))
			else:
				feature_set_CCA = np.array(feat_CCA)
			if not label_set_CCA.size == 0:
				label_set_CCA = np.vstack((label_set_CCA, ll_CCA))
			else:
				label_set_CCA = np.array(ll_CCA)


			if y_LDA == 1:
				if pl_LDA == yy_CCA: #if position and CCA selection are the same
					una_pred[yy_CCA] += 1
				else:
					lda_pred[yy_CCA] += 1*accs_LDA
					cca_pred[yy_CCA] += 1*accs_CCA
			elif y_LDA == 0:
				lda_pred[pl_LDA] -= 0.5*accs_LDA
				cca_pred[yy_CCA] += 0.5*accs_CCA
			pred_counter += 1
			##create limit of repetitions

			tot_pred = np.sum([una_pred, lda_pred, cca_pred], 0)
			if any(tot_pred[tot_pred >= 3]):
				I = [i for i in range(len(tot_pred)) if tot_pred[i] >= 3]
				max_tot_pred = tot_pred[I]
				I_max = np.argmax(max_tot_pred)
				answer = max_tot_pred[I_max]
				print("Final Answer = {}\nRequired votes = {}".format(answer, pred_counter))
				pred_counter = 0
			##::Pass answer as feedback to screen
			##::Save final answer and counter


##EXECUTION
if __name__ == '__main__':
	init_dialog()

	print('::Session parameters::')
	print(u'[Current path][%s]\n[Partic path][%s]\n[CSV filename][%s]\n[Clas. filename][%s]\n[Session Mode][%s]\n[Num Trials][%d]\n[Num Stim][%d]\n[Headset][%s]' 
			% (_currDir, _partDir, _csvsession, _classifile, session_mode, TRIALNUM, P3NUM, headset))

	classifile_path__ = os.path.join(_partDir, _classifile)
	if session_mode	== 1:
		if os.path.isfile(classifile_path__):
			warningDlg = gui.Dlg(title = "File name already exists", labelButtonOK = 'Yes', labelButtonCancel = 'Cancel')
			warningDlg.addText('Another classification model file already exists in this folder.')
			warningDlg.addText('Do you want to rewrite with new file?')
			ok_data = warningDlg.show()
			if not warningDlg.OK:
				core.quit()

	##::Definitions
	LDA_model = LDA(solver = 'svd')
	CCA_model = CCA(n_components = 2, max_iter=20000)
	#
	threadLock = threading.Lock()
	SaveQ = queue.Queue()
	SaveFeatQ = queue.Queue()
	SaveSSVEPQ = queue.Queue()
	DataOutQ1 = queue.Queue()
	DataOutQ2 = queue.Queue()
	FeatQ1 = queue.Queue()
	FeatQ2 = queue.Queue()
	ModelQ1 = queue.Queue()
	ModelQ2 = queue.Queue()
	ClassOutQ1 = queue.Queue()
	ClassOutQ2 = queue.Queue()
	DataClassOutQ1 = queue.Queue()
	DataClassOutQ2 = queue.Queue()
	#
	E = threading.Event()
	HOLD = threading.Event()
	P300TRIG = threading.Event()
	COLLECTTRIG = threading.Event()
	EP300 = threading.Event()
	ESSVEP = threading.Event()
	#
	ELDA = threading.Event()
	ECCA = threading.Event()
	ELDAFUS = threading.Event()
	ECCAFUS = threading.Event()
	#
	EPLabel = threading.Event()
	EPos1Label = threading.Event()
	EPos2Label = threading.Event()
	EPos3Label = threading.Event()
	ES1Label = threading.Event()
	ES2Label = threading.Event()
	ES3Label = threading.Event()
	lockcounter = False
	#
	AcqTh = AcqThread(DataOutQ1, DataOutQ2, SaveQ)
	P300Th = P300Thread(DataOutQ1, FeatQ1)
	SSVEPTh = SSVEPThread(DataOutQ2, FeatQ2, SaveSSVEPQ)
	#
	if session_mode == 1:
		LDATh = LDA_Thread(FeatQ1, ModelQ1, ClassOutQ1, DataClassOutQ1, session_mode)
		CCATh = CCA_Thread(FeatQ2, ModelQ2, ClassOutQ2, DataClassOutQ2, session_mode, SaveFeatQ)
	elif session_mode == 2:
		classifile_path__ = os.path.join(_partDir, _classifile)
		if os.path.isfile(classifile_path__):
			with open(classifile_path__, 'rb') as input:
				cca_model, lda_model, accs_CCA, accs_LDA = pickle.load(input)
			LDATh = LDA_Thread(FeatQ1, lda_model, ClassOutQ1, DataClassOutQ1, session_mode)
			CCATh = CCA_Thread(FeatQ2, cca_model, ClassOutQ2, DataClassOutQ2, session_mode, SaveFeatQ)
			FusTh = Fusion_Thread(DataClassOutQ1, DataClassOutQ2, accs_LDA, accs_CCA)
##INCLUDE FUSION THREAD
		else:
			notFoundDlg = gui.Dlg(title = "Classifile not found", labelButtonOK = 'Ok', labelButtonCancel = 'Well, ok')
			notFoundDlg.addText('There is no classifile for this Participant in this session.')
			notFoundDlg.addText('Please, run a Training Set first.')
			ok_data = notFoundDlg.show()
			if notFoundDlg.OK or not notFoundDlg.OK:
				core.quit()


	##::True Labels for the trials
	TrialLabels = []
	for n in range (0,3):
		for k in range(0,TRIALNUM): ##INCREASE TO 7 #number of trials per square (total trials = n*k = 21)
			TrialLabels.append(n)
	random.shuffle(TrialLabels)
	print("\n|||||||||Trial labels Sequence {",TrialLabels,"}\n")

	# x = (1920/4)		#=480
	# y = (720/4)		#=180
	FlkWin = FlickeringBoxes([0,50])

	##::CONNECTION
	BoardShim.enable_dev_board_logger()
	params = BrainFlowInputParams ()
#---## BOARDS
	if headset == 'cyton':
	#CYTON
		params.serial_port = 'COM5'#'COM3' #'COM11'
		board_id = BoardIds.CYTON_BOARD.value
	elif headset == 'gtec':
	#GTEC
		board_id = BoardIds.UNICORN_BOARD.value
		params.serial_number = 'UN-2019.05.08'
#---
	board = BoardShim (board_id, params)
	print('|Looking for an EEG stream...')
	SPS = board.get_sampling_rate(board_id)
	print('|Sample Rate =', SPS)
	NYQ = int(SPS*0.5)
	board.prepare_session() #connect to board
	print('|Connected\n--\n')
	if session_mode == 0:
		csvfile_path__ = os.path.join(_partDir, _csvpractice)
		imagefile_path__ = os.path.join(_partDir, _imgpractice)
	else:
		csvfile_path__ = os.path.join(_partDir, _csvsession)
		imagefile_path__ = os.path.join(_partDir, _imgsession)
		if session_mode ==2:
			classioutCCA_path__ = os.path.join(_partDir, _classioutCCA)
			classioutLDA_path__ = os.path.join(_partDir, _classioutLDA)
	board.start_stream(45000, 'file://'+csvfile_path__+':w') #streams data and saves it to csv file

	time.sleep(5)

	print('|||Starting...')

	##::START FLICKER AND ACQUISITION
	AcqTh.start()
	P300Th.start()
	SSVEPTh.start()
	CCATh.start()
	LDATh.start()
	if session_mode == 2: FusTh.start()

	FlkWin.start()

	AcqTh.join()
	CCATh.join()
	LDATh.join()
	if session_mode == 2: FusTh.join()

	board.stop_stream()
	board.release_session()

	if SaveQ.qsize():
		while SaveQ.qsize():
			try:
				y, yf, x, tss = SaveQ.get(0) # [thesamples[:,:], thefilterd[:,:], thetimestamps[:], tstampsP300]

			except Empty:
				pass

		##Saving P300 timestamps
		#Read Excel file as a DataFrame
		df = pd.read_csv(csvfile_path__)

		# print(df.head(5))

		#Atributte names to columns
		if headset == 'cyton':
			df.columns = ['Samples', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'AccelX', 'AccelY', 'AccelZ', 
			'Empty0', 'Empty1', 'Empty2', 'Empty3', 'Empty4', 'Empty5', 'Empty6', 'Empty7', 'Empty8', 'Empty9', 'Timestamps', 'Empty10', 'Empty11']
		elif headset == 'gtec':
						# 'Fz', 'C3', 'Cz', 'C4', 'Pz', 'O1', 'Oz', 'O2'
			df.columns = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'AccelX', 'AccelY', 'AccelZ', 
			'Empty0', 'Empty1', 'Empty2', 'Empty3', 'Samples', 'Empty4', 'Timestamps', 'Empty6', 'Empty']
		#Zero P300_ts column
		df['P300_ts'] = -1
		df['P300_labels'] = -1
		df['SSVEP_labels'] = -1
		#Find columns with timestamps
		for cl in df['Timestamps']:
			#tss => [ini_stim_ts, end_stim_ts, ll, int_TL]
			for t0, t1, ll, TL in zip(tss[:,0], tss[:,1], tss[:,2], tss[:,3]):
				if '{:.6f}'.format(cl) == '{:.6f}'.format(t0):
					df.loc[df['Timestamps'] == cl, 'P300_ts'] = 0
					df.loc[df['Timestamps'] == cl, 'P300_labels'] = ll
					df.loc[df['Timestamps'] == cl, 'SSVEP_labels'] = TL
					# print(cl, t0)

				if '{:.6f}'.format(cl) == '{:.6f}'.format(t1):
					df.loc[df['Timestamps'] == cl, 'P300_ts'] = 2
					# print(cl, t1)

		# print(">>>This is Timestamps float\n",df['Timestamps'])
		# df['Timestamps'] = df['Timestamps'].astype(str)
		# print(">>>This is Timestamps string\n",df['Timestamps'])

		# #Find columns with initial timestamps
		# for ts in tss[:,0]:
		# 	# df.replace({'A': {0: 100, 4: 400}}
		# 	print(round(ts,6))
		# 	# df.loc[df['Timestamps'] == ts, 'P300_ts'] = 1
		# 	# df.loc[np.isclose(df['Timestamps'], ts, rtol=1e-06), 'P300_ts'] = 1
		# 	df.loc[df['Timestamps'] == "{:.6f}".format(ts),'P300_ts'] = 1

		# #Find columns with ending timestamps
		# for ts in tss[:,1]:
		# 	# print(df.loc[df['Timestamps'] == ts])
		# 	print(round(ts,6))
		# 	# df.loc[df['Timestamps'] == ts, 'P300_ts'] = 2
		# 	# df.loc[np.isclose(df['Timestamps'], ts, rtol=1e-06), 'P300_ts'] = 2
		# 	df.loc[df['Timestamps'] == "{:.6f}".format(ts),'P300_ts'] = 2

		# #To save it back to csv
		df.to_csv(csvfile_path__)

	if session_mode == 1:
		print("Finishing training session")
		flag1, flag2 = 1, 1
		if ModelQ1.qsize():
			while ModelQ1.qsize():
				try:
					lda_model, accs_LDA = ModelQ1.get(0)
					print("LDA Model - OK")
				except Empty:
					print("[[ERROR>>>]] LDA Model empty")
					flag1 = 0
				except Exception as ex:
					template = "An exception of type {0} occurred. Arguments:\n{1!r}"
					message = template.format(type(ex).__name__, ex.args)
					print(message)
					print("[[ERROR>>>]] LDA Model not found")
					flag1 = 0
		else:
			print("[[ERROR>>>]] LDA Model empty")
			flag1 = 0

		if ModelQ2.qsize():
			while ModelQ2.qsize():
				try:
					cca_model, accs_CCA = ModelQ2.get(0)
					print("CCA Model - OK")
				except Empty:
					print("[[ERROR>>>]] CCA Model empty")
					flag2 = 0
				except Exception as ex:
					template = "An exception of type {0} occurred. Arguments:\n{1!r}"
					message = template.format(type(ex).__name__, ex.args)
					print(message)
					print("[[ERROR>>>]] CCA Model not found")
					flag2 = 0
		else:
			print("[[ERROR>>>]] CCA Model empty")
			flag2 = 0

		##SAVE classifier models with pickle
		if flag1 and flag2:
			with open(classifile_path__, 'wb') as output:
				pickle.dump([cca_model, lda_model, accs_CCA, accs_LDA], output, pickle.HIGHEST_PROTOCOL)
				print("Models saved - OK")
		else:
			print("[[ERROR>>>]] Models not saved")
	
	elif session_mode == 2:
		if DataClassOutQ1.qsize():
			while DataClassOutQ1.qsize():
				try:
					feature_set, label_set, y_set = DataClassOutQ1.get(0)
				except Empty:
					pass
			print('LDA outputs: Feat_set', feature_set.shape, 'Label_set:', label_set.shape, 'y_set:', len(y_set) )
			#### May need stacking adjustments before saving
			# with open(classioutLDA_path__, 'w', newline='') as file:
			# 	writer = csv.writer(classioutLDA_path__)
			# 	writer.writerows([label_set, y_set, feature_set])

		if DataClassOutQ2.qsize():
			while DataClassOutQ2.qsize():
				try:
					feature_set, label_set, y_set = DataClassOutQ2.get(0)
				except Empty:
					pass
			print('CCA outputs: Feat_set', feature_set.shape, 'Label_set:', label_set.shape, 'y_set:', len(y_set) )
			#### May need stacking adjustments before saving
			# with open(classioutCCA_path__, 'w', newline='') as file:
			# 	writer = csv.writer(classioutCCA_path__)
			# 	writer.writerows([label_set, y_set, feature_set])


	##Plotting the normalized and filtered data
	plt.plot(x,y.transpose(),'k')
	plt.plot(x,yf.transpose(),'b')
	plt.savefig(imagefile_path__)
	# plt.show()

	# #Plotting Features
	# if SaveFeatQ.qsize():
	# 	while SaveFeatQ.qsize():
	# 		try:
	# 			x1, x2, x4 = SaveFeatQ.get(0)
	# 		except Empty:
	# 			pass

	# 	for i in range(len(x1)):
	# 		#Comparison of features for different frqs.
	# 		plt.plot(x1[i],'r')
	# 		plt.plot(x2[i],'b')
	# 		plt.plot(x4[i],'g')
	# 		plt.show()
	# 	# plt.savefig("_Features_"+imagefile_path__)


	# ##Plotting Individual CCA filters and filters sum, plus average
	# if SaveSSVEPQ.qsize():
	# 	while SaveSSVEPQ.qsize():
	# 		try:
	# 			f1, f2, f3, fff, mean_ss, ll, xf, yf, y1, y2, y3 = SaveSSVEPQ.get(0)
	# 		except Empty:
	# 			pass
	# 	for i in range(len(mean_ss)):
	# 		bin_lbl = str(ll[i][0]) + str(ll[i][1]) + str(ll[i][2]) ##SOLVE HOW labels are being concat, so it can be read like this
	# 		# plt.plot(f1[i],'r')
	# 		# plt.plot(f2[i],'b')
	# 		# plt.plot(f3[i],'g')
	# 		# plt.plot(fff[i],'k')
	# 		# plt.plot(mean_ss[i], 'gray') ##CHECK WHAT MEAN IS DOING TO THE SIGNAL
	# 		# plt.title("Label: %s" %bin_lbl)
	# 		# plt.show()

	# 		plt.plot(f1[i], np.abs(y1[i]), 'r') #6hz
	# 		plt.plot(f2[i], np.abs(y2[i]), 'b') #10Hz
	# 		plt.plot(f3[i], np.abs(y3[i]), 'g') #15Hz
	# 		plt.plot(xf[i], np.abs(yf[i]), 'k')
	# 		plt.title("Label: %s" %bin_lbl)
	# 		plt.show()


	print("\n\n|||||DONE|||||\n\n")
	FlkWin.stop()
