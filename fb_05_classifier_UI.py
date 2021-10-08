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
import numpy as np
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
from scipy.signal import butter, lfilter
import cProfile, pstats, io
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes#, AggOperation
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import warnings
warnings.filterwarnings("error")

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
class FlickeringBoxes:
	def __init__(self, position):
		#Definitions
		HOLD.clear() #print('h')
		self.H = 800 #window hight (to fit side of the window)
		self.W = 800 #window width

		self.sS = 0.125			#square size
		self.pS = self.sS*1.35	#padding size
		self.p3S = self.sS*1.75	#p300 size

		self.p =[[-0.5, -0.5],[0, 0.5],[0.5, -0.5]] #from the centre; 1 is for 100%, 0 is for 50% and -1 is for 0% of screen
		vert = [[1,1],[1,-1],[-1,-1],[-1,1]]

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
		# T0 = time.time()
		T0 = genclock.getTime()
		counter = 0
		for TL in TrialLabels:
			##::Showing Cue
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
			core.wait(3)
			HOLD.set() #print('H')

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
				for k in range(0,P3NUM): # INCREASE TO 7 #number of blinks per square
					sequenceP300.append(m)
			random.shuffle(sequenceP300)
			# print("\n|||||||||P300 labels 4 TL ",TL," {",sequenceP300,"}\n")

			##::Starting Flickering
			for sel in sequenceP300:
				COLLECTTRIG.clear() #print('\nc_')
				P300TRIG.set() #print('T')
				self.SqP.setPos(self.p[sel])

				if TL == sel:
					EPLabel.set()
				else:
					EPLabel.clear()
				# print('\n\n|Exp: -- P', EPLabel.is_set(),' -- S1', ES1Label.is_set(),' -- S2', ES2Label.is_set(),' -- S3', ES3Label.is_set())
				
				for cyc in range(4):
					for stg in range(8):

						t0 = eachclock.getTime()
						ISI.start(0.02)

						if cyc > 2: #fourth cycle
							if not COLLECTTRIG.is_set():
								counter += 1
								# print(counter ,'th sending')
								COLLECTTRIG.set() #print('C')
							self.SqP.setOpacity(0)
						else:
							self.SqP.setOpacity(1)

						if stg == 0:
							self.Sq0.setColor('#00388A') #0
							self.Sq1.setColor('#59FF00') #0
							self.Sq2.setColor('#59FF00') #0
						elif stg == 1:
							self.Sq0.setColor('#59FF00') #1
							self.Sq1.setColor('#59FF00') #0
							self.Sq2.setColor('#59FF00') #0
						elif stg == 2:
							self.Sq0.setColor('#00388A') #0
							self.Sq1.setColor('#B80117') #1
							self.Sq2.setColor('#59FF00') #0
						elif stg == 3:
							self.Sq0.setColor('#59FF00') #1
							self.Sq1.setColor('#B80117') #1
							self.Sq2.setColor('#59FF00') #0
						elif stg == 4:
							self.Sq0.setColor('#00388A') #0
							self.Sq1.setColor('#59FF00') #0
							self.Sq2.setColor('#B80117') #1
						elif stg == 5:
							self.Sq0.setColor('#59FF00') #1
							self.Sq1.setColor('#59FF00') #0
							self.Sq2.setColor('#B80117') #1
						elif stg == 6:
							self.Sq0.setColor('#00388A') #0
							self.Sq1.setColor('#B80117') #1
							self.Sq2.setColor('#B80117') #1
						elif stg == 7:
							self.Sq0.setColor('#59FF00') #1
							self.Sq1.setColor('#B80117') #1
							self.Sq2.setColor('#B80117') #1
						self.win.flip()
						# print("Step ", stg, "of ", cyc)
						t1 = eachclock.getTime()
						# print('One show = ',t1-t0,'s')
						# print(ISI.complete())
						ISI.complete()
			HOLD.clear() #print('h')			

		T1 = genclock.getTime()
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
	def __init__(self, dataInQ, featureQ):
		threading.Thread.__init__(self)
		self.dataInQ = dataInQ
		self.featureQ = featureQ

	def run(self):
		SSVEPfun(self.dataInQ, self.featureQ)

class CCA_Thread(threading.Thread):
	def __init__(self, featureQ, modelQ, classOutQ, dataclassOutQ, mode):
		threading.Thread.__init__(self)
		self.featureQ = featureQ
		self.modelQ = modelQ
		self.mode = mode
		self.classOutQ = classOutQ
		self.dataclassOutQ = dataclassOutQ

	def run(self):
		if self.mode == 1:
			CCA_classifier_train(self.featureQ, self.modelQ)
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
	global TRIALNUM
	global P3NUM

	_currDir = os.path.dirname(os.path.abspath(__file__))
	session_info = {'ParticipantID':'00', 'SessionNumber':'01', 'Version': 5.0}

	infoDlg1 = gui.DlgFromDict(dictionary=session_info, 
								title='Hybrid BCI system', 
								fixed=['Version'], 
								order = ['Version','ParticipantID','SessionNumber'])
	if not infoDlg1.OK:
		print('Operation Cancelled')
		core.quit()
	else:
		#create dir
		_partDir = os.path.join(_currDir, 'Participants', session_info['ParticipantID'], 'SESS'+session_info['SessionNumber'])
		if not os.path.exists(_partDir):
			os.makedirs(_partDir)

		set_info = {'ParticipantID':session_info['ParticipantID'], 'SessionNumber': session_info['SessionNumber'],
					'SetNum':'01', 'Mode': ['Practice','Training', 'Live']}
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
				TRIALNUM = 2
				P3NUM = 5
				session_mode = 0

			else:
				_classifile = u'%s_%s_classifer_models.plk' % (session_info['ParticipantID'],session_info['SessionNumber'])
				_csvsession = u'%s_%s_%s_saveddata.csv' % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])
				_imgsession = u'%s_%s_%s_saveddata.png' % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])

				if set_info['Mode'] == 'Training':
					TRIALNUM = 2
					P3NUM = 2
					session_mode = 1
				elif set_info['Mode'] == 'Live':
					_classioutCCA = u'%s_%s_%s_CCAresults.csv' % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])
					_classioutLDA = u'%s_%s_%s_LDAresults.csv' % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])
					TRIALNUM = 3#7
					P3NUM = 3#7
					session_mode = 2

			print('SessMode:',session_mode)

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

# @profile
def Acquisition(dataOutQ1, dataOutQ2, saveQ):
	chs_p300 = [1]	#available channels [1 to 8]
	chs_ssvep = [1]	#available channels [1 to 8]
	chs_all = list(set(chs_p300 + chs_ssvep))
	_chs_p300 = [x - 1 for x in chs_p300]
	_chs_ssvep = [x - 1 for x in chs_ssvep]
	# print('||||||||All channels =', chs_all)
	fs = 250			#Hz 
	nyq = 0.5*fs
	low = 5 / nyq
	high = 50 / nyq
	b, a = butter(4, [low, high], btype='band') # b, a = butter(order, [low, high], btype='band')

	rawsamples = np.array([])
	p_avgsamples = np.array([])
	p_filtsamples = np.array([])
	s_avgsamples = np.array([])
	s_filtsamples = np.array([])
	rawtimestamps = np.array([])

	i = 0
	start = time.perf_counter() #time.time() #<for older python versions
	while not E.is_set():#while i<1250:
		HOLD.wait()
		P300TRIG.wait(timeout = 10) #function called when
		if not E.is_set():
			##::Getting Labels
			# print('|Acq: -- P', EPLabel.is_set(),' -- S1', ES1Label.is_set(),' -- S2', ES2Label.is_set(),' -- S3', ES3Label.is_set())
			if EPLabel.is_set():
				ll = 1
			else:
				ll = 0

			if ES1Label.is_set():
				TL = 0
			elif ES2Label.is_set():
				TL = 1
			elif ES3Label.is_set():
				TL = 2
			# print('|Labels ll and TL =', ll, TL)
			ES1Label.clear()
			ES2Label.clear()
			ES3Label.clear()

			##::Getting Signal
			COLLECTTRIG.wait()
			P300TRIG.clear() #print('t')
			# print("|Latest samples: ", board.get_board_data_count() ) #last data
			while board.get_board_data_count() < 125:
				time.sleep(0.001)
			data = board.get_board_data()
			ss = np.array(data[chs_all,:])
			tt = np.array(data[22,:])

			if not rawsamples.size == 0:
				rawsamples = np.hstack((rawsamples, ss))
			else:
				rawsamples = np.array(ss)
			if not rawtimestamps.size == 0:
				rawtimestamps = np.append(rawtimestamps, tt)
			else:
				rawtimestamps = np.array(tt)

			avgsamples = normalization(ss)
			filtsamples = np.array(lfilter(b, a, avgsamples, axis = 1))

			threadLock.acquire()				#queue locked to prevent other threads to access it
			dataOutQ1.put([filtsamples[_chs_p300,-125:], tt[-125:], ll]) 	#data is put in the queue
			dataOutQ2.put([filtsamples[_chs_ssvep,-125:], tt[-125:], TL]) 	#data is put in the queue
			# print('|Samples sent: AVG [',avgsamples.shape,'], Filt\'d [',filtsamples.shape,'], tt [',tt.shape,']' ) #for numpy type
			threadLock.release()				#queue unlocked with info\

			##::(Re)set Events
			EP300.set() # print('P')
			ESSVEP.set() # print('S')
			i += 1
		else:
			break

	threadLock.acquire()
	saveQ.put([rawsamples[:,:], rawtimestamps[:]])
	threadLock.release()
	###
	print('\n\n|||Ending acquisition')
	P300Th.join()
	SSVEPTh.join()
	finish = time.perf_counter()
	# print('___\n|Final Score: Samples [',rawsamples.shape,'], Timestamp [',rawtimestamps.shape,']') #for numpy type
	print('|Streamed in',round(finish-start,3),'s!\n---')


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
					ss, tt, ll = dataInQ.get(0)
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
				print('<<<Attempting mean with empty dataqueue - P300',ss)
			# print("\tShape means P300", mean_ss.shape)

			##::Donwsample == Features ~125sample?
			ss_ds = mean_ss[::2]
			tt_ds = tt[::2]

			threadLock.acquire()
			featureQ.put([ss_ds,  ll]) #tt_ds, ll])
			threadLock.release()

			##::END OF PROCESSING
			ELDA.set() #print('LDA')
			EP300.clear() #print('p')
			ss = np.array([])
			tt = np.array([])
		else:
			break
	print('|||P300 labels (ll) {',len(labels),'}')


def SSVEPfun(dataInQ, featureQ):
	labels = []
	ss = np.array([])
	tt = np.array([])
	fs = 250			#Hz
	nyq = 0.5*fs

	low1 = 5 / nyq
	high1 = 7 / nyq
	#
	low2 = 14 / nyq
	high2 = 16 / nyq
	#
	low3 = 29 / nyq
	high3 = 31 / nyq

	b1, a1 = butter(4, [low1, high1], btype='band')
	b2, a2 = butter(4, [low2, high2], btype='band')
	b3, a3 = butter(4, [low3, high3], btype='band')

	while not E.is_set():
		ESSVEP.wait(timeout = 10)
		if not E.is_set():
			##::Collect Samples from Queue
			while dataInQ.qsize():
				try:
					ss, tt, ll = dataInQ.get(0)
					# print('|SSVEP ss [',ss.shape,']\n|SSVEP tt [',tt.shape,']\n|ll [',ll,']')
				except Empty:
					return

			##::Average all samples together
			labels.append(ll)
			# with warnings.catch_warnings():
			# 	warnings.simplefilter("ignore", category=RuntimeWarning)
			try:
				mean_ss = np.mean(ss, 0)
			except RuntimeWarning:
				print('<<<Attempting mean with empty dataqueue - CCA',ss)
			# print("\tShape means SSVEP", mean_ss.shape)
			f1 = np.array(lfilter(b1, a1, mean_ss))
			f2 = np.array(lfilter(b2, a2, mean_ss))
			f3 = np.array(lfilter(b3, a3, mean_ss))
			feat = np.sum([[f1],[f2],[f3]],0)
			# print('|SSVEP feat [',feat.shape,']\n|ll [',ll,']')
			threadLock.acquire()
			featureQ.put([feat,  ll]) #tt, ll])
			threadLock.release()

			##::END OF PROCESSING
			ECCA.set() #print('CCA')
			ESSVEP.clear() #print('s')
			ss = np.array([])
			tt = np.array([])
		else:
			break
	print('|||SSVEP labels (TL) {',len(labels),'}')

def LDA_classifier_train(featureQ, modelQ):
	feature_set = np.array([])
	label_set = np.array([])
	while not E.is_set():
		ELDA.wait(timeout = 10)
		if not E.is_set():
			while featureQ.qsize():
					try:
						feat, ll = featureQ.get(0)
						# print('|LDA feat [',feat.shape,']\n|ll [',ll,']')
					except Empty:
						return
			if not feature_set.size == 0:
				feature_set = np.vstack((feature_set, feat))
			else:
				feature_set = np.array(feat)
			if not label_set.size == 0:
				label_set = np.hstack((label_set, ll))
			else:
				label_set = np.array(ll)
			# print('|LDA feat [',feature_set.shape,']\n|ll [',label_set.shape,']')
			ELDA.clear() #print('lda')
		else:
			break
	print('Training LDA...')
	print('|||LDA feature_set [',feature_set.shape,']\n|||LDA label_set [',len(label_set),']')
	model = LDA_model.fit(feature_set, label_set)
	threadLock.acquire()
	modelQ.put(model) #tt, ll])
	threadLock.release()
	print('|||Training finished:', model)


def CCA_classifier_train(featureQ, modelQ):
	feature_set = np.array([])
	label_set = np.array([])
	while not E.is_set():
		ECCA.wait(timeout = 10)
		if not E.is_set():
			while featureQ.qsize():
					try:
						feat, ll = featureQ.get(0)
						# print('|CCA feat [',feat.shape,']\n|ll [',ll,']')
					except Empty:
						return
			if not feature_set.size == 0:
				feature_set = np.vstack((feature_set, feat))
			else:
				feature_set = np.array(feat)
			if not label_set.size == 0:
				label_set = np.vstack((label_set, ll))
			else:
				label_set = np.array(ll)
			# print('|CCA feat [',feature_set.shape,']\n|ll [',label_set.shape,']')
			ECCA.clear() #print('cca')
		else:
			break
	print('Training CCA...')
	print('|||CCA feature_set [',feature_set.shape,']\n|||CCA label_set [',label_set.shape,']')
	##Testing if NaN exists:
	array_sum = np.sum(feature_set)
	array_has_nan = np.isnan(array_sum)
	print("\t\t\tFeatures array has NaN:", array_has_nan)
	model = CCA_model.fit(feature_set, label_set)
	threadLock.acquire()
	modelQ.put(model) #tt, ll])
	threadLock.release()
	print('|||Training finished:', model)

######################

def LDA_classifier_predict(featureQ, modelQ, classOutQ, dataclassOutQ):
	feature_set = np.array([])
	label_set = np.array([])
	y_set = np.array([])
	while not E.is_set():
		ELDA.wait(timeout = 10)
		if not E.is_set():
			while featureQ.qsize():
					try:
						feat, ll = featureQ.get(0)
						# print('|LDA feat [',feat.shape,']\n|ll [',ll,']')
					except Empty:
						return
			if not feature_set.size == 0:
				feature_set = np.vstack((feature_set, feat))
			else:
				feature_set = np.array(feat)
			if not label_set.size == 0:
				label_set = np.hstack((label_set, ll))
			else:
				label_set = np.array(ll)
			# print('|LDA feat [',feature_set.shape,']\n|ll [',label_set.shape,']')
			feat = feat.reshape(1,-1)
			y = modelQ.predict(feat)
			print('LDA predict:', y)
			# y_set.append(y)
			# threadLock.acquire()
			# classOutQ.put(y)
			# threadLock.release()
			print('|||||LDA label', ll, 'vs.', y,' LDA predicted')
			ELDA.clear() #print('lda')
		else:
			break
	threadLock.acquire()
	dataclassOutQ.put([feature_set, label_set, y_set])
	threadLock.release()
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
			if not feature_set.size == 0:
				feature_set = np.vstack((feature_set, feat))
			else:
				feature_set = np.array(feat)
			if not label_set.size == 0:
				label_set = np.vstack((label_set, ll))
			else:
				label_set = np.array(ll)
			# print('|CCA feat [',feature_set.shape,']\n|ll [',label_set.shape,']')

			y = modelQ.predict(feat)
			print('CCA predict:', y)
			# y_set.append(y)
			# threadLock.acquire()
			# classOutQ.put(y)
			# threadLock.release()
			print('|||||CCA label', ll, 'vs.', y, ' CCA predicted')
			ECCA.clear() #print('cca')
		else:
			break
	threadLock.acquire()
	dataclassOutQ.put([feature_set, label_set, y_set])
	threadLock.release()
	print('||End of CCA prediction...')


##EXECUTION
if __name__ == '__main__':
	init_dialog()

	print('::Session parameters::')
	print(u'[Current path][%s]\n[Partic path][%s]\n[CSV filename][%s]\n[Clas. filename][%s]\n[Session Mode][%s]\n[Num Trials][%d]\n[Num Stim][%d]' 
			% (_currDir, _partDir, _csvsession, _classifile, session_mode, TRIALNUM, P3NUM))

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
	CCA_model = CCA(n_components = 2) #alter number of components for CCA
	#
	threadLock = threading.Lock()
	SaveQ = queue.Queue()
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
	#
	EPLabel = threading.Event()
	ES1Label = threading.Event()
	ES2Label = threading.Event()
	ES3Label = threading.Event()
	lockcounter = False
	#
	AcqTh = AcqThread(DataOutQ1, DataOutQ2, SaveQ)
	P300Th = P300Thread(DataOutQ1, FeatQ1)
	SSVEPTh = SSVEPThread(DataOutQ2, FeatQ2)
	#
	if session_mode == 1:
		LDATh = LDA_Thread(FeatQ1, ModelQ1, ClassOutQ1, DataClassOutQ1, session_mode)
		CCATh = CCA_Thread(FeatQ2, ModelQ2, ClassOutQ2, DataClassOutQ2, session_mode)
	elif session_mode == 2:
		classifile_path__ = os.path.join(_partDir, _classifile)
		if os.path.isfile(classifile_path__):
			with open(classifile_path__, 'rb') as input:
				cca_model, lda_model = pickle.load(input)
			LDATh = LDA_Thread(FeatQ1, lda_model, ClassOutQ1, DataClassOutQ1, session_mode)
			CCATh = CCA_Thread(FeatQ2, cca_model, ClassOutQ2, DataClassOutQ2, session_mode)
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
	FlkWin = FlickeringBoxes([0,10])

	##::CONNECTION
	BoardShim.enable_dev_board_logger()
	params = BrainFlowInputParams ()
	params.serial_port = 'COM5' #'COM3' #'COM11'
	board_id = BoardIds.CYTON_BOARD.value
	board = BoardShim (board_id, params)
	print('|Looking for an EEG stream...')
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

	FlkWin.start()

	AcqTh.join()
	CCATh.join()
	LDATh.join()

	board.stop_stream()
	board.release_session()

	if session_mode == 1:
		print("Finishing training session")
		flag1, flag2 = 1, 1
		if ModelQ1.qsize():
			while ModelQ1.qsize():
				try:
					lda_model = ModelQ1.get(0)
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
					cca_model = ModelQ2.get(0)
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

		if flag1 and flag2:
			with open(classifile_path__, 'wb') as output:
				pickle.dump([cca_model, lda_model], output, pickle.HIGHEST_PROTOCOL)
				print("Models saved")
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
		#### May need stacking adjustments befpre saving
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
			#### May need stacking adjustments befpre saving
			# with open(classioutCCA_path__, 'w', newline='') as file:
			# 	writer = csv.writer(classioutCCA_path__)
			# 	writer.writerows([label_set, y_set, feature_set])



	if SaveQ.qsize():
		while SaveQ.qsize():
			try:
				y, x = SaveQ.get(0)
			except Empty:
				pass
		plt.plot(x,y.transpose(),'k')
		plt.savefig(imagefile_path__)
		plt.show()

	print("\n\n|||||DONE|||||\n\n")
	FlkWin.stop()
