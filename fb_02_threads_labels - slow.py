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
from psychopy import core, visual
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

import warnings
warnings.filterwarnings("error")


##CLASSES
class FlickeringBoxes:
	def __init__(self, position):
		#Definitions
		self.H = 720-25 #window hight (to fit side of the window)
		self.W = 853 #window width
		# self.H4 = np.floor(H/4)
		# self.W4 = np.floor(W/4)
		self.sS = 0.25			#square size
		self.pS = self.sS*1.35	#padding size
		self.p3S = self.sS*1.75	#p300 size

		# app.logger.warn('\nInitializing Flickering...')
		#Create the Window
		self.win = visual.Window([self.W,self.H], position, monitor = 'SSVEP Paradigm', color = 'black')
		self.win.flip()


	# def profile(fnc):
	# 	def inner(*args, **kwargs):
	# 		pr = cProfile.Profile()
	# 		pr.enable()
	# 		retval = fnc(*args, **kwargs)
	# 		pr.disable()
	# 		s = io.StringIO()
	# 		sortby = 'cumulative'
	# 		ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
	# 		ps.print_stats()
	# 		print(s.getvalue())
	# 		return retval
	# 	return inner
		
	# @profile
	def start(self):
		p300clock = core.Clock()
		genclock = core.Clock()
		ISI = core.StaticPeriod(screenHz = 60)
		self.message = visual.TextStim(self.win, text = 'Flickering routine\n\nReady?')
		self.message.draw()
		self.win.flip()
		# core.wait(1.5)

		self.message = visual.TextStim(self.win, text = '3')
		self.message.draw()
		self.win.flip()
		# core.wait(1.0)

		self.message = visual.TextStim(self.win, text = '2')
		self.message.draw()
		self.win.flip()
		# core.wait(1.0)

		self.message = visual.TextStim(self.win, text = '1')
		self.message.draw()
		self.win.flip()
		# core.wait(1.0)

		print("|||\t\t>>>>>Initializing Flickering>>>>>")
		# T0 = time.time()
		T0 = genclock.getTime()
		p =[[-0.5, -0.5],[0, 0.5],[0.5, -0.5]] #from the centre; 1 is for 100%, 0 is for 50% and -1 is for 0% of screen
		vert = [[1,1],[1,-1],[-1,-1],[-1,1]]

		# self.stSq0 = visual.Rect(self.win, fillColor = 'black', lineColor = 'black', width = self.pS, height = self.pS, pos = p[0])
		# self.stSq1 = visual.Rect(self.win, fillColor = 'black', lineColor = 'black', width = self.pS, height = self.pS, pos = p[1])
		# self.stSq2 = visual.Rect(self.win, fillColor = 'black', lineColor = 'black', width = self.pS, height = self.pS, pos = p[2])
		self.stSq0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = p[0], autoDraw = True)
		self.stSq1 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = p[1], autoDraw = True)
		self.stSq2 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = p[2], autoDraw = True)
		
		for TL in TrialLabels:
			self.message = visual.TextStim(self.win, text = 'Focus on the orange:', pos = [0, 0.75])
			if TL == 0:
				# self.Sq0 = visual.Rect(self.win, fillColor = '#FF6C00', lineColor = '#FF6C00', width = self.sS, height = self.sS, pos = p[0])
				# self.Sq1 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.sS, height = self.sS, pos = p[1])
				# self.Sq2 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.sS, height = self.sS, pos = p[2])
				self.Sq0 = visual.ShapeStim(self.win, vertices = vert, fillColor = '#FF6C00', lineColor = '#FF6C00', size = self.sS, pos = p[0])
				self.Sq1 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = p[1])
				self.Sq2 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = p[2])
			if TL == 1:
				# self.Sq0 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.sS, height = self.sS, pos = p[0])
				# self.Sq1 = visual.Rect(self.win, fillColor = '#FF6C00', lineColor = '#FF6C00', width = self.sS, height = self.sS, pos = p[1])
				# self.Sq2 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.sS, height = self.sS, pos = p[2])
				self.Sq0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = p[0])
				self.Sq1 = visual.ShapeStim(self.win, vertices = vert, fillColor = '#FF6C00', lineColor = '#FF6C00', size = self.sS, pos = p[1])
				self.Sq2 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = p[2])
			if TL == 2:
				# self.Sq0 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.sS, height = self.sS, pos = p[0])
				# self.Sq1 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.sS, height = self.sS, pos = p[1])
				# self.Sq2 = visual.Rect(self.win, fillColor = '#FF6C00', lineColor = '#FF6C00', width = self.sS, height = self.sS, pos = p[2])
				self.Sq0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = p[0])
				self.Sq1 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = p[1])
				self.Sq2 = visual.ShapeStim(self.win, vertices = vert, fillColor = '#FF6C00', lineColor = '#FF6C00', size = self.sS, pos = p[2])

			self.message.draw()
			self.Sq0.draw()
			self.Sq1.draw()
			self.Sq2.draw()

			self.win.flip()
			core.wait(3)

			if TL == 0:
				ES1Label.set()
			elif TL == 1:
				ES2Label.set()
			elif TL == 2:
				ES3Label.set()

			ct = 0
			mx_ct = 1799
			op_off = 0
			op_on = 1
			endflag = False

			sequenceP300 = []
			for m in range (0,3):
				for k in range(0,1): # INCREASE TO 7 #number of blinks per square
					sequenceP300.append(m)
			random.shuffle(sequenceP300)
			blinkLabels = sequenceP300 #copied because sequenceP300 gets destroyed
			print("\n|||||||||P300 labels 4 TL ",TL," {",blinkLabels,"}\n")

			# t0 = time.time()
			t0 = p300clock.getTime()
			current_P300 = sequenceP300.pop(0)
			if TL == current_P300:
				EPLabel.set()
			else:
				EPLabel.clear()
			if TL == 0:
				ES1Label.set()
			elif TL == 1:
				ES2Label.set()
			else:
				ES3Label.set()
			# print('|Exp: -- P', EPLabel.is_set(),' -- S1', ES1Label.is_set(),' -- S2', ES2Label.is_set(),' -- S3', ES3Label.is_set())
			P300TRIG.set()
			# print('\t\t\tP300TRIG set')
			while not endflag:#True:
				# ISI.start(0.5)
				try:
					if ct%2 >= 1:
						self.Sq0 = visual.ShapeStim(self.win, vertices = vert, fillColor = '#00388A', lineColor = '#00388A', pos = p[0], opacity = op_on) #blue
					else:
						self.Sq0 = visual.ShapeStim(self.win, vertices = vert,  fillColor = '#59FF00', lineColor = '#59FF00', pos = p[0], opacity = op_on) #green

					if ct%4 >= 2:
						self.Sq1 = visual.ShapeStim(self.win, vertices = vert, fillColor = '#59FF00', lineColor = '#59FF00', pos = p[1], opacity = op_on) #green
					else:
						self.Sq1 = visual.ShapeStim(self.win, vertices = vert, fillColor = '#B80117', lineColor = '#B80117', pos = p[1], opacity = op_on) #red

					if ct%6 >= 3:
						self.Sq2 = visual.ShapeStim(self.win, vertices = vert, fillColor = '#59FF00', lineColor = '#59FF00', pos = p[2], opacity = op_on) #green
					else:
						self.Sq2 = visual.ShapeStim(self.win, vertices = vert, fillColor = '#B80117', lineColor = '#B80117', pos = p[2], opacity = op_on) #red

					# t1 = time.time()
					t1 = p300clock.getTime()
					# print(t1-t0)
					if t1-t0 < 0.2:
						if current_P300 == 0:
							self.p300Sq0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.p3S, pos = p[0])
						if current_P300 == 1:
							self.p300Sq0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.p3S, pos = p[1])
						if current_P300 == 2:
							self.p300Sq0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.p3S, pos = p[2])
					elif t1-t0 >= 0.2 and t1-t0 < 0.45:
						self.p300Sq0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.p3S, pos = p[0], opacity = op_off)
						COLLECTTRIG.set()
						P300TRIG.clear()
						# print('\t\t\tCOLLECTTRIG set')
					elif t1-t0 >= 0.5:
						# t0 = time.time()
						t0 = p300clock.getTime()
						try:
							current_P300 = sequenceP300.pop(0)
							if TL == current_P300:
								EPLabel.set()
							else:
								EPLabel.clear()
							# print('|Exp: -- P', EPLabel.is_set(),' -- S1', ES1Label.is_set(),' -- S2', ES2Label.is_set(),' -- S3', ES3Label.is_set())
							P300TRIG.set()
							COLLECTTRIG.clear()
							# print("Got it in 0.5:", ISI.complete())
							# print('\t\t\tP300TRIG set')
						except IndexError as e:
							endflag = True
					
					self.p300Sq0.draw()
					self.stSq0.draw()
					self.stSq1.draw()
					self.stSq2.draw()
					self.Sq0.draw()
					self.Sq1.draw()
					self.Sq2.draw()

					self.win.flip()
					ct += 1
					if ct>= mx_ct:
						ct = 0

				except KeyboardInterrupt:
					self.stop()

		time.sleep(0.5) #wait for processes to finish
		E.set()
		# T1 = time.time()
		T1 = genclock.getTime()
		print('||Flickering took', round(T1-T0,3), 's')

	def stop(self):
		self.win.close()
		core.quit()

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

class ClassifThread(threading.Thread):
	def __init__(self, dataOutQ):
		threading.Thread.__init__(self)
		self.dataOutQ = dataOutQ

	def run(self):
		pass
		#Here goes function for classification


##FUNCTIONS
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
	chs_p300 = [1,3,5]	#available channels [1 to 8]
	chs_ssvep = [2,4,6]	#available channels [1 to 8]
	chs_all = list(set(chs_p300 + chs_ssvep))
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
		if P300TRIG.is_set(): #function called when

			##::Getting Labels
			print('|Acq: -- P', EPLabel.is_set(),' -- S1', ES1Label.is_set(),' -- S2', ES2Label.is_set(),' -- S3', ES3Label.is_set())
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
			print('|Labels ll and TL =', ll, TL)
			ES1Label.clear()
			ES2Label.clear()
			ES3Label.clear()

			##::Getting Signal
			# time.sleep(0.5) #give time to fill the buffer
			COLLECTTRIG.wait()
			print('\t\t\tEnd sleep')
			# print("\n\n|Latest samples: ", board.get_board_data_count() ) #last data
			data = board.get_board_data()
			ss_p = np.array(data[chs_p300,:]) #maybe collect only last 125 samples?
			ss_s = np.array(data[chs_ssvep,:])
			ss = np.array(data[chs_all,:])
			tt = np.array(data[22,:])
			print('\t\t\tData collected')

			if not rawsamples.size == 0:
				rawsamples = np.hstack((rawsamples, ss))
			else:
				rawsamples = np.array(ss)
			if not rawtimestamps.size == 0:
				rawtimestamps = np.append(rawtimestamps, tt)
			else:
				rawtimestamps = np.array(tt)

			##::Send info for both SSVEP and P300 threads
			p_avgsamples = normalization(ss_p)
			p_filtsamples = np.array(lfilter(b, a, p_avgsamples, axis = 1))
			s_avgsamples = normalization(ss_s)
			s_filtsamples = np.array(lfilter(b, a, s_avgsamples, axis = 1))
			threadLock.acquire()				#queue locked to prevent other threads to access it
			dataOutQ1.put([p_filtsamples, tt, ll]) 	#data is put in the queue
			dataOutQ2.put([s_filtsamples, tt, TL]) 	#data is put in the queue
			# print('|Samples sent: AVG [',avgsamples.shape,'], Filt\'d [',filtsamples.shape,'], tt [',tt.shape,']' ) #for numpy type
			threadLock.release()				#queue unlocked with info\
			print('\t\t\tFil, norm, queue')
			##::(Re)set Events
			EP300.set()
			ESSVEP.set()
			P300TRIG.clear()
			print('\t\t\tP300TRIG clear')
		else:
			pass

		i += 1

	threadLock.acquire()
	saveQ.put([rawsamples[:,:], rawtimestamps[:]])
	threadLock.release()
	###
	print('\n\n|||Ending acquisition')
	finish = time.perf_counter()
	print('___\n|Final Score: Samples [',rawsamples.shape,'], Timestamp [',rawtimestamps.shape,']') #for numpy type
	print('|Streamed in',round(finish-start,3),'s!\n---')


def P300fun(dataInQ, featureQ):
	labels = []
	ss = np.array([])
	tt = np.array([])
	while not E.is_set():
		if EP300.is_set():
			##::Collect Samples from Queue
			while dataInQ.qsize(): # if not dataqueue.empty():
				try:
					ss, tt, ll = dataInQ.get(0)
					print('|P300 ss [', ss.shape, ']\n|P300 tt [', tt.shape, ']') #for list type
				except Empty:
					return

			##::Average all samples together
			labels.append(ll)
			mean_ss = np.mean(ss, 0)
			# print("\tShape means P300", mean_ss.shape)
			##::Donwsample == Features ~125sample?
			ss_ds = mean_ss[::2]
			tt_ds = tt[::2]

			threadLock.acquire()
			featureQ.put([ss_ds, tt_ds, ll])
			threadLock.release()

			##::END OF PROCESSING
			EP300.clear()
			ss = np.array([])
			tt = np.array([])
	print('|||||||||P300 labels (ll) {',labels,'}')
	print('|||P300 thread Finished')

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
		if ESSVEP.is_set():
			##::Collect Samples from Queue
			while dataInQ.qsize():
				try:
					ss, tt, ll = dataInQ.get(0)
					print('|SSVEP ss [',ss.shape,']\n|SSVEP tt [',tt.shape,']')
				except Empty:
					return

			##::Average all samples together
			labels.append(ll)
			mean_ss = np.mean(ss, 0)
			# print("\tShape means SSVEP", mean_ss.shape)
			f1 = np.array(lfilter(b1, a1, mean_ss))
			f2 = np.array(lfilter(b2, a2, mean_ss))
			f3 = np.array(lfilter(b3, a3, mean_ss))
			feat = np.sum([[f1],[f2],[f3]],0)

			threadLock.acquire()
			featureQ.put([feat, tt, ll])
			threadLock.release()

			##::END OF PROCESSING
			ESSVEP.clear()
			ss = np.array([])
			tt = np.array([])
	print('|||||||||SSVEP labels (TL) {',labels,'}')
	print('|||SSVEP thread Finished')


##EXECUTION
if __name__ == '__main__':
	##::Definitions
	threadLock = threading.Lock()
	SaveQ = queue.Queue()
	DataOutQ1 = queue.Queue()
	DataOutQ2 = queue.Queue()
	FeatQ1 = queue.Queue()
	FeatQ2 = queue.Queue()
	#
	E = threading.Event()
	P300TRIG = threading.Event()
	COLLECTTRIG = threading.Event()
	EP300 = threading.Event()
	ESSVEP = threading.Event()
	EPLabel = threading.Event()
	ES1Label = threading.Event()
	ES2Label = threading.Event()
	ES3Label = threading.Event()
	lockcounter = False
	#
	AcqTh = AcqThread(DataOutQ1, DataOutQ2, SaveQ)
	P300Th = P300Thread(DataOutQ1, FeatQ1)
	SSVEPTh = SSVEPThread(DataOutQ2, FeatQ2)

	##::True Labels for the trials
	TrialLabels = []
	for n in range (0,3):
		for k in range(0,1): ##INCREASE TO 7 #number of trials per square (total trials = n*k = 21)
			TrialLabels.append(n)
	random.shuffle(TrialLabels)
	print("\n|||||||||Trial labels Sequence {",TrialLabels,"}\n")

	# x = (1920/4)		#=480
	# y = (720/4)		#=180
	FlkWin = FlickeringBoxes([480,180])

	##::CONNECTION
	BoardShim.enable_dev_board_logger()
	params = BrainFlowInputParams ()
	params.serial_port = 'COM3' #'COM11'
	board_id = BoardIds.CYTON_BOARD.value
	board = BoardShim (board_id, params)
	print('|Looking for an EEG stream...')
	board.prepare_session() #connect to board
	print('|Connected\n--\n')
	board.start_stream(45000, 'file://fb_02_data.csv:w') #streams data and saves it to csv file
	print('|||Starting...')

	##::START FLICKER AND ACQUISITION
	AcqTh.start()
	P300Th.start()
	SSVEPTh.start()

	FlkWin.start()

	AcqTh.join()
	P300Th.join()
	SSVEPTh.join()

	board.stop_stream()
	board.release_session()

	if SaveQ.qsize():
		while SaveQ.qsize():
			try:
				y, x = SaveQ.get(0)
			except Empty:
				pass
		plt.plot(x,y.transpose(),'k')
		plt.savefig('fb_02_all_samples.png')
		plt.show()

	print("\n\n|||||DONE|||||\n\n")