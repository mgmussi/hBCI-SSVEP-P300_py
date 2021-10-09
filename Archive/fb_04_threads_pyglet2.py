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
# from psychopy import core, visual
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

import pyglet
from pyglet.gl import *
from pyglet import shapes

import warnings
warnings.filterwarnings("error")

class FastRect:
	def __init__(self, x, y, width, height, color, batch):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.color = color
		self.batch = batch
		r = color[0]
		g = color[1]
		b = color[2]
		self.draw_mode = GL_QUADS
		self.vertex_list = self.batch.add(4, self.draw_mode, None, 'v2f/stream', 'c3B/stream')
		self.vertex_list.vertices = [x, y,
									x + width, y,
									x + width, y + height,
									x, y + height]
		self.vertex_list.colors = [r, g, b,
									r, g, b,
									r, g, b,
									r, g, b]

	def changeColor(self, new_color):
		self.__init__(self.x, self.y, self.width, self.height, new_color, self.batch)

	def Reposition(self, x, y):
		self.__init__(x, y, self.width, self.height, self.color, self.batch)


class StimuliWindow(pyglet.window.Window):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		width = args[0]
		hight = args[1]

		#Definitions
		sS = 100			#square size
		pS = sS*1.35	#padding size
		p3S = sS*1.75	#p300 size

		##::Position
		self.p =[[(width-sS/2-3*width/4), (hight-sS/2-3*hight/4)],[(-sS/2+width/2), (hight-sS/2-hight/4)],[(width-sS/2-width/4), (hight-sS/2-3*hight/4)]]
		self.p1 =[[(width-pS/2-3*width/4), (hight-pS/2-3*hight/4)],[(-pS/2+width/2), (hight-pS/2-hight/4)],[(width-pS/2-width/4), (hight-pS/2-3*hight/4)]]
		self.p2 =[[(width-p3S/2-3*width/4), (hight-p3S/2-3*hight/4)],[(-p3S/2+width/2), (hight-p3S/2-hight/4)],[(width-p3S/2-width/4), (hight-p3S/2-3*hight/4)]]

		##::Create Batch
		self.batch = pyglet.graphics.Batch()

		##::Create FastRect elements
		#::P300 Box
		self.SqP = FastRect(self.p2[0][0], self.p2[0][1], p3S,p3S,  color=(255,255,255), batch = self.batch)
		#::Steady Black Boxes
		self.St0 = FastRect(self.p1[0][0], self.p1[0][1], pS,pS,  color=(0,0,0), batch = self.batch)
		self.St1 = FastRect(self.p1[1][0], self.p1[1][1], pS,pS,  color=(0,0,0), batch = self.batch)
		self.St2 = FastRect(self.p1[2][0], self.p1[2][1], pS,pS,  color=(0,0,0), batch = self.batch)
		#::SSVEP Boxes
		self.Sq0 = FastRect(self.p[0][0], self.p[0][1], sS,sS,  color=mygreen, batch = self.batch)
		self.Sq1 = FastRect(self.p[1][0], self.p[1][1], sS,sS,  color=mygreen, batch = self.batch)
		self.Sq2 = FastRect(self.p[2][0], self.p[2][1], sS,sS,  color=mygreen, batch = self.batch)


	def on_draw(self):
		self.clear()
		self.batch.draw()

	def update_win(self, dt):
		print("Last callback: ", dt)
		# t0 = time.time()
		global gencounter
		global stg
		global cyc
		global idx
		global sel
		global TL
		global sequenceP300

		if cyc > 2:
			COLLECTTRIG.set()
			P300TRIG.clear()
			self.SqP.changeColor([0,0,0])
		else:
			self.SqP.changeColor([255,255,255])

		if stg == 0:
			self.Sq0.changeColor(mygreen) #0
			self.Sq1.changeColor(mygreen) #0
			self.Sq2.changeColor(mygreen) #0
		elif stg == 1:
			self.Sq0.changeColor(myblue)  #1
			self.Sq1.changeColor(mygreen) #0
			self.Sq2.changeColor(mygreen) #0
		elif stg == 2:
			self.Sq0.changeColor(mygreen) #0
			self.Sq1.changeColor(myred)	  #1
			self.Sq2.changeColor(mygreen) #0
		elif stg == 3:
			self.Sq0.changeColor(myblue)  #1
			self.Sq1.changeColor(myred)	  #1
			self.Sq2.changeColor(mygreen) #0
		elif stg == 4:
			self.Sq0.changeColor(mygreen) #0
			self.Sq1.changeColor(mygreen) #0
			self.Sq2.changeColor(myred)	  #1
		elif stg == 5:
			self.Sq0.changeColor(myblue)  #1
			self.Sq1.changeColor(mygreen) #0
			self.Sq2.changeColor(myred)	  #1
		elif stg == 6:
			self.Sq0.changeColor(mygreen) #0
			self.Sq1.changeColor(myred)	  #1
			self.Sq2.changeColor(myred)	  #1
		elif stg == 7:
			self.Sq0.changeColor(myblue)  #1
			self.Sq1.changeColor(myred)   #1
			self.Sq2.changeColor(myred)	  #1

		
		# print("Gencount: ", gencounter, "| Stage: ", stg, "| Cycle: ", cyc, "| IDX: ", idx, "|P300.: ", sel, "|TL: ", TL)
		gencounter += 1
		stg += 1
		if stg > 7:
			cyc += 1
			stg = 0

			if cyc > 3:
				idx += 1
				cyc = 0
				gencounter = 1

				try:
					TL = ex_TrialLabels[idx]
					sel = sequenceP300[idx]
					self.SqP.Reposition(self.p2[sel][0], self.p2[sel][1])
					P300TRIG.set()
					COLLECTTRIG.clear()

					##::Setting p300 Label
					if TL == sel:
						EPLabel.set()
					else:
						EPLabel.clear()

					##::Setting SSVEP Label
					if TL == 0:
						ES1Label.set()
					elif TL == 1:
						ES2Label.set()
					elif TL == 2:
						ES3Label.set()

				except IndexError as e:
					print("||||||||||QUITING PYGLET")
					E.set()
					pyglet.clock.unschedule(self.update_win)
					LOOP.exit()
					print("||||||||||QUITED")


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

# @profile
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
			# time.sleep(0.5) #give time to fill the buffer
			COLLECTTRIG.wait()
			# print('\t\t\tEnd sleep')
			# print("\n\n|Latest samples: ", board.get_board_data_count() ) #last data
			data = board.get_board_data()
			ss_p = np.array(data[chs_p300,:]) #maybe collect only last 125 samples?
			ss_s = np.array(data[chs_ssvep,:])
			ss = np.array(data[chs_all,:])
			tt = np.array(data[22,:])
			# print('\t\t\tData collected')

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
			# print('\t\t\tFil, norm, queue')
			##::(Re)set Events
			EP300.set()
			ESSVEP.set()
			P300TRIG.clear()
			# print('\t\t\tP300TRIG clear')
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


@profile
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

##::Initial Values
gencounter = 1
stg = 0
cyc = 0
idx = 0

##::True Labels for the trials
TrialLabels = []
sequenceP300 = []
ex_TrialLabels = []
for n in range (0,3):		#Num. Lables
	for k in range(0,1): 	##INCREASE TO 7 #num trials per square (total trials = n*k = 21)
		TrialLabels.append(n)
random.shuffle(TrialLabels)

for t in TrialLabels:
	for m in range(0,3):
		for p in range(0,1): # INCREASE TO 7 #num blinks per square
			sequenceP300.append(m)
			ex_TrialLabels.append(t)
random.shuffle(sequenceP300)
print("\nTrial labels Sequence {",ex_TrialLabels,"}\nP300 target Sequence {",sequenceP300,"}\n")
TL = ex_TrialLabels[idx]
sel = sequenceP300[idx]

##::Color Definition
myblue = (0,56,138)
mygreen = (89,255,0)
myred = (118,0,23)
myorange = (255,108,0)

##::CONNECTION
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams ()
params.serial_port = 'COM3' #'COM11'
board_id = BoardIds.CYTON_BOARD.value
board = BoardShim (board_id, params)
print('|Looking for an EEG stream...')
board.prepare_session() #connect to board
print('|Connected\n--\n')
board.start_stream(45000, 'file://fb_04_data.csv:w') #streams data and saves it to csv file
print('|||Starting...')

LOOP = pyglet.app.EventLoop()

# @profile
def main():
	win = StimuliWindow(960,720, "Stimuli")
	win.set_location(0, 50)
	stg = 0
	cyc = 0

	##::START FLICKER AND ACQUISITION
	AcqTh.start()
	P300Th.start()
	SSVEPTh.start()

	pyglet.clock.schedule_interval(win.update_win, 1/120)
	#pyglet.app.run()
	LOOP.run()

@LOOP.event
def on_exit():
	print("On Exit!")
	AcqTh.join()
	P300Th.join()
	SSVEPTh.join()
	print("Threads Joined")

	board.stop_stream()
	board.release_session()

	if SaveQ.qsize():
		while SaveQ.qsize():
			try:
				y, x = SaveQ.get(0)
			except Empty:
				pass
		plt.plot(x,y.transpose(),'k')
		plt.savefig('fb_04_all_samples.png')
		plt.show()

	print("\n\n|||||DONE|||||\n\n")


##EXECUTION
if __name__ == '__main__':
	main()

	
