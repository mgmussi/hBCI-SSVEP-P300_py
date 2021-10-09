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
# import AppWindow as app
import random
import time
from pylsl import StreamInlet, resolve_stream
# import tkinter as tk
# import AppWindow as app
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import queue
from queue import Empty
from datetime import datetime
import pandas as pd
from scipy.signal import butter, lfilter
import cProfile, pstats, io
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
		
	@profile
	def start(self):
		p300clock = core.Clock()
		genclock = core.Clock()
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

		# AcqTh.start()
		# P300Th.start()
		# SSVEPTh.start()

		print("\t\t>>>>>Initializing Flickering>>>>>")
		# T0 = time.time()
		T0 = genclock.getTime()
		p =[[-0.5, -0.5],[0, 0.5],[0.5, -0.5]] #from the centre; 1 is for 100%, 0 is for 50% and -1 is for 0% of screen

		self.stSq0 = visual.Rect(self.win, fillColor = 'black', lineColor = 'black', width = self.pS, height = self.pS, pos = p[0])
		self.stSq1 = visual.Rect(self.win, fillColor = 'black', lineColor = 'black', width = self.pS, height = self.pS, pos = p[1])
		self.stSq2 = visual.Rect(self.win, fillColor = 'black', lineColor = 'black', width = self.pS, height = self.pS, pos = p[2])

		self.Sq0 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.sS, height = self.sS, pos = p[0], opacity = 1)
		self.Sq1 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.sS, height = self.sS, pos = p[1])
		self.Sq2 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.sS, height = self.sS, pos = p[2])

		self.Sq0.draw()
		self.Sq1.draw()
		self.Sq2.draw()
		self.stSq0.draw()
		self.stSq1.draw()
		self.stSq2.draw()

		self.win.flip()
		# core.wait(0.5)

		ct = 0
		mx_ct = 1799
		op_off = 0.25
		op_on = 1

		kt = 0
		endflag = False
		
		selection = []
		for l in range (0,3):
			for k in range(0,12):
				selection.append(l)
		random.shuffle(selection)
		random.shuffle(selection)
		random.shuffle(selection)
		# print(selection)

		select = selection.pop(0)
		# t0 = time.time()
		t0 = p300clock.getTime()
		while not endflag:#True:
			try:
				if ct%2 >= 1:
					#blue
					self.Sq0 = visual.Rect(self.win, fillColor = '#00388A', lineColor = '#00388A',  width = self.sS, height = self.sS, pos = p[0], opacity = op_on)
				else:
					#green
					self.Sq0 = visual.Rect(self.win, fillColor = '#59FF00', lineColor = '#59FF00', width = self.sS, height = self.sS, pos = p[0], opacity = op_on)

				if ct%4 >= 2:
					#green
					self.Sq1 = visual.Rect(self.win, fillColor = '#59FF00', lineColor = '#59FF00', width = self.sS, height = self.sS, pos = p[1], opacity = op_on)
				else:
					#red
					self.Sq1 = visual.Rect(self.win, fillColor = '#B80117', lineColor = '#B80117', width = self.sS, height = self.sS, pos = p[1], opacity = op_on)

				if ct%6 >= 3:
					#green
					self.Sq2 = visual.Rect(self.win, fillColor = '#59FF00', lineColor = '#59FF00', width = self.sS, height = self.sS, pos = p[2], opacity = op_on)
				else:
					#red
					self.Sq2 = visual.Rect(self.win, fillColor = '#B80117', lineColor = '#B80117', width = self.sS, height = self.sS, pos = p[2], opacity = op_on)

				# t1 = time.time()
				t1 = p300clock.getTime()
				print(t1-t0)
				if t1-t0 < 0.2:
					if select == 0:
						self.p300Sq0 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.p3S, height = self.p3S, pos = p[0], opacity = op_on)
					elif select == 1:
						self.p300Sq0 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.p3S, height = self.p3S, pos = p[1], opacity = op_on)
					elif select == 2:
						self.p300Sq0 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.p3S, height = self.p3S, pos = p[2], opacity = op_on)
				elif t1-t0 >= 0.2 and t1-t0 < 0.3:
					self.p300Sq0 = visual.Rect(self.win, fillColor = 'black', lineColor = 'black', width = self.p3S, height = self.p3S, pos = p[0], opacity = op_off)
				elif t1-t0 >= 0.3:
					# t0 = time.time()
					t0 = p300clock.getTime()
					try:
					    select = selection.pop(0)
			    		P300TRIG.set()
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
				kt += 1
				if ct>= mx_ct:
					ct = 0

			except KeyboardInterrupt:
				self.stop()

		E.set()
		# T1 = time.time()
		T1 = genclock.getTime()
		print('Whole process took', T1-T0, 's')

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
		Acquisition(inlet, self.dataOutQ1, self.dataOutQ2, self.saveQ) #place where function is called

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
		# print('Mean vals @ i =',i,':\n',mean,'\nAvgSample:\n',avgsamples,'\nMaxx:\n',maxx,'\nMinn:\n',minn)
		print('Caution: Channel RAIL\n')
		# input("Please, evaluate inputs and hit \'Enter\' to continue")
	return avgsamples

def Acquisition(inlet, dataOutQ1, dataOutQ2, saveQ):
	global chk_chs

	chk_chs = [0,1] #available channels [0 to 7]
	fs = 256 		#Hz
	nyq = 0.5*fs
	low = 5 / nyq
	high = 50 / nyq
	b, a = butter(4, [low, high], btype='band') # b, a = butter(order, [low, high], btype='band')

	rawsamples = np.empty([len(chk_chs), 1])
	Pavgsamples = np.empty([len(chk_chs), 1])
	Savgsamples = np.empty([len(chk_chs), 1])
	Pfiltsamples = np.empty([len(chk_chs), 1])
	Sfiltsamples = np.empty([len(chk_chs), 1])
	rawtimestamps = np.empty([1,1])

	i = 0
	print('Starting Acquisition...')
	start = time.perf_counter()
	###
	while not E.is_set():#while i<1250:
		sample, timestamp = inlet.pull_sample() # print('>>Samples [', sample, ']\nTimestamp [', timestamp, ']') #for list type
		if i == 0:
			pass
		else:
			ss = []
			for ch in chk_chs:
				ss.append([sample[ch]])
			ss = np.array(ss)
			#
			if not rawsamples.size == 0:
				rawsamples = np.hstack((rawsamples, ss))
			else:
				rawsamples = np.array(sample)
			#
			if not rawtimestamps.size == 0:
				rawtimestamps = np.append(rawtimestamps, timestamp)
			else:
				rawtimestamps = np.array(timestamp)

		## First Branch:: SSVEP
		if i>0 and i % 250 == 0:
			print('\n___\n<>ACQUI\n--ESSVEP is set >>',ESSVEP.is_set())
			Savgsamples = normalization(rawsamples[:,-250:])
			Sfiltsamples = np.array(lfilter(b, a, Savgsamples, axis = 1))
			threadLock.acquire()				#queue locked to prevent other threads to access it
			dataOutQ2.put([Sfiltsamples[:,-250:], rawtimestamps[-250:]]) 	#data is put in the queue
			# print('|Sizes: AVG [,'Savgsamples.shape,'], Filt\'d [',Sfiltsamples.shape,']' ) #for numpy type
			print('|Sending: <>Samples [',Sfiltsamples[:,-250:].shape,'] <>Timestamp [',rawtimestamps[-250:].shape,']' ) #for numpy type
			threadLock.release()				#queue unlocked with info
			ESSVEP.set()
			print('--ESSVEP is set >>,'ESSVEP.is_set())

		## Second Branch:: P300
		if P300TRIG.is_set():
			if not lockcounter:
				i0 = i
				lockcounter = True
			if i-i0 == 200: #200 samples == 800ms
				print('\n___\n<>ACQUI\n--P300 is set >>',P300TRIG.is_set())
				Pavgsamples = normalization(rawsamples[:,-200:])
				Pfiltsamples = np.array(lfilter(b, a, Pavgsamples, axis = 1))
				threadLock.acquire()				#queue locked to prevent other threads to access it
				dataOutQ1.put([Pfiltsamples[:,-200:], rawtimestamps[-200:]]) 	#data is put in the queue
				# print('|Sizes: AVG [,'Pavgsamples.shape,'], Filt\'d [',Pfiltsamples.shape,']' ) #for numpy type
				print('|Sending: <>Samples [',Pfiltsamples[:,-250:].shape,'] <>Timestamp [',rawtimestamps[-250:].shape,']' ) #for numpy type
				threadLock.release()				#queue unlocked with info
		else:
			lockcounter = False

		# threadLock.acquire()				
		# dataOutQ2.put([rawsamples,rawtimestamps])
		# threadLock.release()

		i += 1
	###
	print('\n\t\tATENTION E is set >>', ESSVEP.is_set(),'\n')
	# print('><>< my i equals ', i, '><><')
	finish = time.perf_counter()

	print('\n___\n|Sizes: Samples [',rawsamples.shape,'], Timestamp [,'rawtimestamps.shape,']') #for numpy type
	print('|Sucessfully streamed in',round(finish-start,3),'s!\n---')


def P300fun(dataInQ, featureQ):
	p300sample = np.empty([len(chk_chs),1])
	p300timestamp = np.empty([1,1])
	# print(f"Is DataInQ size true? {DataOutQ1.qsize()}")
	# print("Is dataInQ emtpy?", DataOutQ1.empty())
	while not E.is_set():
		if EP300.is_set():
			# while dataInQ.qsize(): # if not dataqueue.empty():
			while dataInQ.qsize(): # if not dataqueue.empty():
			# if not dataInQ.empty():
				try:
					p300sample, p300timestamp = dataInQ.get(0)
					print('\n___\n<>P300\n>>Samples [', p300sample, ']\nTimestamp [', p300timestamp, ']') #for list type
				except Empty:
					if not E.is_set():
						EP300.clear()
						p300sample = np.empty([len(chk_chs),1])
						p300timestamp = np.empty([1,1])
					return
			##data with timestamp
			# plt.plot(rawtimestamps, rawsamples.transpose(),'k')
			# plt.plot(rawtimestamps, avgsamples.transpose(), 'red')
			plt.plot(p300timestamp, p300sample.transpose(), 'blue')

			plt.savefig('P300_samples.png')
			plt.show()
	print('P300 thread Finished')

def SSVEPfun(dataInQ, featureQ):
	ssvepsample = np.empty([len(chk_chs),1])
	ssveptimestamp = np.empty([1,1])

	while not E.is_set():
		if ESSVEP.is_set():
			while dataInQ.qsize():
				try:
					ssvepsample, ssveptimestamp = dataInQ.get(0)
					print('\n___\n<>SSVEP\n>>Samples [',ssvepsample.shape,'] >>Timestamp [',ssveptimestamp.shape,']')
				except Empty:
					return
			if not E.is_set():
				print('ESSVEP is set >>',ESSVEP.is_set())
				ESSVEP.clear()
				print('And now is it? >>',ESSVEP.is_set(),'\n---\n')
				ssvepsample = np.empty([len(chk_chs),1])
				ssveptimestamp = np.empty([1,1])
			# ##data with timestamp
			# # plt.plot(rawtimestamps, rawsamples.transpose(),'k')
			# # plt.plot(rawtimestamps, avgsamples.transpose(), 'red')
			# plt.plot(ssveptimestamp, ssvepsample.transpose(), 'blue')

			# plt.savefig('SSVEP_samples.png')
			# plt.show()
	print('SSVEP thread Finished')


##EXECUTION
if __name__ == '__main__':
	#Definitions
	threadLock = threading.Lock()
	SaveQ = queue.Queue()
	DataOutQ1 = queue.Queue()
	DataOutQ2 = queue.Queue()
	FeatQ1 = queue.Queue()
	FeatQ2 = queue.Queue()
	#
	E = threading.Event()
	P300TRIG = threading.Event()
	EP300 = threading.Event()
	ESSVEP = threading.Event()
	lockcounter = False
	#
	AcqTh = AcqThread(DataOutQ1, DataOutQ2, SaveQ)
	P300Th = P300Thread(DataOutQ1, FeatQ1)
	SSVEPTh = SSVEPThread(DataOutQ2, FeatQ2)
	# x = (1920/4)		#=480
	# y = (720/4)		#=180
	FlkWin = FlickeringBoxes([480,180])

	#Connection
	print('Looking for an EEG stream...')
	streams = resolve_stream('type', 'EEG')
	inlet = StreamInlet(streams[0])
	print('Connected!\n')
	
	FlkWin.start()

	# AcqTh.join()
	# P300Th.join()
	# SSVEPTh.join()

	print("\n\n>>>DONE<<<\n\n")



	# plt.figure(figsize = (5,6), dpi = 100)

	# ##data without timestamp
	# # plt.plot(rawsamples.transpose(),'k')
	# # plt.plot(avgsamples.transpose(), 'red')

	# ##data with timestamp
	# # plt.plot(rawtimestamps, rawsamples.transpose(),'k')
	# # plt.plot(rawtimestamps, avgsamples.transpose(), 'red')
	# plt.plot(rawtimestamps, filtsamples.transpose(), 'blue')

	# plt.savefig('all_samples.png')
	# plt.show()
	# print(rawsamples, '\n_____')
	# print(rawtimestamps, '\n_____\n_____')