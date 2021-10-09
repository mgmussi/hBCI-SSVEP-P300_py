from pylsl import StreamInlet, resolve_stream
# import tkinter as tk
# import AppWindow as app
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import queue
from queue import Empty
import time
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import cProfile, pstats, io
import warnings
warnings.filterwarnings("error")
# import pyqtgraph as pg


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
		#Here goes function for P300

threadLock = threading.Lock()
SaveQ = queue.Queue()
DataOutQ1 = queue.Queue()
DataOutQ2 = queue.Queue()
FeatQ1 = queue.Queue()
FeatQ2 = queue.Queue()

E = threading.Event()
EP300 = threading.Event()
ESSVEP = threading.Event()
#
AcqTh = AcqThread(DataOutQ1, DataOutQ2, SaveQ)
P300Th = P300Thread(DataOutQ1, FeatQ1)
SSVEPTh = SSVEPThread(DataOutQ2, FeatQ2)

def normalization(matrix, i):
	## normalization
	mean = np.mean(matrix, 1)
	avgsamples = np.subtract(matrix.transpose(), mean).transpose()
	maxx = np.max(avgsamples, 1)
	minn = np.min(avgsamples, 1)
	try:
		avgsamples = (np.subtract(avgsamples.transpose(), minn)/np.subtract(maxx, minn)).transpose() #(avg-min)/(max-min)
	except RuntimeWarning:
		print(f'Mean vals @ i = {i}:\n{mean}\nAvgSample:\n{avgsamples}\nMaxx:\n{maxx}\nMinn:\n{minn}')
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
	avgsamples = np.empty([len(chk_chs), 1])
	filtsamples = np.empty([len(chk_chs), 1])
	rawtimestamps = np.empty([1,1])

	i = 0
	print('Starting...')
	start = time.perf_counter()
	###
	while i<1250:
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
			print(f'\n___\n<>ACQUI\n--Is it True that ESSVEP is set? >> {ESSVEP.is_set()}')
			avgsamples = normalization(rawsamples[:,-250:], i)
			filtsamples = np.array(lfilter(b, a, avgsamples, axis = 1))
			threadLock.acquire()				#queue locked to prevent other threads to access it
			dataOutQ2.put([filtsamples[:,-250:], rawtimestamps[-250:]]) 	#data is put in the queue
			print(f'|Sending: <>Samples [{filtsamples[:,-250:].shape}] <>Timestamp [{rawtimestamps[-250:].shape}]' ) #for numpy type
			threadLock.release()				#queue unlocked with info
			ESSVEP.set()
			print(f'--And now? >> {ESSVEP.is_set()}')

		## Second Branch:: P300
		# threadLock.acquire()				
		# dataOutQ2.put([rawsamples,rawtimestamps])
		# threadLock.release()

		i += 1
	###
	E.set()
	print(f'\n\t\tATENTION E is set >> {ESSVEP.is_set()}\n')
	# print('><>< my i equals ', i, '><><')
	finish = time.perf_counter()

	print(f'\n___\n|Sizes: Samples [{rawsamples.shape}], Timestamp [{rawtimestamps.shape}]' ) #for numpy type
	print(f'|Sizes: AVG [{avgsamples.shape}], Filt\'d [{filtsamples.shape}]' ) #for numpy type
	print(f'|Sucessfully streamed in {round(finish-start,3)}s!\n---')

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
					print(f'\n___\n<>SSVEP\n>>Samples [{ssvepsample.shape}] >>Timestamp [{ssveptimestamp.shape}]' )
				except Empty:
					return
			if not E.is_set():
				print(f'ESSVEP is set >> {ESSVEP.is_set()}')
				ESSVEP.clear()
				print(f'And now is it? >> {ESSVEP.is_set()}\n---\n')
				ssvepsample = np.empty([len(chk_chs),1])
				ssveptimestamp = np.empty([1,1])
			# ##data with timestamp
			# # plt.plot(rawtimestamps, rawsamples.transpose(),'k')
			# # plt.plot(rawtimestamps, avgsamples.transpose(), 'red')
			# plt.plot(ssveptimestamp, ssvepsample.transpose(), 'blue')

			# plt.savefig('SSVEP_samples.png')
			# plt.show()
	print('SSVEP thread Finished')


if __name__ == '__main__':
	# global inlet
	print('Looking for an EEG stream...')
	streams = resolve_stream('type', 'EEG')
	inlet = StreamInlet(streams[0])
	print('Connected!\n')

	AcqTh.start()
	P300Th.start()
	SSVEPTh.start()
	# time.sleep(.5)
	AcqTh.join()
	P300Th.join()
	SSVEPTh.join()

	print("\n\n>>>DONE<<<\n\n")


def queue2var(q, chans):
	ss, t = [], []
	s, t = q.get(0)
	for ch in chans:
		ss.append([s[ch]])
	# t = pd.Timestamp(t, unit ='s')
	ss = np.array(ss)
	ss = ss #.transpose()
	# print("the matrix:", ss, "\nshape:", ss.shape)
	return ss, t