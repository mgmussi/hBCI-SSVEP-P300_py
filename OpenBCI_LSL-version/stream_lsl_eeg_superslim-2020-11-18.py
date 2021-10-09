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
# import pyqtgraph as pg


class AcqThread(threading.Thread):
	def __init__(self, dataOutQ1, dataOutQ2, stopQ1, stopQ2, saveQ):
		threading.Thread.__init__(self)
		self.stopQ2 = stopQ2
		self.stopQ1 = stopQ1
		self.dataOutQ2 = dataOutQ2
		self.dataOutQ1 = dataOutQ1
		self.saveQ = saveQ

	def run(self):
		pass
		# pullSamples(self.q, self.raw, self.flag) #place where function is called

class P300Thread(threading.Thread):
	def __init__(self, dataInQ, featureQ, stopQ):
		threading.Thread.__init__(self)
		self.stopQ = stopQ
		self.dataInQ = dataInQ

	def run(self):
		pass
		#Here goes function for P300

class SSVEPThread(threading.Thread):
	def __init__(self, dataInQ, featureQ, stopQ):
		threading.Thread.__init__(self)
		self.stopQ = stopQ
		self.dataInQ = dataInQ

	def run(self):
		pass
		#Here goes function for SSVEP

class ClassifThread(threading.Thread):
	def __init__(self, dataOutQ):
		threading.Thread.__init__(self)
		self.stopQ = stopQ
		self.dataOutQ = dataOutQ

	def run(self):
		pass
		#Here goes function for P300


threadLock = threading.Lock()
SaveQ = queue.Queue()
DataOutQ1 = queue.Queue()
DataOutQ2 = queue.Queue()
StopQ1 = queue.Queue()
StopQ2 = queue.Queue()
FeatQ1 = queue.Queue()
FeatQ2 = queue.Queue()
StopQ1.put(0)
StopQ2.put(0)
#
AcqTh = AcqThread(DataOutQ1, DataOutQ2, StopQ1, StopQ2, SaveQ)
P300Th = P300Thread(DataOutQ1, FeatQ1, StopQ1)
SSVEPTh = SSVEPThread(DataOutQ2, FeatQ2, StopQ2)


def Acquisition(inlet):
	global rawsamples
	global avgsamples
	global filtsamples
	global rawtimestamps
	#include Queue or Variable that will indicate timestapm where "ON" trigger occurs (to start/end sample acquisition)

	chk_chs = [0,3,5] #available channels [0 to 7]
	fs = 256 #Hz
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
		sample, timestamp = inlet.pull_sample()
		# sample, timestamp = inlet.pull_chunk()
		# print('>>Samples [', sample, ']\nTimestamp [', timestamp, ']') #for list type
		
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

		## normalization
		mean = np.mean(rawsamples, 1)
		avgsamples = np.subtract(rawsamples.transpose(), mean).transpose()
		maxx = np.max(avgsamples, 1)
		minn = np.min(avgsamples, 1)
		try:
			avgsamples = (np.subtract(avgsamples.transpose(), minn)/np.subtract(maxx, minn)).transpose() #(avg-min)/(max-min)
		except:
			print(f'Mean vals:\n{mean}\nAvgSample:\n{avgsamples}\nMaxx:\n{maxx}\nMinn:\n{minn}')
			input("Please, evaluate inputs and hit \'Enter\' to continue")


		# if i>0 and i % 125 == 0:
		# 	if not filtsamples.size == 0:
		# 		filtsamples = np.hstack((filtsamples, lfilter(b, a, avgsamples, axis = 1)))
		# 	else:
		# 		filtsamples = np.array(lfilter(b, a, avgsamples, axis = 1))
		filtsamples = np.array(lfilter(b, a, avgsamples, axis = 1))
		# print(f'|Sizes_{i}: Samples [{rawsamples.shape}] vs. AVG [{avgsamples.shape}] vs. Filt\'d [{filtsamples.shape}]' ) #for numpy type

		if i == 1:
			##REMOVE FIRST SAMPLE (trash data/spike):
			rawsamples = rawsamples[:,1:]
			avgsamples = avgsamples[:,1:]
			filtsamples = filtsamples[:,1:]
			rawtimestamps = rawtimestamps[1:]
		i += 1
	###
	# print('><>< my i equals ', i, '><><')
	finish = time.perf_counter()

	print(f'\n___\n|Sizes: Samples [{rawsamples.shape}], Timestamp [{rawtimestamps.shape}]' ) #for numpy type
	print(f'|Sizes: AVG [{avgsamples.shape}], Filt\'d [{filtsamples.shape}]' ) #for numpy type
	print(f'|Sucessfully streamed in {round(finish-start,3)}s!\n---')

	plt.figure(figsize = (5,6), dpi = 100)

	##data without timestamp
	# plt.plot(rawsamples.transpose(),'k')
	# plt.plot(avgsamples.transpose(), 'red')

	##data with timestamp
	# plt.plot(rawtimestamps, rawsamples.transpose(),'k')
	# plt.plot(rawtimestamps, avgsamples.transpose(), 'red')
	plt.plot(rawtimestamps, filtsamples.transpose(), 'blue')

	plt.savefig('all_samples.png')
	plt.show()
	# print(rawsamples, '\n_____')
	# print(rawtimestamps, '\n_____\n_____')


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
# ##TRIED implementing the filter, but graph became a line... trying to recharge batteries to see if it solves the problem.
if __name__ == '__main__':
	# global inlet
	print('Looking for an EEG stream...')
	streams = resolve_stream('type', 'EEG')
	inlet = StreamInlet(streams[0])
	print('Connected\n--')

	Acquisition(inlet)

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