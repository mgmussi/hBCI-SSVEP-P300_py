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
	global inlet
	global stopcollecting
	global rawsamples
	global filtsamples
	global rawtimestamps

	chk_chs = [1,3,5]
	fs = 256 #Hz
	nyq = 0.5*fs
	low = 5 / nyq
	high = 50 / nyq
	# b, a = butter(order, [low, high], btype='band')
	b, a = butter(4, [low, high], btype='band')

	print('Looking for an EEG stream...')
	streams = resolve_stream('type', 'EEG')
	inlet = StreamInlet(streams[0])
	print('Connected\n--')

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
		avgsamples = (np.subtract(avgsamples.transpose(), minn)/np.subtract(maxx, minn)).transpose() #(avg-min)/(max-min)

		# time.sleep(1.5)
		# input("Holding...")


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
	print(f'|Sucessfully streamed in {round(finish-start,3)}s!\n--')

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