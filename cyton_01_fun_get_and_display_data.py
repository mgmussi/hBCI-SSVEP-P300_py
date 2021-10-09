from pylsl import StreamInlet, resolve_stream
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
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes#, AggOperation

import warnings
warnings.filterwarnings("error")



class AcqThread(threading.Thread):
	def __init__(self, dataOutQ1, dataOutQ2, saveQ):
		threading.Thread.__init__(self)
		self.dataOutQ2 = dataOutQ2
		self.dataOutQ1 = dataOutQ1
		self.saveQ = saveQ

	def run(self):
		pass
		# pullSamples(self.q, self.raw, self.flag) #place where function is called

class P300Thread(threading.Thread):
	def __init__(self, dataInQ, featureQ):
		threading.Thread.__init__(self)
		self.dataInQ = dataInQ

	def run(self):
		pass
		#Here goes function for P300

class SSVEPThread(threading.Thread):
	def __init__(self, dataInQ, featureQ):
		threading.Thread.__init__(self)
		self.dataInQ = dataInQ

	def run(self):
		pass
		#Here goes function for SSVEP

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
#
AcqTh = AcqThread(DataOutQ1, DataOutQ2, SaveQ)
P300Th = P300Thread(DataOutQ1, FeatQ1)
SSVEPTh = SSVEPThread(DataOutQ2, FeatQ2)


def Acquisition():
	chk_chs = [1,3,5] #available channels [1 to 8]
	fs = 250
	nyq = 0.5*fs
	low = 5 / nyq
	high = 50 / nyq
	b, a = butter(4, [low, high], btype='band') # b, a = butter(order, [low, high], btype='band')

	rawsamples = np.array([])
	avgsamples = np.array([])
	filtsamples = np.array([])
	rawtimestamps = np.array([])

	i = 0
	start = time.perf_counter() #time.time() #<for older python versions

	while i<1250:
		time.sleep(0.5)
		print("Size of Ringbuffer: ", board.get_board_data_count())
		data = board.get_board_data()
		ss = np.array(data[chk_chs,:])
		tt = np.array(data[22,:])
		# print('>>Samples [', ss, ']\nTimestamp [', tt, ']')
		# print('|Shapes ss & tt:\n|>>Samples [',ss.shape,']\n>>|Timestamp [',tt.shape,']')
		if not rawsamples.size == 0:
			rawsamples = np.hstack((rawsamples, ss))
		else:
			rawsamples = np.array(ss)
		if not rawtimestamps.size == 0:
			rawtimestamps = np.append(rawtimestamps, tt)
		else:
			rawtimestamps = np.array(tt)
		# print('|Shapes rawss & ratt:\n|>>Samples [',rawsamples.shape,']\n>>|Timestamp [',rawtimestamps.shape,']')
		#
		## normalization
		mean = np.mean(rawsamples, 1)
		avgsamples = np.subtract(rawsamples.transpose(), mean).transpose()
		maxx = np.max(avgsamples, 1)
		minn = np.min(avgsamples, 1)
		try:
			avgsamples = (np.subtract(avgsamples.transpose(), minn)/np.subtract(maxx, minn)).transpose() #(avg-min)/(max-min)
		except RuntimeWarning:
			print('/////Caution::RAILED Channel(s)')

		filtsamples = np.array(lfilter(b, a, avgsamples, axis = 1))
		# print(f'|Sizes_{i}: Samples [{rawsamples.shape}] vs. AVG [{avgsamples.shape}] vs. Filt\'d [{filtsamples.shape}]' ) #for numpy type

		i = rawsamples.shape[1]

	board.stop_stream()
	board.release_session()
	finish = time.perf_counter() #time.time() #<for older python versions

	print('\n___\n|Sizes: Samples [',rawsamples.shape,'], Timestamp [',rawtimestamps.shape,']')
	print('|Sizes: AVG [',avgsamples.shape,'], Filt\'d [',filtsamples.shape,']' ) #for numpy type
	print('|Sucessfully streamed in ',round(finish-start,3),'s!\n---')

	plt.figure(figsize = (5,6), dpi = 100)

	##data without timestamp
	# plt.plot(rawsamples.transpose(),'k')
	# plt.plot(avgsamples.transpose(), 'red')

	##data with timestamp
	# plt.plot(rawtimestamps, rawsamples.transpose(),'k')
	plt.plot(rawtimestamps, avgsamples.transpose(), 'red')
	plt.plot(rawtimestamps, filtsamples.transpose(), 'blue')

	plt.savefig('cyton_01_all_samples.png')
	plt.show()

if __name__ == '__main__':
	BoardShim.enable_dev_board_logger()
	params = BrainFlowInputParams ()
	params.serial_port = 'COM11'
	board_id = BoardIds.CYTON_BOARD.value
	board = BoardShim (board_id, params)
	print('Looking for an EEG stream...')
	board.prepare_session() #connect to board
	print('Connected\n--')
	board.start_stream(45000, 'file://cyton_01_data.csv:w') #streams data and saves it to csv file
	print('Starting...')

	Acquisition()

	print("\n\n>>>DONE<<<\n\n")