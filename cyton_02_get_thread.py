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
			print('/////Caution::RAILED Channel(s)')
			
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

@profile
def Acquisition(dataOutQ1, dataOutQ2, saveQ):
	chk_chs = [1,3,5] #available channels [1 to 8]
	fs = 250 		#Hz
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
		time.sleep(1)
		ld = board.get_board_data_count() #last data
		data = board.get_board_data()
		ss = np.array(data[chk_chs,:])
		tt = np.array(data[22,:])

		if not rawsamples.size == 0:
			rawsamples = np.hstack((rawsamples, ss))
		else:
			rawsamples = np.array(ss)
		if not rawtimestamps.size == 0:
			rawtimestamps = np.append(rawtimestamps, tt)
		else:
			rawtimestamps = np.array(tt)

		##::First Branch:: SSVEP
		avgsamples = normalization(rawsamples[:,-ld:], i)
		filtsamples = np.array(lfilter(b, a, avgsamples, axis = 1))
		threadLock.acquire()				#queue locked to prevent other threads to access it
		dataOutQ2.put([filtsamples[:,-ld:], rawtimestamps[-ld:]]) 	#data is put in the queue
		# print('|Sending: <>Samples [',filtsamples[:,-ld:].shape,'] <>Timestamp [',rawtimestamps[-ld:].shape,']' ) #for numpy type
		threadLock.release()				#queue unlocked with info
		ESSVEP.set()

		##::Second Branch:: P300
		# threadLock.acquire()				
		# dataOutQ2.put([rawsamples,rawtimestamps])
		# threadLock.release()

		i = rawsamples.shape[1]

	E.set()
	threadLock.acquire()
	saveQ.put([rawsamples[:,:], rawtimestamps[:]])
	threadLock.release()
	print('\n\t\tATENTION E is set >>',ESSVEP.is_set(),'\n')
	finish = time.perf_counter() #time.time() #<for older python versions

	print('\n___\n|Sizes: Samples [',rawsamples.shape,'], Timestamp [',rawtimestamps.shape,']')
	print('|Sizes: AVG [',avgsamples.shape,'], Filt\'d [',filtsamples.shape,']' ) #for numpy type
	print('|Sucessfully streamed in ',round(finish-start,3),'s!\n---')


def P300fun(dataInQ, featureQ):
	p300sample = np.array([])
	p300timestamp = np.array([])
	while not E.is_set():
		if EP300.is_set():
			# while dataInQ.qsize(): # if not dataqueue.empty():
			while dataInQ.qsize(): # if not dataqueue.empty():
			# if not dataInQ.empty():
				try:
					p300sample, p300timestamp = dataInQ.get(0)
					# print('\n___\n<>P300\n>>Samples [', p300sample, ']\nTimestamp [', p300timestamp, ']') #for list type
				except Empty:
					return
			EP300.clear()
			p300sample = np.array([])
			p300timestamp = np.array([])
	print('P300 thread Finished')


def SSVEPfun(dataInQ, featureQ):
	ssvepsample = np.array([])
	ssveptimestamp = np.array([])
	while not E.is_set():
		if ESSVEP.is_set():
			while dataInQ.qsize():
				try:
					ssvepsample, ssveptimestamp = dataInQ.get(0)
					# print('\n___\n<>SSVEP\n>>Samples [',ssvepsample.shape,'] >>Timestamp [',ssveptimestamp.shape,']' )
				except Empty:
					print("except Emtpy works")
					return
			ESSVEP.clear()
			ssvepsample = np.array([])
			ssveptimestamp = np.array([])
	print('SSVEP thread Finished')


if __name__ == '__main__':
	BoardShim.enable_dev_board_logger()
	params = BrainFlowInputParams ()
	params.serial_port = 'COM3' #11'
	board_id = BoardIds.CYTON_BOARD.value
	board = BoardShim (board_id, params)
	print('Looking for an EEG stream...')
	board.prepare_session() #connect to board
	print('Connected\n--')
	board.start_stream(45000, 'file://cyton_02_data.csv:w') #streams data and saves it to csv file
	print('Starting...')

	AcqTh.start()
	P300Th.start()
	SSVEPTh.start()
	
	AcqTh.join()
	P300Th.join()
	SSVEPTh.join()

	board.stop_stream()
	board.release_session()

	while SaveQ.qsize():
		try:
			y, x = SaveQ.get(0)
		except Empty:
			pass
	# print('---\n',x, '\n__\n', y,'\n---')
	plt.plot(x,y.transpose(),'k')
	plt.savefig('cyton_02_all_samples.png')
	plt.show()

	print("\n\n>>>DONE<<<\n\n")