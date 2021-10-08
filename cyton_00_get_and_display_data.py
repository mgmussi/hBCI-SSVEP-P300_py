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

if __name__ == '__main__':
	BoardShim.enable_dev_board_logger()
	params = BrainFlowInputParams ()
	params.serial_port = 'COM3' #COM11
	board_id = BoardIds.CYTON_BOARD.value
	board = BoardShim (board_id, params)

	chk_chs = [1,3,5] #available from 1 to 8
	fs = 250
	nyq = 0.5*fs
	low = 5 / nyq
	high = 50 / nyq
	# b, a = butter(order, [low, high], btype='band')
	b, a = butter(4, [low, high], btype='band')

	print('Looking for an EEG stream...')
	board.prepare_session() #connect to board
	print('Connected\n--\n')

	board.start_stream(45000, 'file://cyton_00_data.csv:w')
	print('Starting...')
	start = time.perf_counter() #time.time() #<for older python versions
	time.sleep(5)			#approx 1250 samples

	rawsamples = np.empty([len(chk_chs), 1])
	avgsamples = np.empty([len(chk_chs), 1])
	filtsamples = np.empty([len(chk_chs), 1])
	rawtimestamps = np.empty([1,1])
	
	###
	data = board.get_board_data()
	print('>>Samples [', data[chk_chs,:], ']\nTimestamp [', data[22,:], ']') #for list type
	
	rawsamples = np.array(data[chk_chs,:])
	rawtimestamps =  np.array(data[22,:])
	# if not rawsamples.size == 0:
	# 	rawsamples = np.hstack((rawsamples, ss))
	# else:
	# 	rawsamples = np.array(ss)
	#
	# if not rawtimestamps.size == 0:
	# 	rawtimestamps = np.append(rawtimestamps, tt)
	# else:
	# 	rawtimestamps = np.array(tt)

	## normalization
	if rawsamples.any():
		mean = np.mean(rawsamples, 1)
		avgsamples = np.subtract(rawsamples.transpose(), mean).transpose()
		maxx = np.max(avgsamples, 1)
		minn = np.min(avgsamples, 1)
		avgsamples = (np.subtract(avgsamples.transpose(), minn)/np.subtract(maxx, minn)).transpose() #(avg-min)/(max-min)

		filtsamples = np.array(lfilter(b, a, avgsamples, axis = 1))

	board.stop_stream()
	board.release_session()
	finish = time.perf_counter() #time.time() #<for older python versions

	print('\n___\n|Sizes: Samples [',rawsamples.shape,'], Timestamp [',rawtimestamps.shape,']')
	print('|Sizes: AVG [',avgsamples.shape,'], Filt\'d [',filtsamples.shape,']' ) #for numpy type
	print('|Sucessfully streamed in ',round(finish-start,3),'s!\n---')
	
	##data with timestamp
	plt.figure(figsize = (5,6), dpi = 100)
	# plt.plot(rawtimestamps, rawsamples.transpose(),'k')
	plt.plot(rawtimestamps, avgsamples.transpose(), 'red')
	plt.plot(rawtimestamps, filtsamples.transpose(), 'blue')

	plt.savefig('cyton_00_all_samples.png')
	plt.show()

	print("\n\n>>>DONE<<<\n\n")