from pylsl import StreamInlet, resolve_stream
import tkinter as tk
import AppWindow as app
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


def assignChan(app, chk_var):
	if app.btn_chconf['text'] == 'Confirm Channels':
		global chk_chs
		chk_chs = []
		selchan = []
		for i in range(8):
			if chk_var[i].get():
				chk_chs.append(i)
		if not chk_chs:
			tk.messagebox.showwarning(title = 'No channel selected', message = 'Please select at least one channel')
			return
		for ch in chk_chs:
			selchan.append(ch + 1)
		app.logger.warn('Selected channels: %s\n--', selchan)
		print(chk_chs)
		app.btn_chconf['text'] = 'Change Channels'
		app.btn_StartEEGLSL['state'] = tk.NORMAL
		app.toggleLock()

	elif app.btn_chconf['text'] == 'Change Channels':
		app.btn_chconf['text'] = 'Confirm Channels'
		# app.btn_StartEEGLSL['state'] = tk.DISABLED
		app.toggleLock()

@profile
##TRIED implementing the filter, but graph became a line... trying to recharge batteries to see if it solves the problem.
def getInlet(app):
	global inlet
	global stopcollecting
	global rawsamples
	global filtsamples
	global rawtimestamps

	fs = 256 #Hz
	nyq = 0.5*fs
	low = 0.2 / nyq
	high = 100 / nyq
	# b, a = butter(order, [low, high], btype='band')
	b, a = butter(4, [low, high], btype='band')

	app.logger.warn('Looking for an EEG strean...')
	streams = resolve_stream('type', 'EEG')
	inlet = StreamInlet(streams[0])
	app.logger.warn('Connected\n--')
	app.btn_chconf.config(state = tk.DISABLED)
	app.btn_StartEEGLSL.config(state = tk.DISABLED)
	app.btn_ReceiveEEG.config(state = tk.NORMAL)

	rawsamples = np.empty([len(chk_chs), 1])
	avgsamples = np.empty([len(chk_chs), 1])
	rawtimestamps = np.empty([1,1])

	i = 0
	start = time.perf_counter()
	###
	while i<5000:
		sample, timestamp = inlet.pull_sample()
		# print('Samples [', sample, ']\nTimestamp [', timestamp, ']\n') #for list type
		ss = []
		for ch in chk_chs:
			ss.append([sample[ch]])
		# t = pd.Timestamp(t, unit ='s')
		ss = np.array(ss)
		if not rawsamples.size == 0:
			rawsamples = np.hstack((rawsamples, ss))
		else:
			rawsamples = np.array(sample)

		if not rawtimestamps.size == 0:
			rawtimestamps = np.append(rawtimestamps, timestamp)
		else:
			rawtimestamps = np.array(timestamp)

		## normalization
		mean = np.mean(rawsamples)
		avgsamples = rawsamples - mean
		maxx = np.max(avgsamples)
		minn = np.min(avgsamples)
		avgsamples = (avgsamples - minn)/(maxx - minn)


		if i>0 and i % 125 == 0:
			filtsamples = np.array(lfilter(b, a, avgsamples, axis = 1))
			if np.size(rawsamples,1)>=1000:
				# app.plotGraph(rawtimestamps[-250:-1], rawsamples[:,-250:-1].transpose())
				# app.plotGraph(rawtimestamps[-250:-1], filtsamples[:,-250:-1].transpose(), 'red')

				app.plotGraphS(rawtimestamps[-1000:-1], avgsamples[:,-1000:-1].transpose(),
					rawtimestamps[-1000:-1], filtsamples[:,-1000:-1].transpose())

				# app.plotGraphS(rawtimestamps[-250:-1], avgsamples[:,-250:-1].transpose(),
				# 	rawtimestamps[-250:-1], filtsamples[:,-250:-1].transpose())
			else:
				# app.plotGraph(rawtimestamps, rawsamples.transpose())
				# app.plotGraph(rawtimestamps, filtsamples.transpose(), 'red')

				app.plotGraphS(rawtimestamps, avgsamples.transpose(), rawtimestamps, filtsamples.transpose())
		i += 1
	###
	finish = time.perf_counter()
	app.logger.warn(f'|Sizes: Samples [{rawsamples.shape}], Timestamp [{rawtimestamps.shape}]' ) #for numpy type
	app.logger.warn(f'|Sucessfully streamed in {round(finish-start,3)}s!\n--')


def filtering(q, app):
	fs = 256 #Hz
	nyq = 0.5*fs
	low = 0.2 / nyq
	high = 40 / nyq
	global rawsamples
	global filtsamples
	global rawtimestamps
	rawsamples = np.empty([len(chk_chs),1])
	filtsamples = np.empty([len(chk_chs),1])

	# b, a = butter(order, [low, high], btype='band')
	b, a = butter(4, [low, high], btype='band')
	# print("filter param: ", b, ",", a)
	while dataqueue.qsize(): # if not dataqueue.empty():
		try:
			ss, ts = queue2var(dataqueue, chk_chs)
			if not rawsamples.size == 0:
				rawsamples = np.hstack((rawsamples, ss))
			else:
				rawsamples = np.array(ss)
			# print(rawsamples)

			if not rawtimestamps.size == 0:
				rawtimestamps = np.append(rawtimestamps, ts)
			else:
				rawtimestamps = np.array(ts)

			# if rawsamples.shape[1] >= 64: #64 samples = 0,25s
			if rawsamples.shape[1] % 15:
				plotsamples = []
				# #only 64 samples
				# filtsamples = np.array(lfilter(b, a, rawsamples[:,-64:], axis = 1))
				# filtsamples = filtsamples.transpose()
				# plotsamples = rawsamples[:,-64:].transpose()
				# app.plotGraph(rawtimestamps[-64:], plotsamples[:,-64:])
				# app.plotGraph(rawtimestamps[-64:], filtsamples[:,-64:])

				#all samples
				filtsamples = np.array(lfilter(b, a, rawsamples, axis = 1))
				filtsamples = filtsamples.transpose()
				plotsamples = rawsamples.transpose()
				app.plotGraph(rawtimestamps, plotsamples)
				# app.plotGraph(rawtimestamps, list(filtsamples))


				# filtsamples = flt #should I append? Or use to process online?
				# print("\t\tSIZE 1 RAWSAMP:",np.size(rawsamples, 1))
				## print("\n\nFiltSamples size:",filtsamples.shape)
				## print("Rawsamples size:",plotsamples.shape)
				## print("Rawtimesize size:",rawtimestamps.shape)
				# print("RAW:",rawsamples)
				# print("\n\nFLT size:",flt.shape)
				# print("FLT:", flt)
				# print("FLTARR:",filtsamples)
				threadLock.acquire()	#thread locks to put info in the queue
				q.put([filtsamples,rawtimestamps]) #data is put in the queue for other threads to access
				threadLock.release()
		except Empty:
			return
	app.root.after(200, filtering, flag, app)



def plotRawSamples(flag, app): #Outside Threads
	global rawsamples
	global rawtimestamps
	rawsamples = np.empty([len(chk_chs),1])
	filtsamples = np.empty([len(chk_chs),1])
	# if not sftopcollecting: #testing if stream reception stopped
	global kounter
	# rawsamples = np.array([[]])
	# rawtimestamps = np.array([])
	while dataqueue.qsize(  ): # if not dataqueue.empty():
		try:
			ss, ts = queue2var(dataqueue, chk_chs)
			if not rawsamples.size == 0:
				rawsamples = np.hstack((rawsamples, ss))
			else:
				rawsamples = np.array(ss)
			if not rawtimestamps.size == 0:
				rawtimestamps = np.append(rawtimestamps, ts)
			else:
				rawtimestamps = np.array(ts)
			if kounter>0 and kounter % 15 == 0:
				# print("Samples: |", rawsamples.shape,"| , | Timestamps:", rawtimestamps.shape, "|")
				if np.size(rawsamples,1)>=250:
					# app.plotGraph(rawtimestamps[-250:-1], rawsamples[-250:-1,:]) #for lists
					app.plotGraph(rawtimestamps[-250:-1], rawsamples[:,-250:-1].transpose()) #for numpy
				else:
					app.plotGraph(rawtimestamps, rawsamples.transpose()) #for lists
			kounter += 1
		except Empty:# except dataqueue.Empty:
			return
	app.root.after(200, plotRawSamples, flag, app)


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


def getEEGstream(app):
	app.logger.warn('|Starting threads...')
	start = time.perf_counter()
	thread1.start()
	##
	# app.logger.warn('|Started...')
	# filtering(filteredq, app)
	# plotFiltSamples(flag, app)
	plotRawSamples(flag, app)
	##
	thread1.join()

## CLOSE THREAD SOMEWHERE HERE?

	finish = time.perf_counter()
	#
	# while rawqueue.qsize(): # if not dataqueue.empty():
	# 	try:
	# 		ss, ts = queue2var(rawqueue, chk_chs)
	# 		if not rawsamples.size == 0:
	# 			rawsamples = np.vstack((rawsamples, ss))
	# 		else:
	# 			rawsamples = np.array(ss)
	# 		if not rawtimestamps.size == 0:
	# 			rawtimestamps = np.append(rawtimestamps, ts)
	# 		else:
	# 			rawtimestamps = np.array(ts)
	# 	except dataqueue.Empty:
	# 		pass
	# print(rawsamples)
	# print(rawtimestamps)
	# app.logger.warn(f'|Sizes: Samples [{len(rawsamples)}, {len(rawsamples[0])}], Timestamp {len(rawtimestamps)} ') #for list type
	app.logger.warn(f'|Sizes: Samples [{rawsamples.shape}], Timestamp [{rawtimestamps.shape}]') #for numpy type
	app.logger.warn(f'|Sucessfully streamed in {round(finish-start,3)}s!\n--')
	print("\n\n>>>DONE<<<\n\n")




if __name__ == '__main__':
	print(">>Called as \'__main__\'<<\n")
	print("Finding connection...")
	streams = resolve_stream('type', 'EEG')
	inlet = StreamInlet(streams[0])
	print("Done\n")
	#
	print("Starting Timer!")
	start = time.perf_counter()
	thread1.start()
	thread2.start()

	thread1.join()
	thread2.join()
	print("Threads Joined")
	finish = time.perf_counter()
	#
	#
	print(f'Sizes: Samples [{len(samples)}, {len(samples[0])}], {len(timestamps)} timestamps')
	print(f'Sucessfully streamed in {round(finish-start,3)}s!\n--')

else:
	print('stream_lsl_eeg loaded')



## BACKUP #########################################

# def printSamples(q,flag): #THREAD #2
# 	i = 0
# 	while not flag.get(): #testing the flag every round
# 		threadLock.acquire() #thread locks to get info from the queue
# 		if not q.empty():
# 			sample, timestamp = q.get()
# 			print("Sample ", i, ": ",sample,timestamp)
# 			samples.append(sample)
# 			timestamps.append(timestamp)
# 			threadLock.release() #thread unlocks after info is out
# 			i += 1
# 		else:
# 			threadLock.release()
# 	return

# def filtering(q, app): #acquiring and filtering in different processes
# 	fs = 256 #Hz
# 	nyq = 0.5*fs
# 	low = 0.2 / nyq
# 	high = 40 / nyq
# 	global rawsamples
# 	global filtsamples
# 	global rawsamples
# 	global filtsamples
# 	global rawtimestamps
# 	rawsamples = np.empty([len(chk_chs),1])
# 	filtsamples = np.empty([len(chk_chs),1])
# 	# b, a = butter(order, [low, high], btype='band')
# 	b, a = butter(4, [low, high], btype='band')
# 	# print("filter param: ", b, ",", a)
# 	while dataqueue.qsize(): # if not dataqueue.empty():
# 		try:
# 			ss, ts = queue2var(dataqueue, chk_chs)
# 			if not rawsamples.size == 0:
# 				rawsamples = np.hstack((rawsamples, ss))
# 			else:
# 				rawsamples = np.array(ss)
# 			# print(rawsamples)

# 			if not rawtimestamps.size == 0:
# 				rawtimestamps = np.append(rawtimestamps, ts)
# 			else:
# 				rawtimestamps = np.array(ts)				
# 			# print(rawtimestamps)
# 			# sp.signal()
# 		except Empty:
# 			return
# 	print("\t\tSIZE 1 RAWSAMP:",np.size(rawsamples, 1))
# 	for k in range(np.size(rawsamples, 1)):
# 		print("\n\nFiltSamples size:",filtsamples.shape)
# 		# if not filtsamples.shape[1] == 1:
# 		# print("Got into ==1!")
# 		print("RAW:",rawsamples)
# 		flt = np.array([lfilter(b, a, rawsamples[:,k])])
# 		flt = flt.transpose()
# 		print("FLT:", flt)
# 		filtsamples = np.append(filtsamples, flt, axis = 1)
# 		print("FLTARR:",filtsamples)
# 		# else:
# 		# 	print("Got into ELSE!")
# 		# 	print("RAW:",rawsamples)
# 		# 	flt = np.array([lfilter(b, a, rawsamples[:,k])])
# 		# 	flt = flt.transpose()
# 		# 	filtsamples = np.array(lfilter(b, a, flt))
# 		threadLock.acquire()	#thread locks to put info in the queue
# 		q.put([filtsamples,rawtimestamps]) #data is put in the queue for other threads to access
# 		threadLock.release()
# 	# print(rawsamples)
# 	# print('\n')
# 	# print(rawtimestamps)
# 	# print('\n\n----')
# 	app.root.after(200, filtering, flag, app)
