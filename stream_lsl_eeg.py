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


class myThread(threading.Thread):
	def __init__(self, threadID, name, q, raw, flag, app = float("NaN")):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.q = q
		self.raw = raw
		self.flag = flag
		self.app = app

	def run(self):
		print("Starting ", self.name)
		pullSamples(self.q, self.raw, self.flag) #place where function is called




#Thread elements
threadLock = threading.Lock()
dataqueue = queue.Queue()
filteredq = queue.Queue()
rawqueue = queue.Queue()
flag = queue.Queue()
flag.put(0)
thread1 = myThread(1,"Thread-1", dataqueue, rawqueue, flag)

#Golbal Variables declaration
rawsamples = []
filtsamples = []
rawtimestamps = np.empty([1,1])
prosamples = []
show_samples, show_timestamps = [],[]
stopcollecting = 0
kounter = 0
chk_chs = []


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
		app.btn_StartEEGLSL['state'] = tk.DISABLED
		app.toggleLock()


def getInlet(app):
	global inlet
	app.logger.warn('Looking for an EEG strean...')
	streams = resolve_stream('type', 'EEG')
	inlet = StreamInlet(streams[0])
	app.logger.warn('Connected\n--')
	app.btn_chconf.config(state = tk.DISABLED)
	app.btn_StartEEGLSL.config(state = tk.DISABLED)
	app.btn_ReceiveEEG.config(state = tk.NORMAL)
	# app.line, = plt.plot([], [])



def pullSamples(q, raw, flag): #THREAD #1
	global stopcollecting
	i = 0
	while i<1500: #<<< in the future: "while flag.get() not 2"
		flag.put(0) #putting a value at every round because the other function will test it every round
		sample, timestamp = inlet.pull_sample()
		threadLock.acquire()			#thread loacks to put info in the queue
		q.put([sample,timestamp]) 		#data is put in the queue for other threads to access
		raw.put([sample,timestamp])		#raw data
		threadLock.release()			#thread unlocks after info is in
		i += 1
	stopcollecting = 1
	flag.put(1)	#thread sends a last flag to indicat it is finished. Once the other thread unpacks the 1, it will know info stream is over


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


@profile
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
					app.plotGraphS(rawtimestamps[-250:-1], rawsamples[:,-250:-1].transpose()) #for numpy
				else:
					app.plotGraphS(rawtimestamps, rawsamples.transpose()) #for lists
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
