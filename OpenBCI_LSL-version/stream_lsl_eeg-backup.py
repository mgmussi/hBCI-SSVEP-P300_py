from pylsl import StreamInlet, resolve_stream
import tkinter as tk
import AppWindow as app
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import queue
import time
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter


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
		if self.threadID == 1:
			print("Starting ", self.name)
			pullSamples(self.q, self.raw, self.flag) #place where function is called
		elif self.threadID == 2:
			print("Starting ", self.name)
			printSamples(self.q, self.flag)
		else:
			print("Starting ", self.name)
			plotSamplesTH(self.q, self.flag, self.app)


#
#
#
#Thread elements
threadLock = threading.Lock()
dataqueue = queue.Queue()
filteredq = queue.Queue()
rawqueue = queue.Queue()
flag = queue.Queue()
flag.put(0)
thread1 = myThread(1,"Thread-1", dataqueue, rawqueue, flag)
# thread2 = myThread(2,"Thread-2", dataqueue,flag)
# thread3 = myThread(3,"Thread-3", dataqueue,flag,app)

#Golbal Variables declaration
rawsamples = np.empty([8,1])
filtsamples = np.empty([8,1])
rawtimestamps = np.empty([1,1])
prosamples = []
show_samples, show_timestamps = [],[]
stopcollecting = 0
kounter = 0
chk_chs = []

#
#
#


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
	app.btn_ReceiveEEG.config(state = tk.NORMAL)


def pullSamples(q, raw, flag): #THREAD #1
	global stopcollecting
	i = 0
	while i<500: #<<< in the future: "while flag.get() not 2"
		flag.put(0) #putting a value at every round because the other function will test it every round
		sample, timestamp = inlet.pull_sample()
		threadLock.acquire()	#thread locks to put info in the queue
		q.put([sample,timestamp]) #data is put in the queue for other threads to access
		# threadLock.release()	#thread unlocks after info is in
		# #
		# threadLock.acquire()
		raw.put([sample,timestamp]) #raw data
		threadLock.release()
		i += 1
	stopcollecting = 1
	flag.put(1)	#thread sends a last flag to indicat it is finished. Once the other thread unpacks the 1, it will know info stream is over
	# print("Exit flag pullSamples: ", stopcollecting)


def filtering(app, q):
	fs = 256 #Hz
	nyq = 0.5*fs
	low = 0.2 / nyq
	high = 40 / nyq
	global rawsamples
	global filtsamples
	# b, a = butter(order, [low, high], btype='band')
	b, a = butter(4, [low, high], btype='band')
	while dataqueue.qsize(  ): # if not dataqueue.empty():
		try:
			ss, ts = queue2var(dataqueue, chk_chs)
			if not rawsamples.size == 0:
				rawsamples = np.vstack((rawsamples, ss))
			else:
				rawsamples = np.array(ss)

			if not rawtimestamps.size == 0:
				rawtimestamps = np.append(rawtimestamps, ts)
			else:
				rawtimestamps = np.array(ts)

			sp.signal()
		except dataqueue.Empty:
			return
	for k in range(np.size(rawsamples, 1)):
		if not filtsamples.size == 0:
			filtsamples = np.append(filtsamples, lfilter(b, a, rawsamples[:,k]), axis = 1)
		else:
			filtsamples = lfilter(b, a, rasamples[:,k])

		threadLock.acquire()	#thread locks to put info in the queue
		q.put([filtsamples,rawtimestamp]) #data is put in the queue for other threads to access
		threadLock.release()
	# print(rawsamples)
	# print('\n')
	# print(rawtimestamps)
	# print('\n\n----')
	app.root.after(200, plotSamples, flag, app)


def FFTData(q, raw, flag): 	#should run in parallel with pullSamples (thread)
	if not flag.get(): 		#testing if stream reception stopped
		while dataqueue.qsize(  ): # if not dataqueue.empty():
			try:
				samples, timestamps = queue2var(dataqueue, chk_chs)
				
			except dataqueue.Empty:
				return


def plotRawSamples(flag, app): #Outside Threads
	# if not sftopcollecting: #testing if stream reception stopped
	global kounter
	# rawsamples = np.array([[]])
	# rawtimestamps = np.array([])
	while dataqueue.qsize(  ): # if not dataqueue.empty():
		try:
			ss, ts = queue2var(dataqueue, chk_chs)
			if not rawsamples.size == 0:
				rawsamples = np.vstack((rawsamples, ss))
			else:
				rawsamples = np.array(ss)
			if not rawtimestamps.size == 0:
				rawtimestamps = np.append(rawtimestamps, ts)
			else:
				rawtimestamps = np.array(ts)
			# show_samples = samples[-250:]
			# show_timestamps = timestamps[-250:]
			if kounter>0 and kounter % 15 == 0:
				# app.plotGraph(rawtimestamps[-250:], rawsamples[-250:])
				# app.plotGraph(rawtimestamps[-250:-1, :], rawsamples[-250:-1, :])
				if np.size(rawsamples,0)>=250:
					app.plotGraph(rawtimestamps[-250:-1], rawsamples[-250:-1,:])
				else:
					app.plotGraph(rawtimestamps, rawsamples)
			kounter += 1
		except dataqueue.Empty:
			return
	app.root.after(200, plotSamples, flag, app)


def plotFiltSamples(flag, app): #Outside Threads
	global kounter
	while filteredq.qsize(  ): # if not dataqueue.empty():
		try:
			ss, ts = queue2var(filteredq, chk_chs)
			if not filtplot.size == 0:
				filtplot = np.vstack((filtplot, ss))
			else:
				filtplot = np.array(ss)

			if not rawtimestamps.size == 0:
				rawtimestamps = np.append(rawtimestamps, ts)
			else:
				rawtimestamps = np.array(ts)

			if kounter>0 and kounter % 15 == 0:
				if np.size(rawsamples,0)>=250:
					app.plotGraph(rawtimestamps[-250:-1], rawsamples[-250:-1,:])
				else:
					app.plotGraph(rawtimestamps, rawsamples)
			kounter += 1
		except dataqueue.Empty:
			return
	app.root.after(200, plotSamples, flag, app)


def queue2var(q, chans):
	ss, t = [], []
	s, t = q.get(0)
	for ch in chans:
		ss.append(s[ch])
	# t = pd.Timestamp(t, unit ='s')
	ss = np.array(ss)
	return ss, t


def getEEGstream(app):
	app.logger.warn('|Starting threads...')
	app.logger.warn('|Pulling samples from selected channels:\n|%s', chk_chs)
	#
	# rawsamples = np.zeros([8,1])
	# print(rawsamples)
	# rawtimestamps = np.zeros([1,1])
	# filtsamples = np.array([])
	# filtplot = np.array([])
	# ss = np.array([])
	# ts = np.array([])
	# ss, ts = [], []
	start = time.perf_counter()
	thread1.start()
	##
	app.logger.warn('|Successfully started!')
	filtering(app, filteredq)
	plotFiltSamples(flag, app)
	# plotSamples(flag, app)
	##
	thread1.join()

	## CLOSE THREAD SOMEWHERE HERE?

	finish = time.perf_counter()
	#
	while rawqueue.qsize(): # if not dataqueue.empty():
		try:
			ss, ts = queue2var(rawqueue, chk_chs)
			if not rawsamples.size == 0:
				rawsamples = np.vstack((rawsamples, ss))
			else:
				rawsamples = np.array(ss)
			if not rawtimestamps.size == 0:
				rawtimestamps = np.append(rawtimestamps, ts)
			else:
				rawtimestamps = np.array(ts)
		except dataqueue.Empty:
			pass

	# print(rawsamples)
	# print(rawtimestamps)
	print("\n\n>>>DONE<<<\n\n")
	app.logger.warn(f'|Sizes: Samples [{len(rawsamples)}, {len(rawsamples[0])}], {len(rawtimestamps)} timestamps')
	app.logger.warn(f'|Sucessfully streamed in {round(finish-start,3)}s!\n--')



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