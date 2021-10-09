import numpy as np
import multiprocessing as mp
import random
from pylsl import StreamInlet, resolve_stream
import time
import threading
import queue

class myThread(threading.Thread):
	def __init__(self, threadID, name, q, f):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.q = q
		self.f = f

	def run(self):
		if self.threadID == 1:
			print("Starting ", self.name)
			pullSamples(self.q, self.f) #place where function is called
		else:
			print("Starting ", self.name)
			printSamples(self.q, self.f)


def pullSamples(q,flag):
	i = 0
	while i<3000:
		flag.put(0) #putting a value at every round because the other function will test it every round
		sample, timestamp = inlet.pull_sample()
		threadLock.acquire()	#thread locks to put info in the queue
		q.put([sample,timestamp])
		# print("Sample", i, " in")
		# print(sample)
		# samples.append(sample)
		# timestamps.append(timestamp)
		threadLock.release()	#thread unlocks after info is in
		i += 1
	flag.put(1)	#thread sends a last flag to indicat it is finished. Once the other thread unpacks the 1, it will know info stream is over
	print("Exit flag on")
	return sample, timestamp

def printSamples(q,flag):
	i = 0
	while not flag.get(): #testing the flag every round
		threadLock.acquire() #thread locks to get info from the queue
		if not q.empty():
			sample, timestamp = q.get()
			print("Sample ", i, ": ",sample,timestamp)
			samples.append(sample)
			timestamps.append(timestamp)
			threadLock.release() #thread unlocks after info is out
			i += 1
		else:
			threadLock.release()
	return

if __name__ == '__main__':
	print("Finding connection...")
	streams = resolve_stream('type', 'EEG')
	inlet = StreamInlet(streams[0])
	print("Done\n")
	#
	# global exit_flag
	# exit_flag = 0
	threadLock = threading.Lock()
	dataqueue = queue.Queue()
	flag = queue.Queue()
	flag.put(0)
	thread1 = myThread(1,"Thread-1", dataqueue,flag)
	thread2 = myThread(2,"Thread-2", dataqueue,flag)
	#
	samples,timestamps = [],[]
	#
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
	print(f'Sucessfully streamed in {round(finish-start,3)}s!')