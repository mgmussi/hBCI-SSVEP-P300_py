import numpy as np
import multiprocessing as mp
import random
from pylsl import StreamInlet, resolve_stream
import time
import threading

class myThread(threading.Thread):
	def __init__(self, threadID, name):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
	def run(self):
		print("Starting {}", self.name)
		# threadLock.acquire()
		pullSamples(self.threadID)
		# threadLock.release()

def pullSamples():
	i = 0
	while i<10:
		sample, timestamp = inlet.pull_sample()
		samples.append(sample)
		timestamps.append(timestamp)
		i += 1
	return sample, timestamp

if __name__ == '__main__':
	print("Finding connection...")
	streams = resolve_stream('type', 'EEG')
	inlet = StreamInlet(streams[0])
	print("Done\n")
	#
	samples,timestamps = [],[]
	#
	#
	print("Starting Timer!")
	start = time.perf_counter()
	pullSamples()
	finish = time.perf_counter()
	#
	#
	print(f'Sizes: Samples [{len(samples)}, {len(samples[0])}], {len(timestamps)} timestamps')
	print(f'Sucessfully streamed in {round(finish-start,3)}s!')

