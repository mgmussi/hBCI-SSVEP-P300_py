#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Used to create flashing boxes for the SSVEP paradigm
For 60Hz monitor:
MonFr/ DesFr = NumFr (NumFr/2 >> ON, NumFr/2 >> OFF)
60/30 = 2 (1 on, 1 off)
60/15 = 4 (2 on, 2 off)
60/10 = 6 ...
60/7.5 = 8
60/6 = 10
60/5 = 12

To control frames, use if(NFrame % NumFr >= NumFr/2) which will make cycle 50-50% ON-OFF
"""
from psychopy import core, visual
import threading

class FlickThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		FlickerStart()

def FlickerStart():

	#Create the Window
	win = visual.Window([853,695], [480,180], monitor = 'SSVEP Paradigm', color = 'black')
	win.flip()
	core.wait(1.0)

	message = visual.TextStim(win, text = 'Flickering routine\n\nReady?')
	message.draw()
	win.flip()
	core.wait(1.5)

	win.close()
	core.quit()


FlickTh = FlickThread()

if __name__ == '__main__':
	FlickTh.start()
	FlickTh.join()