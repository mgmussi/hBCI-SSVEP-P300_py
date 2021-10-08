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
import numpy as np
# import AppWindow as app
import random
import time

class FlickeringBoxes:
	def __init__(self, position):
		#Definitions
		self.H = 720-25 #window hight (to fit side of the window)
		self.W = 853 #window width
		# self.H4 = np.floor(H/4)
		# self.W4 = np.floor(W/4)
		self.sS = 0.25			#square size
		self.pS = self.sS*1.35	#padding size
		self.p3S = self.sS*1.75	#p300 size

		# app.logger.warn('\nInitializing Flickering...')
		#Create the Window
		self.win = visual.Window([self.W,self.H], position, monitor = 'SSVEP Paradigm', color = 'black')
		self.win.flip()

	def start(self):
		p300clock = core.Clock()
		genclock = core.Clock()
		core.wait(1.0)
		self.message = visual.TextStim(self.win, text = 'Flickering routine\n\nReady?')
		self.message.draw()
		self.win.flip()
		core.wait(1.5)

		self.message = visual.TextStim(self.win, text = '3')
		self.message.draw()
		self.win.flip()
		core.wait(1.0)

		self.message = visual.TextStim(self.win, text = '2')
		self.message.draw()
		self.win.flip()
		core.wait(1.0)

		self.message = visual.TextStim(self.win, text = '1')
		self.message.draw()
		self.win.flip()
		core.wait(1.0)

		# T0 = time.time()
		T0 = genclock.getTime()
		p =[[-0.5, -0.5],[0, 0.5],[0.5, -0.5]] #from the centre; 1 is for 100%, 0 is for 50% and -1 is for 0% of screen
		self.stSq0 = visual.Rect(self.win, fillColor = 'black', lineColor = 'black', width = self.pS, height = self.pS, pos = p[0])
		self.stSq1 = visual.Rect(self.win, fillColor = 'black', lineColor = 'black', width = self.pS, height = self.pS, pos = p[1])
		self.stSq2 = visual.Rect(self.win, fillColor = 'black', lineColor = 'black', width = self.pS, height = self.pS, pos = p[2])

		self.Sq0 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.sS, height = self.sS, pos = p[0], opacity = 1)
		self.Sq1 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.sS, height = self.sS, pos = p[1])
		self.Sq2 = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.sS, height = self.sS, pos = p[2])

		self.Sq0.draw()
		self.Sq1.draw()
		self.Sq2.draw()
		self.stSq0.draw()
		self.stSq1.draw()
		self.stSq2.draw()

		self.win.flip()
		core.wait(0.5)

		ct = 0
		mx_ct = 1799
		op_off = 0.25
		op_on = 1

		kt = 0
		endflag = False
		
		selection = []
		for l in range (0,3):
			for k in range(0,12):
				selection.append(l)
		random.shuffle(selection)
		select = selection.pop(0)

		# t0 = time.time()
		t0 = p300clock.getTime()
		while not endflag:
			try:
				if ct%2 >= 1:
					self.Sq0 = visual.Rect(self.win, fillColor = '#00388A', lineColor = '#00388A',  width = self.sS, height = self.sS, pos = p[0], opacity = op_on) #blue
				else:
					self.Sq0 = visual.Rect(self.win, fillColor = '#59FF00', lineColor = '#59FF00', width = self.sS, height = self.sS, pos = p[0], opacity = op_on) #green

				if ct%4 >= 2:
					self.Sq1 = visual.Rect(self.win, fillColor = '#59FF00', lineColor = '#59FF00', width = self.sS, height = self.sS, pos = p[1], opacity = op_on) #green
				else:
					self.Sq1 = visual.Rect(self.win, fillColor = '#B80117', lineColor = '#B80117', width = self.sS, height = self.sS, pos = p[1], opacity = op_on) #red

				if ct%6 >= 3:
					self.Sq2 = visual.Rect(self.win, fillColor = '#59FF00', lineColor = '#59FF00', width = self.sS, height = self.sS, pos = p[2], opacity = op_on) #green
				else:
					self.Sq2 = visual.Rect(self.win, fillColor = '#B80117', lineColor = '#B80117', width = self.sS, height = self.sS, pos = p[2], opacity = op_on) #red

				# t1 = time.time()
				t1 = p300clock.getTime()
				print(t1-t0)
				if t1-t0 < 0.2:
					if select == 0:
						self.p300Sq = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.p3S, height = self.p3S, pos = p[0], opacity = op_on)
					elif select == 1:
						self.p300Sq = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.p3S, height = self.p3S, pos = p[1], opacity = op_on)
					elif select == 2:
						self.p300Sq = visual.Rect(self.win, fillColor = 'white', lineColor = 'white', width = self.p3S, height = self.p3S, pos = p[2], opacity = op_on)
				elif t1-t0 >= 0.2 and t1-t0 < 0.45:
					self.p300Sq = visual.Rect(self.win, fillColor = 'black', lineColor = 'black', width = self.p3S, height = self.p3S, pos = p[0], opacity = op_off)
				elif t1-t0 >= 0.5:
					# t0 = time.time()
					t0 = p300clock.getTime()
					try:
					    select = selection.pop(0)
					except IndexError as e:
					    endflag = True

				self.p300Sq.draw()
				self.stSq0.draw()
				self.stSq1.draw()
				self.stSq2.draw()
				self.Sq0.draw()
				self.Sq1.draw()
				self.Sq2.draw()

				self.win.flip()
				ct += 1
				kt += 1
				if ct>= mx_ct:
					ct = 0

			except KeyboardInterrupt:
				self.stop()

		# T1 = time.time()
		T1 = genclock.getTime()
		print('Whole process took',T1-T0,'s')

	def stop(self):
		self.win.close()
		core.quit()

if __name__ == '__main__':
	w = 1920/2
	h = 720/2
	x = (w/2)
	y = (h/2)
	FlkWin = FlickeringBoxes([x,y])
	FlkWin.start()