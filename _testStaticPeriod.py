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
		self.genclock = core.MonotonicClock()
		self.eachclock = core.MonotonicClock()
		self.H = 720-25 #window hight (to fit side of the window)
		self.W = 853 #window width
		# self.H4 = np.floor(H/4)
		# self.W4 = np.floor(W/4)
		self.sS = 0.125			#square size
		self.pS = self.sS*1.35	#padding size
		self.p3S = self.sS*1.75	#p300 size
		self.p =[[-0.5, -0.5],[0, 0.5],[0.5, -0.5]] #from the centre; 1 is for 100%, 0 is for 50% and -1 is for 0% of screen
		vert = [[1,1],[1,-1],[-1,-1],[-1,1]]

		# app.logger.warn('\nInitializing Flickering...')
		#Create the Window
		self.win = visual.Window([self.W,self.H], position, monitor = 'SSVEP Paradigm', color = 'black')

		self.SqP = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.p3S, pos = self.p[2], autoDraw = True)
		self.St0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = self.p[0], autoDraw = True)
		self.St1 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = self.p[1], autoDraw = True)
		self.St2 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = self.p[2], autoDraw = True)
		self.Sq0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = self.p[0], autoDraw = True)
		self.Sq1 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = self.p[1], autoDraw = True)
		self.Sq2 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = self.p[2], autoDraw = True)

		self.SqP.draw()
		self.St0.draw()
		self.St1.draw()
		self.St2.draw()
		self.Sq0.draw()
		self.Sq1.draw()
		self.Sq2.draw()

		self.win.flip()
		core.wait(5)

	def start(self):
		ISI = core.StaticPeriod(screenHz = 240)
		sequenceP300 = []
		for m in range (0,3):
			for k in range(0,3): # INCREASE TO 7 #number of blinks per square
				sequenceP300.append(m)
		random.shuffle(sequenceP300)

		T0 = self.genclock.getTime()
		counter = 1
		for sel in sequenceP300:
			self.SqP.setPos(self.p[sel])
			for cyc in range(4):
				for stg in range(8):
					t0 = self.eachclock.getTime()
					ISI.start(0.05)

					if cyc > 2:
						self.SqP.setOpacity(0)
					else:
						self.SqP.setOpacity(1)

					if stg == 0:
						self.Sq0.setColor('#00388A') #0
						self.Sq1.setColor('#59FF00') #0
						self.Sq2.setColor('#59FF00') #0
					elif stg == 1:
						self.Sq0.setColor('#59FF00') #1
						self.Sq1.setColor('#59FF00') #0
						self.Sq2.setColor('#59FF00') #0
					elif stg == 2:
						self.Sq0.setColor('#00388A') #0
						self.Sq1.setColor('#B80117') #1
						self.Sq2.setColor('#59FF00') #0
					elif stg == 3:
						self.Sq0.setColor('#59FF00') #1
						self.Sq1.setColor('#B80117') #1
						self.Sq2.setColor('#59FF00') #0
					elif stg == 4:
						self.Sq0.setColor('#00388A') #0
						self.Sq1.setColor('#59FF00') #0
						self.Sq2.setColor('#B80117') #1
					elif stg == 5:
						self.Sq0.setColor('#59FF00') #1
						self.Sq1.setColor('#59FF00') #0
						self.Sq2.setColor('#B80117') #1
					elif stg == 6:
						self.Sq0.setColor('#00388A') #0
						self.Sq1.setColor('#B80117') #1
						self.Sq2.setColor('#B80117') #1
					elif stg == 7:
						self.Sq0.setColor('#59FF00') #1
						self.Sq1.setColor('#B80117') #1
						self.Sq2.setColor('#B80117') #1

					self.win.flip()
					# print("Step ", stg, "of ", cyc)
					t1 = self.eachclock.getTime()
					print('One show = ',t1-t0,'s')
					print(ISI.complete())
		T1 = self.genclock.getTime()
		print('Flashing took',T1-T0,'s')
		self.stop()

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