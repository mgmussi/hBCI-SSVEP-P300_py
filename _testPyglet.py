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
# from psychopy import core, visual
import numpy as np
# import AppWindow as app
import random
import time
import pyglet
from pyglet.gl import *
from pyglet import shapes

class StimuliWindow(pyglet.window.Window):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		width = args[0]
		hight = args[1]

		print(width, hight)
		#Definitions
		sS = 100			#square size
		pS = sS*1.35	#padding size
		p3S = sS*1.75	#p300 size

		##::Position
		self.p =[[(width-sS/2-3*width/4), (hight-sS/2-3*hight/4)],[(-sS/2+width/2), (hight-sS/2-hight/4)],[(width-sS/2-width/4), (hight-sS/2-3*hight/4)]]
		self.p1 =[[(width-pS/2-3*width/4), (hight-pS/2-3*hight/4)],[(-pS/2+width/2), (hight-pS/2-hight/4)],[(width-pS/2-width/4), (hight-pS/2-3*hight/4)]]
		self.p2 =[[(width-p3S/2-3*width/4), (hight-p3S/2-3*hight/4)],[(-p3S/2+width/2), (hight-p3S/2-hight/4)],[(width-p3S/2-width/4), (hight-p3S/2-3*hight/4)]]

		##::Create Batch
		self.batch = pyglet.graphics.Batch()

		##::Create display elements
		#::P300 Box
		self.SqP = shapes.Rectangle(self.p2[0][0], self.p2[0][1], p3S,p3S,  color=(255,255,255), batch = self.batch)
		#::Steady Black Boxes
		self.St0 = shapes.Rectangle(self.p1[0][0], self.p1[0][1], pS,pS,  color=(0,0,0), batch = self.batch)
		self.St1 = shapes.Rectangle(self.p1[1][0], self.p1[1][1], pS,pS,  color=(0,0,0), batch = self.batch)
		self.St2 = shapes.Rectangle(self.p1[2][0], self.p1[2][1], pS,pS,  color=(0,0,0), batch = self.batch)
		#::SSVEP Boxes
		self.Sq0 = shapes.Rectangle(self.p[0][0], self.p[0][1], sS,sS,  color=mygreen, batch = self.batch)
		self.Sq1 = shapes.Rectangle(self.p[1][0], self.p[1][1], sS,sS,  color=mygreen, batch = self.batch)
		self.Sq2 = shapes.Rectangle(self.p[2][0], self.p[2][1], sS,sS,  color=mygreen, batch = self.batch)


	def on_draw(self):
		self.clear()
		self.batch.draw()
	
	def update_win(self, dt):
		print("Last callback: ", dt)
		# t0 = time.time()
		global gencounter
		global stg
		global cyc
		global tr
		global idx
		global sel
		global sequenceP300

		if cyc > 2:
			self.SqP.opacity = 0
		else:
			self.SqP.opacity = 255

		if stg == 0:
			self.Sq0.color = mygreen #0
			self.Sq1.color = mygreen #0
			self.Sq2.color = mygreen #0
		elif stg == 1:
			self.Sq0.color = myblue #1
			self.Sq1.color = mygreen #0
			self.Sq2.color = mygreen #0
		elif stg == 2:
			self.Sq0.color = mygreen #0
			self.Sq1.color = myred #1
			self.Sq2.color = mygreen #0
		elif stg == 3:
			self.Sq0.color = myblue #1
			self.Sq1.color = myred #1
			self.Sq2.color = mygreen #0
		elif stg == 4:
			self.Sq0.color = mygreen #0
			self.Sq1.color = mygreen #0
			self.Sq2.color = myred #1
		elif stg == 5:
			self.Sq0.color = myblue #1
			self.Sq1.color = mygreen #0
			self.Sq2.color = myred #1
		elif stg == 6:
			self.Sq0.color = mygreen #0
			self.Sq1.color = myred #1
			self.Sq2.color = myred #1
		elif stg == 7:
			self.Sq0.color = myblue #1
			self.Sq1.color = myred #1
			self.Sq2.color = myred #1

		# t1 = time.time()
		# print('Stage in ', t1-t0, 's')

		
		# print("Gencount: ", gencounter, "| Stage: ", stg, "| Cycle: ", cyc, "| IDX: ", idx, "|Sel.: ", sel, "|OP.: ",self.SqP.opacity )
		gencounter += 1
		stg += 1
		if stg > 7:
			cyc += 1
			stg = 0

			if cyc > 3:
				idx += 1
				cyc = 0
				gencounter = 1

				try:
					sel = sequenceP300[idx]
					self.SqP.position = self.p2[sel][0], self.p2[sel][1]
				except IndexError as e:
					tr += 1
					idx = 0

					try:
						curr_trial = TrialLabels[tr]
						sequenceP300 = []
						for m in range (0,3):
							for k in range(0,1): # INCREASE TO 7 #number of blinks per square
								sequenceP300.append(m)
						random.shuffle(sequenceP300)
						print("\nP300 labels Sequence {",sequenceP300,"}\n")
						sel = sequenceP300[idx]
						self.SqP.position = self.p2[sel][0], self.p2[sel][1]
					except IndexError as e:
						pyglet.clock.unschedule(self.update_win)
						pyglet.app.exit()






if __name__ == '__main__':
	##::True Labels for the trials
	TrialLabels = []
	for n in range (0,3):
		for k in range(0,1): ##INCREASE TO 7 #number of trials per square (total trials = n*k = 21)
			TrialLabels.append(n)
	random.shuffle(TrialLabels)
	print("\nTrial labels Sequence {",TrialLabels,"}\n")

	##::Initial Values
	gencounter = 1
	stg = 0
	cyc = 0
	tr = 0
	idx = 0
	curr_trial = TrialLabels[tr]
	sequenceP300 = []
	for m in range (0,3):
		for k in range(0,1): # INCREASE TO 7 #number of blinks per square
			sequenceP300.append(m)
	random.shuffle(sequenceP300)
	print("\nP300 labels Sequence {",sequenceP300,"}\n")
	sel = sequenceP300[idx]

	##::Color Definition
	myblue = (0,56,138)
	mygreen = (89,255,0)
	myred = (118,0,23)
	myorange = (255,108,0)

	win = StimuliWindow(960,720, "Stimuli")
	stg = 0
	cyc = 0


	pyglet.clock.schedule_interval(win.update_win, 1/60)
	pyglet.app.run()