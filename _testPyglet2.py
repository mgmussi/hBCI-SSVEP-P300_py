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
import cProfile, pstats, io


class FastRect:
	def __init__(self, x, y, width, height, color, batch):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.color = color
		self.batch = batch
		r = color[0]
		g = color[1]
		b = color[2]
		self.draw_mode = GL_QUADS
		self.vertex_list = self.batch.add(4, self.draw_mode, None, 'v2f/stream', 'c3B/stream')
		self.vertex_list.vertices = [x, y,
									x + width, y,
									x + width, y + height,
									x, y + height]
		self.vertex_list.colors = [r, g, b,
									r, g, b,
									r, g, b,
									r, g, b]

	def changeColor(self, new_color):
		self.__init__(self.x, self.y, self.width, self.height, new_color, self.batch)

	def Reposition(self, x, y):
		self.__init__(x, y, self.width, self.height, self.color, self.batch)


class StimuliWindow(pyglet.window.Window):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		width = args[0]
		hight = args[1]

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

		# ##::Create Rect elements
		# #::P300 Box
		# self.SqP = shapes.Rectangle(self.p2[0][0], self.p2[0][1], p3S,p3S,  color=(255,255,255), batch = self.batch)
		# #::Steady Black Boxes
		# self.St0 = shapes.Rectangle(self.p1[0][0], self.p1[0][1], pS,pS,  color=(0,0,0), batch = self.batch)
		# self.St1 = shapes.Rectangle(self.p1[1][0], self.p1[1][1], pS,pS,  color=(0,0,0), batch = self.batch)
		# self.St2 = shapes.Rectangle(self.p1[2][0], self.p1[2][1], pS,pS,  color=(0,0,0), batch = self.batch)
		# #::SSVEP Boxes
		# self.Sq0 = shapes.Rectangle(self.p[0][0], self.p[0][1], sS,sS,  color=mygreen, batch = self.batch)
		# self.Sq1 = shapes.Rectangle(self.p[1][0], self.p[1][1], sS,sS,  color=mygreen, batch = self.batch)
		# self.Sq2 = shapes.Rectangle(self.p[2][0], self.p[2][1], sS,sS,  color=mygreen, batch = self.batch)

		##::Create FastRect elements
		#::P300 Box
		self.SqP = FastRect(self.p2[0][0], self.p2[0][1], p3S,p3S,  color=(255,255,255), batch = self.batch)
		#::Steady Black Boxes
		self.St0 = FastRect(self.p1[0][0], self.p1[0][1], pS,pS,  color=(0,0,0), batch = self.batch)
		self.St1 = FastRect(self.p1[1][0], self.p1[1][1], pS,pS,  color=(0,0,0), batch = self.batch)
		self.St2 = FastRect(self.p1[2][0], self.p1[2][1], pS,pS,  color=(0,0,0), batch = self.batch)
		#::SSVEP Boxes
		self.Sq0 = FastRect(self.p[0][0], self.p[0][1], sS,sS,  color=mygreen, batch = self.batch)
		self.Sq1 = FastRect(self.p[1][0], self.p[1][1], sS,sS,  color=mygreen, batch = self.batch)
		self.Sq2 = FastRect(self.p[2][0], self.p[2][1], sS,sS,  color=mygreen, batch = self.batch)


	def on_draw(self):
		self.clear()
		self.batch.draw()
	
	def update_win(self, dt):
		print("Last callback: ", dt)
		# t0 = time.time()
		global gencounter
		global stg
		global cyc
		global idx
		global sel
		global TL
		global sequenceP300

		if cyc > 2:
			self.SqP.changeColor([0,0,0])
		else:
			self.SqP.changeColor([255,255,255])

		if stg == 0:
			self.Sq0.changeColor(mygreen) #0
			self.Sq1.changeColor(mygreen) #0
			self.Sq2.changeColor(mygreen) #0
		elif stg == 1:
			self.Sq0.changeColor(myblue)  #1
			self.Sq1.changeColor(mygreen) #0
			self.Sq2.changeColor(mygreen) #0
		elif stg == 2:
			self.Sq0.changeColor(mygreen) #0
			self.Sq1.changeColor(myred)	  #1
			self.Sq2.changeColor(mygreen) #0
		elif stg == 3:
			self.Sq0.changeColor(myblue)  #1
			self.Sq1.changeColor(myred)	  #1
			self.Sq2.changeColor(mygreen) #0
		elif stg == 4:
			self.Sq0.changeColor(mygreen) #0
			self.Sq1.changeColor(mygreen) #0
			self.Sq2.changeColor(myred)	  #1
		elif stg == 5:
			self.Sq0.changeColor(myblue)  #1
			self.Sq1.changeColor(mygreen) #0
			self.Sq2.changeColor(myred)	  #1
		elif stg == 6:
			self.Sq0.changeColor(mygreen) #0
			self.Sq1.changeColor(myred)	  #1
			self.Sq2.changeColor(myred)	  #1
		elif stg == 7:
			self.Sq0.changeColor(myblue)  #1
			self.Sq1.changeColor(myred)   #1
			self.Sq2.changeColor(myred)	  #1

		t1 = time.time()
		# print('Stage in ', t1-t0, 's')
		# print("Gencount: ", gencounter, "| Stage: ", stg, "| Cycle: ", cyc, "| IDX: ", idx, "|P300.: ", sel, "|TL: ", TL)
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
					TL = ex_TrialLabels[idx]
					sel = sequenceP300[idx]
					self.SqP.Reposition(self.p2[sel][0], self.p2[sel][1])
				except IndexError as e:
					pyglet.clock.unschedule(self.update_win)
					pyglet.app.exit()


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

@profile
def main():
	win = StimuliWindow(960,720, "Stimuli")
	win.set_location(0, 50)
	stg = 0
	cyc = 0

	pyglet.clock.schedule_interval(win.update_win, 1/120)
	pyglet.app.run()


##::True Labels for the trials
TrialLabels = []
sequenceP300 = []
ex_TrialLabels = []
for n in range (0,3):		#Num. Lables
	for k in range(0,1): 	##INCREASE TO 7 #num trials per square (total trials = n*k = 21)
		TrialLabels.append(n)
random.shuffle(TrialLabels)

for t in TrialLabels:
	for m in range(0,3):
		for p in range(0,1): # INCREASE TO 7 #num blinks per square
			sequenceP300.append(m)
			ex_TrialLabels.append(t)
random.shuffle(sequenceP300)
print("\nTrial labels Sequence {",ex_TrialLabels,"}\nP300 target Sequence {",sequenceP300,"}\n")

##::Initial Values
gencounter = 1
stg = 0
cyc = 0
idx = 0
TL = ex_TrialLabels[idx]
sel = sequenceP300[idx]

##::Color Definition
myblue = (0,56,138)
mygreen = (89,255,0)
myred = (118,0,23)
myorange = (255,108,0)


if __name__ == '__main__':
	main()	