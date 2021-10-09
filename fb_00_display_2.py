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
import winsound

P3NUM = 7
TRIALNUM = 3


##::True Labels for the trials
TrialLabels = []
for n in range (0,3):
	for k in range(0,TRIALNUM): ##INCREASE TO 7 #number of trials per square (total trials = n*k = 21)
		TrialLabels.append(n)
random.shuffle(TrialLabels)
print("\n|||||||||Trial labels Sequence {",TrialLabels,"}\n")

class FlickeringBoxes:
	def __init__(self, position):
		#Definitions
		self.H = 800 #window hight (to fit side of the window)
		self.W = 1920 #window width

		self.sS = 0.125			#square size
		self.pS = self.sS*1.35	#padding size
		self.p3S = self.sS*1.75	#p300 size

		self.p =[[-0.5, -0.5],[0, 0.5],[0.5, -0.5]] #from the centre; 1 is for 100%, 0 is for 50% and -1 is for 0% of screen
		# vert = [[1,1],[1,-1],[-1,-1],[-1,1]]
		xp = (2-(self.W/self.H))/2
		yp = (2*(self.H/self.W))/1
		#maybe use Pythagoras?
		vert = [[xp, yp],[xp, -yp],[-xp, -yp],[-xp, yp]]

		##::Create  and show the Window
		self.win = visual.Window([self.W,self.H], position, monitor = 'SSVEP Paradigm', color = 'black')
		self.win.flip()

		##::Create Squares
		self.SqP = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.p3S, pos = self.p[2], autoDraw = True)
		self.St0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = self.p[0], autoDraw = True)
		self.St1 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = self.p[1], autoDraw = True)
		self.St2 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = self.p[2], autoDraw = True)
		self.Sq0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = self.p[0], autoDraw = True)
		self.Sq1 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = self.p[1], autoDraw = True)
		self.Sq2 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = self.p[2], autoDraw = True)
		#new vvv
		self.cr0 = visual.TextStim(self.win, text = '+', pos = self.p[0], color = 'black')
		self.cr0.autoDraw = True
		self.cr1 = visual.TextStim(self.win, text = '+', pos = self.p[1], color = 'black')
		self.cr1.autoDraw = True
		self.cr2 = visual.TextStim(self.win, text = '+', pos = self.p[2], color = 'black')
		self.cr2.autoDraw = True

	def start(self):
		eachclock = core.MonotonicClock()
		genclock = core.MonotonicClock()
		ISI = core.StaticPeriod(screenHz = 240) #change depending on screen refresh rate
		# sq_colors = ['#59FF00', '#B80117'] #R&G
		sq_colors = ['#000000', '#FFFFFF'] #B&W << Best Results
		
		self.SqP.setOpacity(0)

		self.message = visual.TextStim(self.win, text = 'Flickering routine\n\nReady?')
		self.message.draw()
		self.win.flip()
		core.wait(1.5)

		self.message.setText('3')
		self.message.draw()
		self.win.flip()
		core.wait(1.0)

		self.message.setText('2')
		self.message.draw()
		self.win.flip()
		core.wait(1.0)

		self.message.setText('1')
		self.message.draw()
		self.win.flip()
		core.wait(1.0)

		print("|||\t\t>>>>>Initializing Flickering>>>>>") #print('0')
		# T0 = genclock.getTime()
		for TL in TrialLabels:
			

#-----------##::Showing Cue
			winsound.Beep(261, 200)
			self.message2 = visual.TextStim(self.win, text = 'Focus on the orange:', pos = [0, 0.75])
			if TL == 0:
				self.Sq0.setColor('#FF6C00')
				self.Sq1.setColor('white')
				self.Sq2.setColor('white')
			if TL == 1:
				self.Sq0.setColor('white')
				self.Sq1.setColor('#FF6C00')
				self.Sq2.setColor('white')
			if TL == 2:
				self.Sq0.setColor('white')
				self.Sq1.setColor('white')
				self.Sq2.setColor('#FF6C00')
			self.message2.draw()
			self.win.flip()

			sq0_flag = 1
			sq1_flag = 1
			sq2_flag = 1

			core.wait(3) ## more time from 3 to 5?
			
#-----------##

			##::Creating P300 sequence
			sequenceP300 = []
			for m in range (0,3):
				for k in range(0,P3NUM): #number of blinks per square
					sequenceP300.append(m)
			random.shuffle(sequenceP300)

			##::Avoid P300 flashing in the same place twice in a row
			i = 0
			while i < len(sequenceP300)-1:
				t = 1
				# print(i, t, len(sequenceP300)-i)
				while sequenceP300[i] == sequenceP300[i+t] and t < len(sequenceP300)-i:
					t += 1
					# print(i, t, len(sequenceP300)-i)
					if t + i > len(sequenceP300)-1: #there are duplicates at the end of the list
						sequenceP300.reverse()
						i = 0
						t = 1
				if t != 1:
					a = sequenceP300[i+1]
					sequenceP300[i+1] = sequenceP300[i+t]
					sequenceP300[i+t] = a
					# print(sequenceP300)     
				i += 1

			##::Starting Flickering
			# T0_ = eachclock.getTime()
			for sel in sequenceP300:
				self.SqP.setPos(self.p[sel]) #setting position for P300

#-------------------
				# t0 = eachclock.getTime()
				for cyc in range(1, 61, 1): #half a second (60frames @ 120Hz)

					# ISI.start(0.01245) # <<- If stim starts to lag, use this value before 'if cyc > 48', instead
#-------------------##P300 stimuli
					if cyc > 48: #three fourths of cycle
						self.SqP.setOpacity(0)
					else:
						self.SqP.setOpacity(1)
						pass
#-----------------------
					

#-------------------##SSVEP stimuli
					if cyc % 4 == 0:							#for 20Hz use % 3 		#for 15Hz use % 4		#for 30Hz use % 2
						sq0_flag = not sq0_flag
						self.Sq0.setColor(sq_colors[sq0_flag])
					if cyc % 6 == 0:							#for 15Hz use % 4 		#for 10Hz use % 6		#for 15Hz use % 4
						sq1_flag = not sq1_flag
						self.Sq1.setColor(sq_colors[sq1_flag])
					if cyc % 10 == 0:							#for 12Hz use % 5 		#for 6Hz use % 10		#for 7.5Hz use % 8
						sq2_flag = not sq2_flag
						self.Sq2.setColor(sq_colors[sq2_flag])
#-----------------------###maybe use the same color as the cue?
					
					self.win.flip()
					ISI.start(0.0081)
					
					
					# print(ISI.complete())
					ISI.complete()
				# t1 = eachclock.getTime()
				# print('Half a second? = ',t1-t0,'s')
			# T1_ = eachclock.getTime()
			# print('All P300 =',T1_-T0_,'s\nEach cycle =',(T1_-T0_)/len(sequenceP300))

		# T1 = genclock.getTime()
		# print('||Flickering took', round(T1-T0,3), 's')

		self.Sq0.setOpacity(0)
		self.Sq1.setOpacity(0)
		self.Sq2.setOpacity(0)
		self.SqP.setOpacity(0)
		self.message.setText('Experiment is ending...')
		self.message.draw()
		self.win.flip()

	def stop(self):
		self.win.close()
		core.quit()
		sys.exit()
		# time.sleep(1)

if __name__ == '__main__':

	FlkWin = FlickeringBoxes([0,50])
	FlkWin.start()