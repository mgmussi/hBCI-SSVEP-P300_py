#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Updates from v8.6:
- New fusion features
- New selector features
- Using channels Pz (and T1 and T2) for SSVEP

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
VERSION = 9.1
import os
import sys
import pickle
import csv
from psychopy import core, visual, gui, monitors
import math
import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
import random
import time
from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import queue
from queue import Empty
from datetime import datetime
import pandas as pd
pd.options.display.float_format = '{:20.6f}'.format

import scipy
from scipy.signal import butter, lfilter, kaiserord, firwin, iirnotch, decimate
from scipy.fft import fft, fftfreq

import cProfile, pstats, io
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes#, AggOperation

from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

import mne
from mne.decoding import Scaler
mne_scaler = Scaler(scalings = 'median')

import warnings
warnings.filterwarnings("error")
warnings.filterwarnings('ignore', 'Solver terminated early.*') #for when "sklearn.exceptions.ConvergenceWarning: Maximum number of iterations reached"
warnings.filterwarnings('ignore', 'Maximum number of iterations reached')

import winsound



'''
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
'''

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

##CLASSES
class FlickeringBoxes: #using monitor @ 120Hz = 12, 15, 20 or 6, 10, 15
	# frequency = 2500  # Set Frequency To 2500 Hertz
	# duration = 200  # Set Duration To 1000 ms == 1 second

	def __init__(self, position):
		#Monitor
		screenHight = 1080	# screen height in px
		screeenWidth = 1920	# screen width in px
		displayHight = 1080 #800	# display height in px
		displayWidth = 1920	# display width in px
		monitorwidth = 34.9	# monitor width in cm
		viewdist = 60. # viewing distance in cm
		monitorname = 'SSVEP P300 Paradigm'
		scrn = 0 # 0 to use main screen, 1 to use external screen
		mon = monitors.Monitor(monitorname, width=monitorwidth, distance=viewdist)
		mon.setSizePix((screeenWidth, screenHight))

		#Definitions
		self.sS = 0.125				#square size
		self.pS = self.sS*1.35		#padding size
		self.p3S = self.sS*1.75		#p300 size

		self.p =[[-0.5, -0.5],[0, 0.5],[0.5, -0.5]] #from the centre; 1 is for 100%, 0 is for 50% and -1 is for 0% of screen

		xp = 0.25*screeenWidth/displayWidth #0.25
		yp = (1 + (displayHight/displayWidth)) * xp # xp*(discreeenWidth/displayWidthsplayWidth/displayHight) + (displayWidth-displayHight)/(displayWidth*(4))

		vert = [[xp, yp],[xp, -yp],[-xp, -yp],[-xp, yp]]

		##::Create  and show the Window
		self.win = visual.Window(size = (displayWidth, displayHight), position = position, screen=scrn, monitor = mon, color = 'black')
		self.win.flip()

		##::Create Squares
		self.SqP = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.p3S, pos = self.p[2], autoDraw = True)
		self.St0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = self.p[0], autoDraw = True)
		self.St1 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = self.p[1], autoDraw = True)
		self.St2 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'black', lineColor = 'black', size = self.pS, pos = self.p[2], autoDraw = True)
		self.Sq0 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = self.p[0], autoDraw = True)
		self.Sq1 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = self.p[1], autoDraw = True)
		self.Sq2 = visual.ShapeStim(self.win, vertices = vert, fillColor = 'white', lineColor = 'white', size = self.sS, pos = self.p[2], autoDraw = True)

		self.cr0 = visual.TextStim(self.win, text = '+', pos = self.p[0], color = 'black')
		self.cr0.autoDraw = True
		self.cr1 = visual.TextStim(self.win, text = '+', pos = self.p[1], color = 'black')
		self.cr1.autoDraw = True
		self.cr2 = visual.TextStim(self.win, text = '+', pos = self.p[2], color = 'black')
		self.cr2.autoDraw = True

		# self.sq_colors = ['#59FF00', '#B80117'] #R&G
		self.sq_colors = ['#000000', '#FFFFFF'] #B&W << Best Results
		self.screenRefreshRate = 240 #change depending on screen refresh rate


	def practice_sequence(self):
		ISI = core.StaticPeriod(screenHz = self.screenRefreshRate)
		
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

		print("[Flickering...]\n") #print('0')
		for TL in TrialLabels:
			##::Showing Cue
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

			core.wait(3)

			##::Creating P300 sequence
			sequenceP300 = []
			for m in range (0,3):
				for k in range(0,P3NUM): #number of blinks per square
					sequenceP300.append(m)
			random.shuffle(sequenceP300)

			##::Avoid P300 flashing in the same place twice in a row
			sequenceP300 = self.unwind_P300(sequenceP300)

			##::Starting Flickering
			for sel in sequenceP300:
				self.SqP.setPos(self.p[sel]) #setting position for P300

				for cyc in range(1, 61, 1): #half a second (60frames @ 120Hz)
					#P300 stimuli
					if cyc > 48: #three fourths of cycle
						self.SqP.setOpacity(0)
					else:
						self.SqP.setOpacity(paradigmP300)
						pass

					#SSVEP stimuli
					if paradigmSSVEP:
						if cyc % 4 == 0:							#for 20Hz use % 3 		#for 15Hz use % 4		#for 30Hz use % 2
							sq0_flag = not sq0_flag
							self.Sq0.setColor(self.sq_colors[sq0_flag])
						if cyc % 6 == 0:							#for 15Hz use % 4 		#for 10Hz use % 6		#for 15Hz use % 4
							sq1_flag = not sq1_flag
							self.Sq1.setColor(self.sq_colors[sq1_flag])
						if cyc % 10 == 0:							#for 12Hz use % 5 		#for 6Hz use % 10		#for 7.5Hz use % 8
							sq2_flag = not sq2_flag
							self.Sq2.setColor(self.sq_colors[sq2_flag])
					
					self.win.flip()
					ISI.start(0.0081)
					ISI.complete()

		self.Sq0.setOpacity(0)
		self.Sq1.setOpacity(0)
		self.Sq2.setOpacity(0)
		self.SqP.setOpacity(0)
		self.message.setText('Experiment is ending...')
		self.message.draw()
		self.win.flip()
		core.wait(3)
		self.stop()

	def training_sequence(self):
		HOLD.clear() #print('h')
		ISI = core.StaticPeriod(screenHz = self.screenRefreshRate) 
		
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

		print("[Flickering...]\n") #print('0')
		# T0 = genclock.getTime()
		for TL in TrialLabels:
			##::Showing Cue
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
			core.wait(3) ## more time from 3 to 5?

			sq0_flag = 1
			sq1_flag = 1
			sq2_flag = 1

			HOLD.set()

			##::Setting SSVEP Label
			if TL == 0:
				ES1Label.set()
			elif TL == 1:
				ES2Label.set()
			elif TL == 2:
				ES3Label.set()

			##::Creating P300 sequence
			sequenceP300 = []
			for m in range (0,3):
				for k in range(0,P3NUM): #number of blinks per square
					sequenceP300.append(m)
			random.shuffle(sequenceP300)

			##::Avoid P300 flashing in the same place twice in a row
			sequenceP300 = self.unwind_P300(sequenceP300)

			##::Starting Flickering
			for sel in sequenceP300:
				COLLECTTRIG.clear() #print('\nc_')
				P300TRIG.set() #print('T')
				self.SqP.setPos(self.p[sel]) #setting position for P300

				##::Setting P300 Labels
				#:Target x Non-Target Label
				if TL == sel:
					EPLabel.set()
				else:
					EPLabel.clear()
				#:Position Label
				if sel == 0:
					EPos1Label.set()
				elif sel == 1:
					EPos2Label.set()
				elif sel == 2:
					EPos3Label.set()

				for cyc in range(1, 61, 1): #half a second (60frames @ 120Hz)

					##P300 stimuli
					if cyc > 48: #three fourths of cycle
						if not COLLECTTRIG.is_set():
							COLLECTTRIG.set() #print('C')
						self.SqP.setOpacity(0)
					else:
						self.SqP.setOpacity(paradigmP300)
						pass

					##SSVEP stimuli
					if paradigmSSVEP:
						if cyc % 4 == 0:							#for 20Hz use % 3 		#for 15Hz use % 4		#for 30Hz use % 2
							sq0_flag = not sq0_flag
							self.Sq0.setColor(self.sq_colors[sq0_flag])
						if cyc % 6 == 0:							#for 15Hz use % 4 		#for 10Hz use % 6		#for 15Hz use % 4
							sq1_flag = not sq1_flag
							self.Sq1.setColor(self.sq_colors[sq1_flag])
						if cyc % 10 == 0:							#for 12Hz use % 5 		#for 6Hz use % 10		#for 7.5Hz use % 8
							sq2_flag = not sq2_flag
							self.Sq2.setColor(self.sq_colors[sq2_flag])
					# maybe use the same color as the cue?

					self.win.flip()
					ISI.start(0.0081)# ISI.start(0.01245) <<- If stim starts to lag, use this value before 'if cyc > 48', instead
					ISI.complete()
			HOLD.clear() #print('h')
		E.set() #print('1')

		self.Sq0.setOpacity(0)
		self.Sq1.setOpacity(0)
		self.Sq2.setOpacity(0)
		self.SqP.setOpacity(0)
		self.message.setText('Experiment is ending...')
		self.message.draw()
		self.win.flip()

		winsound.Beep(261, 50)
		winsound.Beep(329, 50)
		winsound.Beep(415, 50)

	def live_sequence(self):
		HOLD.clear() #print('h')
		ISI = core.StaticPeriod(screenHz = self.screenRefreshRate)
		
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

		print("[Flickering...]\n") #print('0')
		# T0 = genclock.getTime()
		for TL in TrialLabels:
			##::Showing Cue
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
			core.wait(3) #if time increases, add more time to <Events>.wait() timeout

			sq0_flag = 1
			sq1_flag = 1
			sq2_flag = 1

			HOLD.set()

			##::Setting SSVEP Label
			if TL == 0:
				ES1Label.set()
			elif TL == 1:
				ES2Label.set()
			elif TL == 2:
				ES3Label.set()

			##::Creating P300 sequence
			sequenceP300 = []
			for m in range (0,3):
				for k in range(0,P3NUM): #number of blinks per square
					sequenceP300.append(m)
			random.shuffle(sequenceP300)

			##::Avoid P300 flashing in the same place twice in a row
			sequenceP300 = self.unwind_P300(sequenceP300)

			##::Starting Flickering
			for sel in sequenceP300:
				
				if EFeedback.is_set():
					break

				COLLECTTRIG.clear() #print('\nc_')
				P300TRIG.set() #print('T')
				self.SqP.setPos(self.p[sel]) #setting position for P300

				##::Setting P300 Labels
				#:Target x Non-Target Label
				if TL == sel:
					EPLabel.set()
				else:
					EPLabel.clear()
				#:Position Label
				if sel == 0:
					EPos1Label.set()
				elif sel == 1:
					EPos2Label.set()
				elif sel == 2:
					EPos3Label.set()

				for cyc in range(1, 61, 1): #half a second (60frames @ 120Hz)

					##P300 stimuli
					if cyc > 48: #three fourths of cycle
						if not COLLECTTRIG.is_set():
							COLLECTTRIG.set() #print('C')
						self.SqP.setOpacity(0)
					else:
						self.SqP.setOpacity(paradigmP300)
						pass

					##SSVEP stimuli
					if paradigmSSVEP:
						if cyc % 4 == 0:							#for 20Hz use % 3 		#for 15Hz use % 4		#for 30Hz use % 2
							sq0_flag = not sq0_flag
							self.Sq0.setColor(self.sq_colors[sq0_flag])
						if cyc % 6 == 0:							#for 15Hz use % 4 		#for 10Hz use % 6		#for 15Hz use % 4
							sq1_flag = not sq1_flag
							self.Sq1.setColor(self.sq_colors[sq1_flag])
						if cyc % 10 == 0:							#for 12Hz use % 5 		#for 6Hz use % 10		#for 7.5Hz use % 8
							sq2_flag = not sq2_flag
							self.Sq2.setColor(self.sq_colors[sq2_flag])
					# maybe use the same color as the cue?

					self.win.flip()
					ISI.start(0.0081)# ISI.start(0.01245) <<- If stim starts to lag, use this value before 'if cyc > 48', instead
					ISI.complete()
			HOLD.clear() #stops data collection

			##::Show Feedback
			EFeedback.wait(timeout = 5) #in case of timeout, waits for Feedback flags
			winsound.Beep(391, 75)
			winsound.Beep(195, 75)
			winsound.Beep(391, 75)
			self.message2 = visual.TextStim(self.win, text = 'Chosen square:')
			if EFeed1.is_set():
				self.Sq0.setColor('#59FF00')
				self.Sq1.setColor('white')
				self.Sq2.setColor('white')
				self.SqP.setOpacity(1)
				self.SqP.setPos(self.p[0])
				self.SqP.setColor('#59FF00')
			if EFeed2.is_set():
				self.Sq0.setColor('white')
				self.Sq1.setColor('#59FF00')
				self.Sq2.setColor('white')
				self.SqP.setOpacity(1)
				self.SqP.setPos(self.p[1])
				self.SqP.setColor('#59FF00')
			if EFeed3.is_set():
				self.Sq0.setColor('white')
				self.Sq1.setColor('white')
				self.Sq2.setColor('#59FF00')
				self.SqP.setOpacity(1)
				self.SqP.setPos(self.p[2])
				self.SqP.setColor('#59FF00')
			self.message2.draw()
			self.win.flip()
			core.wait(3) #if time increases, add more time to Events.wait() below
			self.SqP.setColor('#FFFFFF')
			self.SqP.setOpacity(0)
			EFeed1.clear()
			EFeed2.clear()
			EFeed3.clear()
			EFeedback.clear()

		E.set() #print('1')

		self.Sq0.setOpacity(0)
		self.Sq1.setOpacity(0)
		self.Sq2.setOpacity(0)
		self.SqP.setOpacity(0)
		self.message.setText('Experiment is ending...')
		self.message.draw()
		self.win.flip()

		winsound.Beep(261, 150)
		winsound.Beep(329, 150)
		winsound.Beep(415, 150)


	def unwind_P300(self, sequenceP300):
		i = 0
		while i < len(sequenceP300)-1:
			t = 1
			while sequenceP300[i] == sequenceP300[i+t] and t < len(sequenceP300)-i:
				t += 1
				if t + i > len(sequenceP300)-1: #there are duplicates at the end of the list
					sequenceP300.reverse()
					i = 0
					t = 1
			if t != 1:
				a = sequenceP300[i+1]
				sequenceP300[i+1] = sequenceP300[i+t]
				sequenceP300[i+t] = a   
			i += 1
		return(sequenceP300)

	def stop(self):
		self.win.close()
		core.quit()
		# time.sleep(1)

class AcqThread(threading.Thread):
	def __init__(self, dataOutQ1, dataOutQ2, saveQ):
		threading.Thread.__init__(self)
		self.dataOutQ2 = dataOutQ2
		self.dataOutQ1 = dataOutQ1
		self.saveQ = saveQ

	def run(self):
		Acquisition(self.dataOutQ1, self.dataOutQ2, self.saveQ) #place where function is called

class P300Thread(threading.Thread):
	def __init__(self, dataInQ, featureQ):
		threading.Thread.__init__(self)
		self.dataInQ = dataInQ
		self.featureQ = featureQ

	def run(self):
		P300fun(self.dataInQ, self.featureQ)

class SSVEPThread(threading.Thread):
	def __init__(self, dataInQ, featureQ):
		threading.Thread.__init__(self)
		self.dataInQ = dataInQ
		self.featureQ = featureQ

	def run(self):
		SSVEPfun(self.dataInQ, self.featureQ)

class SSVEP_Class_Thread(threading.Thread):
	def __init__(self, featureQ, modelQ, classOutQ, dataclassOutQ, mode, SSVEP_filename):
		threading.Thread.__init__(self)
		self.featureQ = featureQ
		self.modelQ = modelQ
		self.mode = mode
		self.classOutQ = classOutQ
		self.dataclassOutQ = dataclassOutQ
		self.filename = SSVEP_filename

	def run(self):
		if self.mode == 1:
			SSVEP_classifier_train(self.featureQ, self.modelQ, self.filename)
		elif self.mode == 2:
			SSVEP_classifier_predict(self.featureQ, self.modelQ, self.classOutQ, self.dataclassOutQ, self.filename)

class P300_Class_Thread(threading.Thread):
	def __init__(self, featureQ, modelQ, classOutQ, dataclassOutQ, mode, P300_filename):
		threading.Thread.__init__(self)
		self.featureQ = featureQ
		self.modelQ = modelQ
		self.mode = mode
		self.classOutQ = classOutQ
		self.dataclassOutQ = dataclassOutQ
		self.filename = P300_filename

	def run(self):
		if self.mode == 1:
			P300_classifier_train(self.featureQ, self.modelQ, self.filename)
		elif self.mode == 2:
			P300_classifier_predict(self.featureQ, self.modelQ, self.classOutQ, self.dataclassOutQ, self.filename)

class Fusion_Thread(threading.Thread):
	def __init__(self, predict_LDA, predict_CCA, accs_LDA, accs_CCA, filename):
		threading.Thread.__init__(self)
		self.predict_CCA = predict_CCA
		self.predict_LDA = predict_LDA
		self.accs_LDA = accs_LDA
		self.accs_CCA = accs_CCA
		self.filename = filename

	def run(self):
		fusion(self.predict_LDA, self.predict_CCA, self.accs_LDA, self.accs_CCA, self.filename)


##FUNCTIONS
def init_dialog():
	global _currDir
	global _partDir
	global _csvsession
	#new
	global _csvsippets
	global _imgsession
	global _classifile
	global _classioutLDA_SSVEP
	global _classioutLDA_P300
	global session_mode
	global headset
	global paradigm
	global TRIALNUM
	global P3NUM
	global paradigmP300
	global paradigmSSVEP
	global _LivePerformance


	_currDir = os.path.dirname(os.path.abspath(__file__))

	#Dialog 1
	session_info = {'ParticipantID':'00', 'SessionNumber':'01', 'Version': VERSION, 'Mode': ['Training', 'Practice', 'Live'], 'Paradigm':['P300','SSVEP','Hybrid']}
	infoDlg1 = gui.DlgFromDict(dictionary=session_info,
								title='Hybrid BCI system',
								fixed=['Version'],
								order = ['Version','ParticipantID','SessionNumber','Mode', 'Paradigm'])
	if not infoDlg1.OK:
		print('Operation Cancelled')
		core.quit()
	else:
		
		paradigm = session_info['Paradigm']
		if paradigm == 'P300':
			paradigmP300 = 1
			paradigmSSVEP = 0
		if paradigm == 'SSVEP':
			paradigmP300 = 0
			paradigmSSVEP = 1
		if paradigm == 'Hybrid':
			paradigmP300 = 1
			paradigmSSVEP = 1

		if session_info['Mode'] == 'Practice':
			TRIALNUM = 3
			P3NUM = 7
			session_mode = 0
		else:
			#Dialog 2
			set_info = {'ParticipantID':session_info['ParticipantID'], 'SessionNumber': session_info['SessionNumber'],
						'SetNum':'01', 'Headset':['','cyton', 'gtec']}
			infoDlg2 = gui.DlgFromDict(dictionary=set_info,
										title='Hybrid BCI system',
										fixed=['ParticipantID', 'SessionNumber'],
										order = ['ParticipantID','SessionNumber','SetNum','Headset'])
			if not infoDlg2.OK:
				print('Operation Cancelled')
				core.quit()
			else:
				#Check for headset selection
				headset = set_info['Headset']
				if headset == '':
					print('ERROR! NO Headset selected. Quitting application...')
					core.quit()
				print('[SessMode][{}]\n[Headset][{}]'.format(session_info['Mode'], headset))

				#create dir
				
				if session_info['Paradigm'] == 'P300':
					_partDir = os.path.join(_currDir, 'Participants', session_info['ParticipantID'], 'SESS' + session_info['SessionNumber'] + 'p')
				if session_info['Paradigm'] == 'SSVEP':
					_partDir = os.path.join(_currDir, 'Participants', session_info['ParticipantID'], 'SESS' + session_info['SessionNumber'] + 's')
				if session_info['Paradigm'] == 'Hybrid':
					_partDir = os.path.join(_currDir, 'Participants', session_info['ParticipantID'], 'SESS' + session_info['SessionNumber'] + 'h')

				if not os.path.exists(_partDir):
					os.makedirs(_partDir)

				_classifile = u'%s_%s_classifier_models.plk' % (session_info['ParticipantID'],session_info['SessionNumber'])
				_csvsession = u'%s_%s_%s_saveddata.csv' % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])
				_csvsippets = u'%s_%s_%s_savedsnippet.csv' % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])
				_imgsession = u'%s_%s_%s_saveddata.png' % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])
				_classioutLDA_SSVEP = u'%s_%s_%s_LDAresults_SSVEPs.txt' % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])
				_classioutLDA_P300  = u'%s_%s_%s_LDAresults_P300.txt'  % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])
				_LivePerformance = u'%s_%s_%s_performance.txt' % (session_info['ParticipantID'],session_info['SessionNumber'],set_info['SetNum'])

				if session_info['Mode'] == 'Training':
					TRIALNUM = 5
					P3NUM = 7
					session_mode = 1

				elif session_info['Mode'] == 'Live':
					TRIALNUM = 5
					P3NUM = 7
					session_mode = 2

def normalization(matrix, x_max = 1, x_min = -1):
	try:
		avgsamples = scaler.fit_transform(matrix)
	except RuntimeWarning:
			print('Caution::may have RAILED channel(s)')
			print(']]Caution::may have RAILED channel(s)[[')
	return avgsamples

# @profile
def Acquisition(dataOutQ1, dataOutQ2, saveQ):
	'''
	Paradigm		IDEAL CHs 				CYTON 					GTEC
	[P300] 		 	[PO8, PO7, POz, CPz]	[P3, Pz, P4, Cz]		[C3, Cz, C4, Pz]
	[SSVEP] 		[O1, Oz, O2]			[T5, Pz, T6,  O1, O2] 	[O1, O2, Oz]
	'''
	if headset == 'cyton':
		chs_p300 = [2,3,4,5]
		chs_ssvep = [1,3,8,6,7] #new channels
		ts_idx = 22
	elif headset == 'gtec':
		chs_p300 = [2,3,4,5]
		chs_ssvep = [6,7,8]
		ts_idx = 17

	chs_all = list(set(chs_p300 + chs_ssvep))
	_chs_all = [x - 1 for x in chs_all]
	_chs_p300 = [x-min(chs_all) for x in chs_p300]
	_chs_ssvep = [x-min(chs_all) for x in chs_ssvep]

	fs = SPS	#Hz
	low = 5 / NYQ
	high = 30 / NYQ
	win = 75 #padding

	#Notch
	b1, a1 = iirnotch(60, 1, SPS)

	##FIR
	width = 2/NYQ #transition width
	ripple_db = 11 #attenuation
	N, beta = kaiserord(ripple_db, width)
	b = firwin(N, [low, high], pass_zero = 'bandpass',  window=('kaiser', beta))
	delay = int(0.5 * (N-1)) # / SPS
	print('[FIR Order] ', N)
	print('[Delay    ] ', delay)

	thesamples = []
	thefilterd = []
	thetimestamps = []
	thefilttimestamps = []
	tstampsP300 = []

	p_avgsamples = np.array([])
	p_filtsamples = np.array([])
	s_avgsamples = np.array([])
	s_filtsamples = np.array([])

	i = 0
	start = time.perf_counter() #time.time() #<for older python versions
	while not E.is_set(): #while still displaying stimuli
		HOLD.wait() #time out to display instructions, cues and feedback
		P300TRIG.wait(timeout = 13) #function called when P300 starts

		##::Save stimuli beginning timestamp
		current_data = board.get_current_board_data(10)
		ini_stim_ts = current_data[ts_idx,-1]

		if not E.is_set():
			##::Getting Labels
			if EPLabel.is_set():
				ll = 1
			else:
				ll = 0

			if EPos1Label.is_set():
				pl = 0
			elif EPos2Label.is_set():
				pl = 1
			elif EPos3Label.is_set():
				pl = 2

			if ES1Label.is_set():
				TL = 0
			elif ES2Label.is_set():
				TL = 1
			elif ES3Label.is_set():
				TL = 2
			# print('|Labels ll and TL =', ll, TL)

			EPos1Label.clear()
			EPos2Label.clear()
			EPos3Label.clear()
			ES1Label.clear()
			ES2Label.clear()
			ES3Label.clear()

			##::Collecting Signal
			COLLECTTRIG.wait() #function called when P300 ends
			P300TRIG.clear() #print('t')
			while board.get_board_data_count() < NYQ:
				pass #time.sleep(0.001)
			data = board.get_board_data() 	#<class 'numpy.ndarray'>
			data_ls = data.tolist() 		#list


			##::Save stimuli ending timestamp
			end_stim_ts = data[ts_idx,-1]

			##::Save data from chosen electrodes
			ss = np.array([data_ls[i][-NYQ:] for i in chs_all])
			tt = data_ls[ts_idx][-NYQ:]
			
			##::Filtering and normalizing samples
			pf = np.hstack(( np.flip(ss[:, 0:win], axis = 1), ss[:, :], np.flip(ss[:, -win:], axis = 1) )) #DIFFERENT LINE (using list)	#create padding  #user np.pad 'reflect' instead?
			notch = lfilter(b1, a1, pf, axis = 1)	#apply notch
			s_notch = notch[:,win:-win]	#crop padding
			pf2 = np.hstack(( np.flip(s_notch[:, 0:win], axis = 1), s_notch[:, :], np.flip(s_notch[:, -win:], axis = 1) )) #create padding 2
			posfilt = lfilter(b, [1.0], pf2, axis = 1) #apply FIR
			avgsamples = scaler.fit_transform(posfilt[:, win+delay:-win].T).T.tolist() #crop padding 2, removes delay and normalizing

			##::Storing data
			thesamples.append(ss.tolist())
			thefilterd.append(avgsamples)
			thetimestamps.append(tt)
			thefilttimestamps.append(tt[delay:])
			
			tstampsP300.append([ini_stim_ts, end_stim_ts, ll, TL, pl])

			##::Data into queue
			threadLock.acquire()				#queue locked to prevent other threads to access it
			#sending list, list and int labels
			dataOutQ1.put([[avgsamples[i] for i in _chs_p300], tt, ll, pl])
			dataOutQ2.put([[avgsamples[i] for i in _chs_ssvep], tt, TL])
			threadLock.release()				#queue unlocked with info

			##::(Re)set Events
			EP300.set() # print('P')
			ESSVEP.set() # print('S')
			i += 1
		else:
			break

	print('[Samples Array Size]', np.array(thesamples).shape)
	print('[FltSamp Array Size]', np.array(thefilterd).shape)
	print('[TmStamp Array Size]', np.array(thetimestamps).shape)
	print('[FltTStp Array Size]', np.array(thefilttimestamps).shape)

	threadLock.acquire()
	saveQ.put([thesamples, thefilterd, [thetimestamps, thefilttimestamps], tstampsP300])
	threadLock.release()
	###
	print('\n\n[Ending acquisition]')
	P300Th.join()
	SSVEPTh.join()
	finish = time.perf_counter()
	# print('___\n|Final Score: Samples [',thesamples.shape,'], Timestamp [',thetimestamps.shape,']') #for numpy type
	print('[Streamed in',round(finish-start,3),'s]\n---')

######################
######################

def P300fun(dataInQ, featureQ):
	ss = []
	tt = []
	while not E.is_set():
		EP300.wait(timeout = 13)
		if not E.is_set():

			##::Collect Samples from Queue
			while dataInQ.qsize():
				try:
					ss, tt, ll, pl = dataInQ.get(0)
				except Empty:
					return

			# Downsample by factor 15
			ss_ds = decimate(ss, 15, axis = 1).reshape(-1)

			threadLock.acquire()
			featureQ.put([ss_ds,  ll, pl]) #tt_ds, ll])
			threadLock.release()

			##::END OF PROCESSING
			ELDA.set() #print('LDA')
			EP300.clear() #print('p')
			ss = []
			tt = []
		else:
			break
	print('[P300 fun] :: [OVER]')


def SSVEPfun(dataInQ, featureQ):
	ss = []
	tt = []
	
	while not E.is_set():
		ESSVEP.wait(timeout = 13)
		if not E.is_set():
			feat = []

			##::Collect Samples from Queue
			while dataInQ.qsize():
				try:
					ss, tt, ll = dataInQ.get(0)
				except Empty:
					return

			##::Get PSD bins sum for each target frq and its double frq (+-0.5Hz)
			##total number of features: chs*2*frqs = 2*2*3 = 12 features
			xfft = fftfreq(SPS*2, 1/SPS) #precision of 0.5Hz
			for s_ch in ss:
				# print(s_ch.shape)
				s_ch_pad = np.pad(s_ch, (0, SPS*2-len(s_ch) ), 'constant') #adding zero padding to the edges
				yfft = np.abs(fft(s_ch_pad))**2 #PSD
				for frq in [15, 30, 10, 20, 6, 12]: #for each frq and its double frq +-0.5Hz
					idx = [i for i in range(len(xfft)) if xfft[i] >= (frq-0.5) and xfft[i] <= (frq+0.5)]
					feat.append(np.sum(yfft[idx]))
			feat = np.array(feat)

			threadLock.acquire()
			featureQ.put([feat, ll]) #tt, ll])
			threadLock.release()

			##::END OF PROCESSING
			ECCA.set() #print('CCA')
			ESSVEP.clear() #print('s')
			ss = np.array([])
			tt = np.array([])
		else:
			break
	print('[SSVEP fun] :: [OVER]')

######################

def validate_classifier(feature_set, label_set, SSVEP_mode, splits, filename):
	##::Validation
	kf = KFold(n_splits = splits, random_state = 42, shuffle = True)
	accs = []
	count = 0

	f = open(filename, 'a')
	if SSVEP_mode and paradigmSSVEP:
		f.write('[VALIDATION RESULTS FOR SSVEP CLASSIFICATION]\n')
	elif not SSVEP_mode and paradigmP300:
		f.write('[VALIDATION RESULTS FOR P300 CLASSIFICATION]\n')

	for train_index, test_index in kf.split(feature_set):
		int_pred = []
		int_lbls = []
		bin_y_train = []
		rgt = 0
		if SSVEP_mode and paradigmSSVEP:
			print("[Training fold SSVEP -", count, "\b]")
			f.write("\n[Training fold SSVEP - {}]\n".format(count))
		elif not SSVEP_mode and paradigmP300:
			print("[Training fold P300 -", count, "\b]")
			f.write("\n[Training fold P300 - {}]\n".format(count))

		X_train, X_test = feature_set[train_index], feature_set[test_index]
		y_train, y_test = label_set[train_index], label_set[test_index]

		##::Fit data into model
		if SSVEP_mode:
			model = LDA_SSVEP_model.fit(X_train, y_train)
		else:
			model = LDA_P300_model.fit(X_train, y_train)

		##::Predict test
		y = model.predict(X_test)

		##::Conditioning
		int_pred = [int(yy) for yy in y]
		int_lbls = [int(x) for x in y_test]
		conf_mat = confusion_matrix(int_lbls, int_pred)
		f.write(">Pred: {}\n".format(int_pred))
		f.write(">Lbls: {}\n".format(int_lbls))

		##::Calculating accuracy
		for i in range(len(int_pred)):
			if int_pred[i] == int_lbls[i]:
				rgt += 1
		accs.append(rgt/len(int_lbls))

		#print & save results
		if SSVEP_mode and paradigmSSVEP:
			print(int_pred)		#predicted
			print(int_lbls)	#vs. lbls
			print(rgt, "labels match")
			print("[ACC LDA SSVEP = ", '{:.3f}'.format(accs[count]*100), "\b%]")
			f.write(">ACC LDA SSVEP = {:.3f}%\n>Confusion Matrix:\n{}\n\n".format(accs[count]*100, conf_mat))
		elif not SSVEP_mode and paradigmP300:
			print(int_pred)		#predicted
			print(int_lbls)	#vs. lbls
			print(rgt, "labels match")
			print("[ACC LDA P300  = ", '{:.3f}'.format(accs[count]*100), "\b%]")
			f.write(">ACC LDA P300 = {:.3f}%\n>Confusion Matrix:\n{}\n\n".format(accs[count]*100, conf_mat))
		count += 1
	accs_mean = np.mean(accs)
	accs_devi = np.std(accs)
	if SSVEP_mode and paradigmSSVEP:
		print("[Average ACC LDA SSVEP = ", '{:.3f}±{:.3f}'.format(accs_mean*100, accs_devi*100), "\b%]")
		f.write("\t[AVG VALIDATION RESULT]\n\t>Average ACC LDA SSVEP = {:.3f}±{:.3f}".format(accs_mean*100, accs_devi*100))
	elif not SSVEP_mode and paradigmP300:
		print("[Average ACC LDA P300  = ", '{:.3f}±{:.3f}'.format(accs_mean*100, accs_devi*100), "\b%]")
		f.write("\t[AVG VALIDATION RESULT]\n\t>Average ACC LDA P300 = {:.3f}±{:.3f}".format(accs_mean*100, accs_devi*100))

	f.close()
	return accs, accs_mean

######################

def P300_classifier_train(featureQ, modelQ, filename):
	feature_set = []
	label_set = []
	while not E.is_set():
		ELDA.wait(timeout = 13)
		if not E.is_set():
			while featureQ.qsize():
					try:
						feat, ll, pl = featureQ.get(0)  #pl is not used for training, only for prediction
					except Empty:
						return

			feature_set.append(feat)
			label_set.append(ll)

			ELDA.clear()
		else:
			break

	feature_set = np.array(feature_set)
	label_set = np.array(label_set)

	print('[Training LDA...]')
	print('[LDA feature_set ',feature_set.shape,']\n[LDA label_set ',label_set.shape,']')

	##Validation
	accs, accs_mean = validate_classifier(feature_set, label_set, 0, 10, filename)

	model = LDA_P300_model.fit(feature_set, label_set)
	threadLock.acquire()
	modelQ.put([model, accs_mean])
	threadLock.release()
	print('[Training finished]', model)


def SSVEP_classifier_train(featureQ, modelQ, filename):
	feature_set = []
	label_set = []
	rgt = 0

	while not E.is_set():
		ECCA.wait(timeout = 13)
		if not E.is_set():
			while featureQ.qsize():
					try:
						feat, ll = featureQ.get(0)
					except Empty:
						return

			feature_set.append(feat)
			label_set.append(ll)

			ECCA.clear() #print('cca')
		else:
			break

	feature_set = np.array(feature_set)
	label_set = np.array(label_set)

	print('[Training CCA...]')
	print('[CCA feature_set ',feature_set.shape,']\n[CCA label_set ',label_set.shape,']')

	##Validation
	accs, accs_mean = validate_classifier(feature_set, label_set, 1, 10, filename)

	model = LDA_SSVEP_model.fit(feature_set, label_set)
	threadLock.acquire()
	modelQ.put([model, accs_mean])
	threadLock.release()
	print('[Training finished]', model)

######################

def P300_classifier_predict(featureQ, modelQ, classOutQ, dataclassOutQ, filename):
	pred_set = []
	label_set = []
	pl_set = []
	rgt = 0
	if paradigmP300:
		f = open(filename, 'a')
		f.write('[LIVE RESULTS FOR P300 CLASSIFICATION]\n\n')
	while not E.is_set():
		ELDA.wait(timeout = 13)
		if not E.is_set():
			while featureQ.qsize():
					try:
						feat, ll, pl = featureQ.get(0)
					except Empty:
						return
			y = modelQ.predict(feat.reshape(1,-1))
			if paradigmP300:
				print('[P300 label', ll, '] vs. [', y[0],' predicted @', pl, ']')
			pred_set.append(y)
			label_set.append(ll)
			pl_set.append(pl)

			threadLock.acquire()
			dataclassOutQ.put([feat, ll, pl, y])
			threadLock.release()
			ELDA.clear()
			ELDAFUS.set()
		else:
			if paradigmP300:
				f.write('\n----\n\n')
			break

	##::Conditioning
	int_pred = [int(x) for x in pred_set]
	int_lbls = [int(x) for x in label_set]
	int_pl = [int(x) for x in pl_set]
	conf_mat = confusion_matrix(int_lbls, int_pred)

	##::Calculating accuracy
	for i in range(len(int_pred)):
		if int_pred[i] == int_lbls[i]:
			rgt += 1
	acc = rgt/len(int_lbls)

	if paradigmP300:
		f.write('\n\t[PREDICTION RESULTS]\n\t>Pred: {}\n'.format(int_pred))
		f.write('\t>Lbls: {}\n'.format(int_lbls))
		f.write('\t>PosL: {}\n'.format(int_pl))
		f.write("\t>Correct: {}/{}\n\t>ACC LDA P300 = {:.3f}%\n\t>Confusion Matrix:\n{}".format(rgt,len(int_lbls),acc*100, conf_mat))
		f.close()
	print('[End of P300 prediction]')


def SSVEP_classifier_predict(featureQ, modelQ, classOutQ, dataclassOutQ, filename):
	pred_set = []
	label_set = []
	rgt = 0
	if paradigmSSVEP:
		f = open(filename, 'a')
		f.write('[LIVE RESULTS FOR SSVEP CLASSIFICATION]\n\n')
	while not E.is_set():
		ECCA.wait(timeout = 13)
		if not E.is_set():
			while featureQ.qsize():
					try:
						feat, ll = featureQ.get(0)
					except Empty:
						return
			y = modelQ.predict(feat.reshape(1,-1))
			if paradigmSSVEP:
				print('[SSVEP label', ll, '] vs. [', y[0], ' predicted]')
			pred_set.append(y)
			label_set.append(ll)

			threadLock.acquire()
			dataclassOutQ.put([feat, ll, y])
			threadLock.release()
			ECCA.clear()
			ECCAFUS.set()
		else:
			if paradigmSSVEP:
				f.write('\n----\n\n')
			break

	##::Conditioning
	int_pred = [int(x) for x in pred_set]
	int_lbls = [int(x) for x in label_set]
	conf_mat = confusion_matrix(int_lbls, int_pred)

	##::Calculating accuracy
	for i in range(len(int_pred)):
		if int_pred[i] == int_lbls[i]:
			rgt += 1
	acc = rgt/len(int_lbls)

	if paradigmSSVEP:
		f.write('\n\t[PREDICTION RESULTS]\n\t>Pred: {}\n'.format(int_pred))
		f.write('\t>Lbls: {}\n'.format(int_lbls))
		f.write("\t>Correct: {}/{}\n\t>ACC LDA SSVEP = {:.3f}%\n\t>Confusion Matrix:\n{}".format(rgt,len(int_lbls),acc*100, conf_mat))
		f.close()
	print('[End of SSVEP prediction]')

######################
def fusion(Data_LDA_P300Q, Data_LDA_SSVEPQ, acc_LDA_P300, acc_LDA_SSVEP, filename):
	f = open(filename, 'a')
	if paradigmSSVEP and paradigmP300:
		f.write('[LIVE RESULTS FOR HYBRID CLASSIFICATION]\n\n')
	else:
		f.write('[LIVE RESULTS FOR SELECTION CLASSIFICATION]\n\n')

	feature_set = np.array([])
	label_set = np.array([])
	y_set = np.array([])
	predict_arr = []
	lbl_arr = []
	votes_log = []
	answer = []
	goal = 3
	timeout = P3NUM*3
	correct = 0
	pred_counter = 0
	una_pred = np.zeros((3,1), dtype = 'f')
	p300_lda_pred = np.zeros((3,1), dtype = 'f')
	ssvep_lda_pred = np.zeros((3,1), dtype = 'f')
	tot_pred = np.zeros((3,1), dtype = 'f')
	trials_count = 0
	cumul_clas = 1 #new
	prev_clas = -1 #new

	while not E.is_set():
		ELDAFUS.wait(timeout = 13)
		ECCAFUS.wait(timeout = 13)
		if not E.is_set():
			while Data_LDA_P300Q.qsize():
					try:
						feat_LDA, ll_LDA, pl_LDA, y_LDA = Data_LDA_P300Q.get(0)
					except Empty:
						return

			while Data_LDA_SSVEPQ.qsize():
					try:
						feat_CCA, ll_CCA, y_CCA = Data_LDA_SSVEPQ.get(0)
					except Empty:
						return

			y_SSVEP = int(y_CCA) #transform SSVEP label to int
			if prev_clas == y_SSVEP:			#new
				cumul_clas = cumul_clas*1.1 #new
			else:							#new
				prev_clas = y_SSVEP			#new
				cumul_clas = 1 				#new

			##Fusion
			if paradigmSSVEP and paradigmP300:
				if y_LDA == 1:
					if pl_LDA == y_SSVEP: #if position and CCA selection are the same
						una_pred[y_SSVEP] += 1*acc_LDA_P300 + 1*acc_LDA_SSVEP #new half value ea.
					else:
						p300_lda_pred[y_SSVEP] += acc_LDA_P300
						ssvep_lda_pred[y_SSVEP] += acc_LDA_SSVEP*cumul_clas #new
						ssvep_lda_pred[~(np.arange(len(ssvep_lda_pred)) == y_SSVEP)] -= 0.25*acc_LDA_SSVEP #new half value ea.
				elif y_LDA == 0:
					p300_lda_pred[pl_LDA] -= 0.25*acc_LDA_P300 #new half value ea.
					p300_lda_pred[~(np.arange(len(p300_lda_pred)) == pl_LDA)] += 0.125*acc_LDA_P300 #new half value ea.
					ssvep_lda_pred[y_SSVEP] += acc_LDA_SSVEP*cumul_clas #new
					ssvep_lda_pred[~(np.arange(len(ssvep_lda_pred)) == y_SSVEP)] -= 0.25*acc_LDA_SSVEP #new half value ea.
				pred_counter += 1
				tot_pred = np.sum([una_pred, p300_lda_pred, ssvep_lda_pred], 0)

				# ##::Print prediction arrays
				print(np.array([una_pred, p300_lda_pred, ssvep_lda_pred, tot_pred]).reshape(4,3, order = 'F').T)
				print("\n\n")
			
			##P300
			elif paradigmP300 and not paradigmSSVEP:
				if y_LDA == 1:
					tot_pred[pl_LDA] += 2*acc_LDA_P300
				else:
					tot_pred[pl_LDA] -= 0.5*acc_LDA_P300
					tot_pred[~(np.arange(len(tot_pred)) == pl_LDA)] += 0.25*acc_LDA_P300 #news
				pred_counter += 1

				# ##::Print prediction arrays
				print(np.array(tot_pred).T)
				print("\n\n")
			
			##SSVEP
			elif paradigmSSVEP and not paradigmP300:
				tot_pred[y_SSVEP] += 2*acc_LDA_SSVEP*cumul_clas #new
				tot_pred[~(np.arange(len(tot_pred)) == y_SSVEP)] -= 0.5*acc_LDA_SSVEP
				pred_counter += 1

				# ##::Print prediction arrays
				print(np.array(tot_pred).T)
				print("\n\n")


			if any(tot_pred[tot_pred >= goal]) or pred_counter >= timeout:
				voted_ans = np.argmax(tot_pred)

				#Sending feedback back to display
				if voted_ans == 0:
					EFeed1.set()
				elif voted_ans == 1:
					EFeed2.set()
				elif voted_ans == 2:
					EFeed3.set()
				EFeedback.set()

				threshold = tot_pred[voted_ans]
				predict_arr.append(voted_ans)
				lbl_arr.append(int(ll_CCA))
				votes_log.append(pred_counter)
				if ll_CCA == voted_ans:
					correct += 1
				print("Voted Answer  = {} ({})\t||\tRequired votes = {}".format(voted_ans, threshold[0], pred_counter))
				print("True Answer   = {}".format(ll_CCA))
				f.write('>Tot_pred arr: {}\n---\n'.format(tot_pred.T))

				#Reset counters
				una_pred = np.zeros((3,1))
				p300_lda_pred = np.zeros((3,1))
				ssvep_lda_pred = np.zeros((3,1))
				tot_pred = np.zeros((3,1))
				pred_counter = 0
				trials_count += 1

			ELDAFUS.clear()
			ECCAFUS.clear()
		else:
			break

	acc = correct/trials_count
	conf_mat = confusion_matrix(np.array(lbl_arr), np.array(predict_arr)) #, labels=[0, 1, 2])
	print("\n\nPred:\t", predict_arr)	#decimal predicted
	print("Lbls:\t", lbl_arr)  #vs. decimal true lbls
	print("#Votes:\t", votes_log) #votes needed for decision
	f.write('\n\t[SELECTION RESULTS]\n\t>Pred: {}\n'.format(predict_arr))
	f.write('\t>Lbls: {}\n'.format(lbl_arr))
	f.write("\t>#Vts: {}\n".format(votes_log))
	if paradigmSSVEP and paradigmP300:
		print("Fusion ACC: {:.3f}% | SSVEP LDA ACC: {:.3f}% | P300 LDA ACC: {:.3f}%\nCCA Confusion Matrix\n{}".format(acc*100, acc_LDA_SSVEP*100, acc_LDA_P300*100, conf_mat))
		f.write("\t>Fusion ACC = {:.3f}% | Offline SSVEP LDA ACC = {:.3f}% | Offline P300 LDA ACC = {:.3f}%\n\t>Confusion Matrix:\n{}".format(acc*100, acc_LDA_SSVEP*100, acc_LDA_P300*100, conf_mat))
	elif paradigmP300 and not paradigmSSVEP:
		print("Selection ACC: {:.3f}% | P300 LDA ACC: {:.3f}%\nCCA Confusion Matrix\n{}".format(acc*100, acc_LDA_P300*100, conf_mat))
		f.write("\t>Selection ACC = {:.3f}% | Offline P300 LDA ACC = {:.3f}%\n\t>Confusion Matrix:\n{}".format(acc*100, acc_LDA_P300*100, conf_mat))
	elif not paradigmP300 and paradigmSSVEP:
		print("Selection ACC: {:.3f}% | SSVEP LDA ACC: {:.3f}%\nCCA Confusion Matrix\n{}".format(acc*100, acc_LDA_SSVEP*100, conf_mat))
		f.write("\t>Selection ACC = {:.3f}% | Offline SSVEP LDA ACC = {:.3f}%\n\t>Confusion Matrix:\n{}".format(acc*100, acc_LDA_SSVEP*100, conf_mat))
	f.close()


##::EXECUTION
if __name__ == '__main__':
	
	print('::Session parameters::')
	init_dialog()
	FlkWin = FlickeringBoxes([0,0])

	##::True Labels for the trials
	TrialLabels = []
	for n in range (0,3):
		for k in range(0,TRIALNUM):
			TrialLabels.append(n)
	random.shuffle(TrialLabels)
	print("[Trial labels Sequence]",TrialLabels)

	if session_mode == 1 or session_mode == 2:
		print(u'[Current path][%s]\n[Partic path][%s]\n[CSV filename][%s]\n[Clas. filename][%s]\n[Paradigm][%s]\n[Num Trials][%d]\n[Num Stim][%d]' 
				% (_currDir, _partDir, _csvsession, _classifile, paradigm, TRIALNUM, P3NUM))
		classifile_path__ = os.path.join(_partDir, _classifile)
		LDA_P300_path__ = os.path.join(_partDir, _classioutLDA_P300)
		LDA_SSVEP_path__ = os.path.join(_partDir, _classioutLDA_SSVEP)
		LivePerf_path__ = os.path.join(_partDir, _LivePerformance)
		if session_mode	== 1:
			if os.path.isfile(classifile_path__):
				warningDlg = gui.Dlg(title = "File name already exists", labelButtonOK = 'Yes', labelButtonCancel = 'Cancel')
				warningDlg.addText('Another classification model file already exists in this folder.')
				warningDlg.addText('Do you want to rewrite with new file?')
				ok_data = warningDlg.show()
				if not warningDlg.OK:
					core.quit()

		##::Definitions
		LDA_P300_model = LDA(solver = 'svd')
		LDA_SSVEP_model = LDA(solver = 'svd')

		threadLock = threading.Lock()
		SaveQ = queue.Queue()
		SaveFeatQ = queue.Queue()
		DataOutQ1 = queue.Queue()
		DataOutQ2 = queue.Queue()
		FeatQ1 = queue.Queue()
		FeatQ2 = queue.Queue()
		ModelQ1 = queue.Queue()
		ModelQ2 = queue.Queue()
		ClassOutQ1 = queue.Queue()
		ClassOutQ2 = queue.Queue()
		DataClassOutQ1 = queue.Queue()
		DataClassOutQ2 = queue.Queue()
		#
		E = threading.Event()
		HOLD = threading.Event()
		P300TRIG = threading.Event()
		COLLECTTRIG = threading.Event()
		EP300 = threading.Event()
		ESSVEP = threading.Event()
		#
		ELDA = threading.Event()
		ECCA = threading.Event()
		ELDAFUS = threading.Event()
		ECCAFUS = threading.Event()
		#
		EPLabel = threading.Event()
		EPos1Label = threading.Event()
		EPos2Label = threading.Event()
		EPos3Label = threading.Event()
		ES1Label = threading.Event()
		ES2Label = threading.Event()
		ES3Label = threading.Event()
		lockcounter = False
		#
		EFeedback = threading.Event()
		EFeed1 = threading.Event()
		EFeed2 = threading.Event()
		EFeed3 = threading.Event()
		#
		AcqTh = AcqThread(DataOutQ1, DataOutQ2, SaveQ)
		P300Th = P300Thread(DataOutQ1, FeatQ1)
		SSVEPTh = SSVEPThread(DataOutQ2, FeatQ2)
		#
		if session_mode == 1:
			P300_LDA_Th = P300_Class_Thread(FeatQ1, ModelQ1, ClassOutQ1, DataClassOutQ1, session_mode, LDA_P300_path__)
			SSVEP_LDA_Th = SSVEP_Class_Thread(FeatQ2, ModelQ2, ClassOutQ2, DataClassOutQ2, session_mode, LDA_SSVEP_path__)
		elif session_mode == 2:
			#load classifiers and accuracies
			classifile_path__ = os.path.join(_partDir, _classifile)
			if os.path.isfile(classifile_path__):
				with open(classifile_path__, 'rb') as input:
					cca_model, lda_model, accs_CCA, accs_LDA = pickle.load(input)
				P300_LDA_Th = P300_Class_Thread(FeatQ1, lda_model, ClassOutQ1, DataClassOutQ1, session_mode, LDA_P300_path__)
				SSVEP_LDA_Th = SSVEP_Class_Thread(FeatQ2, cca_model, ClassOutQ2, DataClassOutQ2, session_mode, LDA_SSVEP_path__)
				FusTh = Fusion_Thread(DataClassOutQ1, DataClassOutQ2, accs_LDA, accs_CCA, LivePerf_path__)
			else:
				notFoundDlg = gui.Dlg(title = "Classifile not found", labelButtonOK = 'Ok', labelButtonCancel = 'Well, ok')
				notFoundDlg.addText('There is no classifile for this Participant in this session.')
				notFoundDlg.addText('Please, run a Training Set first.')
				ok_data = notFoundDlg.show()
				if notFoundDlg.OK or not notFoundDlg.OK:
					core.quit()

		##::CONNECTION
		BoardShim.enable_dev_board_logger()
		params = BrainFlowInputParams ()
		## BOARDS
		if headset == 'cyton':
		#CYTON
			params.serial_port = 'COM5'#'COM3' #'COM11'
			board_id = BoardIds.CYTON_BOARD.value
		elif headset == 'gtec':
		#GTEC
			board_id = BoardIds.UNICORN_BOARD.value
			params.serial_number = 'UN-2019.05.08'

		board = BoardShim (board_id, params)
		print('|Looking for an EEG stream...')
		SPS = board.get_sampling_rate(board_id)
		print('[Sample Rate][', SPS,']')
		NYQ = int(SPS*0.5)
		board.prepare_session() #connect to board
		print('|Connected\n--\n')
		if session_mode == 0:
			csvfile_path__ = os.path.join(_partDir, _csvpractice)
			imagefile_path__ = os.path.join(_partDir, _imgpractice)
		else:
			csvfile_path__ = os.path.join(_partDir, _csvsession)
			imagefile_path__ = os.path.join(_partDir, _imgsession)
			# if session_mode ==2:
			# 	classioutCCA_path__ = os.path.join(_partDir, _classioutCCA)
			# 	classioutLDA_path__ = os.path.join(_partDir, _classioutLDA)
		#new
		csvsnippet_path__ = os.path.join(_partDir, _csvsippets)

		board.start_stream(45000, 'file://'+csvfile_path__+':w') #streams data and saves it to csv file
		print('|Streamming file assigned\n--\n')

		time.sleep(5)

		print('[Starting Threads]')

		##::START FLICKER AND ACQUISITION
		AcqTh.start()
		P300Th.start()
		SSVEPTh.start()
		SSVEP_LDA_Th.start()
		P300_LDA_Th.start()

		if session_mode == 1:
			print('[Starting training display]')
			FlkWin.training_sequence()
		elif session_mode == 2:
			print('[Starting Fusion]')
			FusTh.start()
			print('[Starting live display]')
			FlkWin.live_sequence()

		AcqTh.join()
		SSVEP_LDA_Th.join()
		P300_LDA_Th.join()
		if session_mode == 2:
			print('[Joining Fusion]')
			FusTh.join()
		print('[Threads joined]')

		
		board.stop_stream()
		print('[Board stream stopped]')
		board.release_session()
		print('[Session released]')

		if SaveQ.qsize():
			while SaveQ.qsize():
				try:
					y, yf, X, tss = SaveQ.get(0) #<< [thesamples, thefilterd, [thetimestamps, thefilttimestamps], [thesamples[:,:], thefilterd[:,:], thetimestamps[:], tstampsP300]]
				except Empty:
					pass

			x = X[0]
			xf = X[1]

			#new
			write_vec = []
			tot = np.array(y).shape[0] #*np.array(y).shape[2]
			t = 0
			with open(csvsnippet_path__, 'w', newline = "") as f:
				writer = csv.writer(f)
				Y = np.array(y)
				X = np.array(x)
				print("\n\nArray sizes:\nY:", Y.shape,'\nX:', X.shape)
				for i in range(np.array(x).shape[0]): 			#number of snippets
					write_vec = []
					write_vec = np.hstack((Y[i,:,:].T, X[i,:].reshape(-1,1),))
					writer.writerows(write_vec)
					# 	progress bar
					msg = "[Saving Snippets |" + '\u2588' * math.floor( ((t/tot*100) % 100)/2 ) + '\u2591' * math.ceil(50-( ((t/tot*100) % 100)/2 )) + '] [Elem ' + str(t) + ' ~ '+'{:.2f}%'.format(t/tot*100) + ']'
					print(msg, end="\r")
					t += 1
				print('')

			

			##Saving P300 timestamps
			#Read Excel file as a DataFrame
			df = pd.read_csv(csvfile_path__)

			#Atributte names to columns
			print('[Labeling offline data]')
			if headset == 'cyton':
				df.columns = ['Samples', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'AccelX', 'AccelY', 'AccelZ', 
				'Empty0', 'Empty1', 'Empty2', 'Empty3', 'Empty4', 'Empty5', 'Empty6', 'Empty7', 'Empty8', 'Empty9', 'Timestamps', 'Empty10', 'Empty11']
			elif headset == 'gtec':
				df.columns = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'AccelX', 'AccelY', 'AccelZ', 
				'Empty0', 'Empty1', 'Empty2', 'Empty3', 'Samples', 'Empty4', 'Timestamps', 'Empty6', 'Empty']
			#Zero P300_ts column
			df['P300_ts'] = -1
			df['P300_labels'] = -1
			df['P300_position'] = -1
			df['SSVEP_labels'] = -1
			k = 0
			#Find columns with timestamps
			for cl in df['Timestamps']:
				#tss => [ini_stim_ts, end_stim_ts, ll, int_TL, pl]
				for val in tss:
					t0, t1, ll, TL, pl = val[0], val[1], val[2], val[3], val[4]
					if '{:.6f}'.format(cl) == '{:.6f}'.format(t0):
						df.loc[df['Timestamps'] == cl, 'P300_ts'] = 0
						df.loc[df['Timestamps'] == cl, 'P300_labels'] = ll
						df.loc[df['Timestamps'] == cl, 'P300_position'] = pl
						df.loc[df['Timestamps'] == cl, 'SSVEP_labels'] = TL

					if '{:.6f}'.format(cl) == '{:.6f}'.format(t1):
						df.loc[df['Timestamps'] == cl, 'P300_ts'] = 2

				#progress bar
				msg = "[Progress |" + '\u2588' * math.floor( ((k/df['Timestamps'].size*100) % 100)/2 ) + '\u2591' * math.ceil(50-( ((k/df['Timestamps'].size*100) % 100)/2 )) + '] [Elem ' + str(k) + ' ~ '+'{:.2f}%'.format(k/df['Timestamps'].size*100) + ']'
				print(msg, end="\r")
				k += 1
			print('')

			# #To save it back to csv
			df.to_csv(csvfile_path__)

		##::Saving Classification models
		if session_mode == 1:
			print('[Saving classification models...]')
			flag1, flag2 = 1, 1
			if ModelQ1.qsize():
				while ModelQ1.qsize():
					try:
						lda_model, accs_LDA = ModelQ1.get(0)
						print("[P300 LDA Model][OK]")
					except Empty:
						print("[ERROR!!!][P300 LDA Model ModelQ empty]")
						flag1 = 0
					except Exception as ex:
						template = "An exception of type {0} occurred. Arguments:\n{1!r}"
						message = template.format(type(ex).__name__, ex.args)
						print(message)
						print("[ERROR!!!][P300 LDA Model not found]")
						flag1 = 0
			else:
				print("[ERROR!!!][P300 LDA Model QSIZE empty]")
				flag1 = 0

			if ModelQ2.qsize():
				while ModelQ2.qsize():
					try:
						cca_model, accs_CCA = ModelQ2.get(0)
						print("[SSVEP LDA Model][OK]")
					except Empty:
						print("[ERROR!!!][SSVEP LDA Model ModelQ empty]")
						flag2 = 0
					except Exception as ex:
						template = "An exception of type {0} occurred. Arguments:\n{1!r}"
						message = template.format(type(ex).__name__, ex.args)
						print(message)
						print("[ERROR!!!][SSVEP LDA Model not found]")
						flag2 = 0
			else:
				print("[ERROR!!!][SSVEP LDA Model QSIZE empty]")
				flag2 = 0

			##SAVE classifier models with pickle
			if flag1 and flag2:
				with open(classifile_path__, 'wb') as output:
					pickle.dump([cca_model, lda_model, accs_CCA, accs_LDA], output, pickle.HIGHEST_PROTOCOL)
					print("[Models saved]")
			else:
				print("[ERROR!!!][Models not saved]")
		
		##::Saving Classification Results
		elif session_mode == 2:
			print('[Saving classification data...]')
			if DataClassOutQ1.qsize():
				while DataClassOutQ1.qsize():
					try:
						feature_set, label_set, y_set = DataClassOutQ1.get(0)
					except Empty:
						pass
				print('LDA outputs: Feat_set', feature_set.shape, 'Label_set:', label_set.shape, 'y_set:', len(y_set) )
				#### May need stacking adjustments before saving
				# with open(classioutLDA_path__, 'w', newline='') as file:
				# 	writer = csv.writer(classioutLDA_path__)
				# 	writer.writerows([label_set, y_set, feature_set])

			if DataClassOutQ2.qsize():
				while DataClassOutQ2.qsize():
					try:
						feature_set, label_set, y_set = DataClassOutQ2.get(0)
					except Empty:
						pass
				print('CCA outputs: Feat_set', feature_set.shape, 'Label_set:', label_set.shape, 'y_set:', len(y_set) )
				#### May need stacking adjustments before saving
				# with open(classioutCCA_path__, 'w', newline='') as file:
				# 	writer = csv.writer(classioutCCA_path__)
				# 	writer.writerows([label_set, y_set, feature_set])


		print("TIMESTAMPS  : ", x.shape)
		print("TIMESTAMPS F: ", xf.shape)
		print("DATA RAW    : ", y.shape)
		print("DATA FILTERD: ", yf.shape)

	elif session_mode == 0:
		print(u'\n[Session Mode][%s]\n[Paradigm][%s]\n[Num Trials][%d]\n[Num Stim][%d]' % (session_mode, paradigm, TRIALNUM, P3NUM))
		FlkWin.practice_sequence() #practice mode does not require data collection


	print("\n\n[[DONE]]\n\n")
	FlkWin.stop()
