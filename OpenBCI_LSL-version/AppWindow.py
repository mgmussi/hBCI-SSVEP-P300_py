
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import scroller as scrl
import logging
import requests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #NavigationToolbar2TkAgg <<Did not work
import random
from pandas import DataFrame
# import stream_lsl_eeg_simplified as leeg
import stream_lsl_eeg_simplified as leeg
from "FlickeringBoxes-2020-11-23" import *
import numpy as np
import cProfile, pstats, io


#Definitions
H = 720
W = 1280

#Color palette>> https://www.color-hex.com/color-palette/92077
bg_color = "#efeff2"
sc_color = "#dadae3"
tx_color = "#000000"
dt_color = "#a69eb0"
wn_color = "#f2e2cd"


class AppWindow:
	def toggleLock(self):
		#see that the name is changed first, that's why it's inverted
		if self.btn_chconf['text'] == 'Change Channels': # if self.lck_var.get() == 1:
			for j in range(1,9):
				eval('self.ch%d_cb'% j).config(state = tk.DISABLED)

			#this part of the code is exclusively to use set_data
			self.lines = []
			numchan = 0
			for ch in self.chk_var:
				if ch.get():
					numchan += 1
			print("Num chan:", numchan)
			for index in range(numchan):
				lobj = self.ax.plot([],[], lw=2, color=tx_color)[0]
				self.lines.append(lobj)

			# self.logger.warn('Channels Locked')
		elif self.btn_chconf['text'] == 'Confirm Channels': # elif self.lck_var.get() == 0:
			for j in range(1,9):
				eval('self.ch%d_cb'% j).config(state = tk.NORMAL)
			# self.logger.warn('Channels Unlocked')

	def openFlick(self):
		self.logger.warn('\nInitializing Flickering...')
		w = self.root.winfo_screenwidth()
		h = self.root.winfo_screenheight()
		x = (w/2) - W/6 +5#- W/2 + W/3
		y = (h/2) - H/2 +45
		FlkWin = FlickeringBoxes([x,y])
		FlkWin.start()


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

	# @profile
	def plotGraph(self, x, y, mycolor1 = tx_color):
		self.ax.clear()
		if isinstance(y, list):
			# plot data alternatively
			self.ax.plot(x, y, color = mycolor)
		elif isinstance(y, np.ndarray):

			# working plot
			self.ax.plot(x, y, color = mycolor1)

			# new plotting plugin

			# # Test 1 set_data
			# for lnum,line in enumerate(self.lines):
			# 	line.set_data(x[:], y[:, lnum])

			# # Test 2 set_data
			# for lnum,line in enumerate(self.lines):
			# 	# print(x, y[:, lnum])
			# 	self.lines[lnum].set_ydata(y[:, lnum])
			# 	self.lines[lnum].set_xdata(x)


			self.figure.canvas.draw_idle()
			plt.ylabel('Magnitude', fontsize = 9, color = tx_color)
			plt.xlabel('Time', fontsize = 9, color = tx_color)
		self.figure.canvas.draw()
###########
	def plotGraphS(self, x, y, z, w, mycolor1 = tx_color, mycolor2 = 'red'):
		self.ax.clear()
		if isinstance(y, list):
			# plot data alternatively
			self.ax.plot(x, y, color = mycolor)
		elif isinstance(y, np.ndarray):

			# working plot
			self.ax.plot(x, y, color = mycolor1)
			self.ax.plot(z, w, color = mycolor2)

			# new plotting plugin


			# # Test 1 set_data
			# for lnum,line in enumerate(self.lines):
			# 	line.set_data(x[:], y[:, lnum])

			# # Test 2 set_data
			# for lnum,line in enumerate(self.lines):
			# 	# print(x, y[:, lnum])
			# 	self.lines[lnum].set_ydata(y[:, lnum])
			# 	self.lines[lnum].set_xdata(x)


			self.figure.canvas.draw_idle()
			plt.ylabel('Magnitude', fontsize = 9, color = tx_color)
			plt.xlabel('Time', fontsize = 9, color = tx_color)
		self.figure.canvas.draw()



	def __init__(self):
		##Application Design
		self.root = tk.Tk() #start of application
		self.root.wm_title("Hybrid BCI - SSVEP and Eye Tracker")
		ws = self.root.winfo_screenwidth()
		hs = self.root.winfo_screenheight()
		x = (ws/2) - W/2
		y = (hs/2) - H/2
		self.root.geometry('%dx%d+%d+%d' % (W, H, x, y))

		self.canvas = tk.Canvas(self.root, height = H, width = W, bg = bg_color, highlightthickness=0)
		self.canvas.pack(fill = 'both', expand = True)
		#
		##Side panel
		self.ctr_frame = tk.LabelFrame(self.canvas, text = "Control Panel", fg = tx_color, bg = sc_color)
		self.ctr_frame.place(relx = 0, width = W/3, relheight = 1)
		#
		##Frame for the SSVEP Stuff
		self.EEG_frame = tk.LabelFrame(self.ctr_frame, text = "SSVEP", fg = tx_color, bg = sc_color, borderwidth = 0)
		self.EEG_frame.place(anchor = 'nw', relx = 0.01, rely = 0.02, relwidth = 0.98, relheight = 0.23)
		#Button for Getting EEG LSLs inlet
		self.btn_StartEEGLSL = tk.Button(self.EEG_frame, text = "Start EEG LSL Inlet", bg = bg_color, fg = tx_color, state = tk.DISABLED, command = lambda: leeg.getInlet(self))
		self.btn_StartEEGLSL.place(anchor = 'nw', relx = 0.52, rely = 0.15, width = 196, height = 40)
		#Button to test the SSVEP only w/ EEG
		self.btn_txt_Receive = tk.StringVar()
		self.btn_ReceiveEEG = tk.Button(self.EEG_frame, text = "Receive EEG signal", bg = bg_color, fg = tx_color, state = tk.DISABLED, command = lambda: leeg.getEEGstream(self))
		self.btn_txt_Receive.set("Receive EEG signal")
		self.btn_ReceiveEEG.place(anchor = 'nw', relx = 0.52, rely = 0.5, width = 196, height = 40)
		#Channel option selection
		self.chn_frame = tk.LabelFrame(self.EEG_frame, text = "EEG Channels", bg = sc_color, fg = tx_color, borderwidth = 1)
		self.chn_frame.place(anchor = 'ne', relx = 0.48, rely = 0.10, width = 185, height = 115)
		#Un/Lock option
		# SZ = 80
		# self.lck_open = tk.PhotoImage(file = "img/locker_open.png") #, width = 50, height = 50)
		# self.lck_close = tk.PhotoImage(file = "img/locker_close.png")
		# self.img_lck_open = self.lck_open.subsample(SZ)
		# self.img_lck_close = self.lck_close.subsample(SZ)
		# self.lck_var = tk.IntVar()
		# self.lck_cb = tk.Checkbutton(self.chn_frame,image = self.img_lck_open, selectimage = self.img_lck_close,
		# 	indicatoron = False, onvalue = 1, offvalue = 0, bg = sc_color, borderwidth = 0,
		# 	bd = 0, padx = 0, pady = 0, selectcolor = wn_color, variable = self.lck_var, command = lambda: self.toggleLock())
		# self.lck_cb.place(anchor = "ne", relx = 1.01, rely = -0.15)
		#Channels Checkboxes
		self.chk_var = []
		for i in range(8):
			self.chk_var.append(tk.IntVar())
		#
		self.ch1_cb = tk.Checkbutton(self.chn_frame, text = "Channel 1", bg = sc_color, fg = tx_color, padx = 0, pady = 0, activebackground = sc_color, variable = self.chk_var[0])
		self.ch1_cb.place(anchor = "nw", relx = 0.01, rely = 0)
		#
		self.ch2_cb = tk.Checkbutton(self.chn_frame, text = "Channel 2", bg = sc_color, fg = tx_color, padx = 0, pady = 0, activebackground = sc_color, variable = self.chk_var[1])
		self.ch2_cb.place(anchor = "nw", relx = 0.01, rely = 0.2)
		#
		self.ch3_cb = tk.Checkbutton(self.chn_frame, text = "Channel 3", bg = sc_color, fg = tx_color, padx = 0, pady = 0, activebackground = sc_color, variable = self.chk_var[2])
		self.ch3_cb.place(anchor = "nw", relx = 0.01, rely = 0.4)
		#
		self.ch4_cb = tk.Checkbutton(self.chn_frame, text = "Channel 4", bg = sc_color, fg = tx_color, padx = 0, pady = 0, activebackground = sc_color, variable = self.chk_var[3])
		self.ch4_cb.place(anchor = "nw", relx = 0.01, rely = 0.6)
		#
		self.ch5_cb = tk.Checkbutton(self.chn_frame, text = "Channel 5", bg = sc_color, fg = tx_color, padx = 0, pady = 0, activebackground = sc_color, variable = self.chk_var[4])
		self.ch5_cb.place(anchor = "nw", relx = 0.5, rely = 0)
		#
		self.ch6_cb = tk.Checkbutton(self.chn_frame, text = "Channel 6", bg = sc_color, fg = tx_color, padx = 0, pady = 0, activebackground = sc_color, variable = self.chk_var[5])
		self.ch6_cb.place(anchor = "nw", relx = 0.5, rely = 0.2)
		#
		self.ch7_cb = tk.Checkbutton(self.chn_frame, text = "Channel 7", bg = sc_color, fg = tx_color, padx = 0, pady = 0, activebackground = sc_color, variable = self.chk_var[6])
		self.ch7_cb.place(anchor = "nw", relx = 0.5, rely = 0.4)
		#
		self.ch8_cb = tk.Checkbutton(self.chn_frame, text = "Channel 8", bg = sc_color, fg = tx_color, padx = 0, pady = 0, activebackground = sc_color, variable = self.chk_var[7])
		self.ch8_cb.place(anchor = "nw", relx = 0.5, rely = 0.6)
		#Channel confirm Button
		self.btn_chconf = tk.Button(self.chn_frame, text = "Confirm Channels", bg = bg_color, fg = tx_color, command = lambda: leeg.assignChan(self, self.chk_var))
		self.btn_chconf.place(anchor = 'sw', relx = 0, rely =1, relwidth = 1, height = 15)
		#
		##Frame for the EyeTracking (ET) stuff
		self.ET_frame = tk.LabelFrame(self.ctr_frame, text = "Eye Tracker", fg = tx_color, bg = sc_color, borderwidth = 0)
		self.ET_frame.place(anchor = 'nw', relx = 0.01, rely = 0.25, relwidth = 0.98, relheight = 0.15)
		#User Name
		self.usr_lbl = tk.Label(self.ET_frame, fg = tx_color, bg = sc_color, text = "User name")
		self.usr_lbl.place(anchor = 'nw', relx = 0.02, rely = 0.05)
		self.user_name = tk.Entry(self.ET_frame, fg = tx_color, bd = 2)
		self.user_name.place(anchor = 'ne', relx = 0.5, rely = 0.25, width = 200)
		#Calibrate Button
		self.btn_clb = tk.Button(self.ET_frame, text ="Calibrate User", bg = bg_color, fg = tx_color)
		self.btn_clb.place(anchor = 'nw', relx = 0.5, rely = 0.26, width = 206)
		#Flickering boxes Button
		self.btn_flk = tk.Button(self.ET_frame, text = "Flickering Boxes Test", bg = bg_color, fg = tx_color, command = lambda: self.openFlick())
		self.btn_flk.place(anchor = 'nw', relx = 0.02, rely = 0.6, relwidth = 0.98)
		#
		##Frame for SSVEP+ET
		self.SSET_frame = tk.LabelFrame(self.ctr_frame, text = "SSVEP and Eye Tracker", fg = tx_color, bg = sc_color, borderwidth = 0)
		self.SSET_frame.place(anchor = 'nw', relx = 0.01, rely = 0.4, relwidth = 0.98, relheight = 0.10)
		#SSVEP & EyeGaze Button
		self.btn_experim = tk.Button(self.SSET_frame, text = "SSVEP + EyeGaze", bg = wn_color, fg = 'white', state = tk.DISABLED)
		self.btn_experim.place(anchor = 'nw', relx = 0.02, rely = 0.01, relwidth = 0.98, relheight = 0.95)
		#
		##Frame for Graphic stuff
		self.Grp_frame = tk.LabelFrame(self.ctr_frame, text = "Graphic", fg = tx_color, bg = sc_color, borderwidth = 0)
		self.Grp_frame.place(anchor = 'nw', relx = 0.01, rely = 0.52, relwidth = 0.98, relheight = 0.23)
		#create figure/graph
		self.figure = plt.figure(figsize = (5,6), dpi = 100)
		self.figure.patch.set_facecolor(sc_color)
		self.ax = self.figure.add_subplot(111)
		self.ax.set_title('FFT')
		self.ax.set_facecolor(dt_color)
		self.ax.spines["top"].set_linewidth(1)
		self.ax.spines["bottom"].set_linewidth(1)
		self.ax.spines["left"].set_linewidth(1)
		self.ax.spines["right"].set_linewidth(1)
		self.ax.set_xmargin(2)
		matplotlib.rcParams['text.color'] = tx_color
		matplotlib.rcParams['xtick.color'] = tx_color
		matplotlib.rcParams['ytick.color'] = tx_color
		matplotlib.rcParams['axes.labelcolor'] = tx_color
		matplotlib.rcParams['font.size'] =  6.5
		plt.subplots_adjust(bottom=0.31, left=0.136, top=0.9, right=0.99)
		self.ax.clear()
		#To use set_data
		self.line, = self.ax.plot([], [], lw=1, color = tx_color)
		self.line.set_data([],[])
		# self.figure.canvas.draw()

		#place graph
		self.chart_type = FigureCanvasTkAgg(self.figure, self.Grp_frame)
		self.chart_type.get_tk_widget().place(anchor = 'nw', relx = 0.02, rely = 0, relwidth = 0.98, relheight = 0.98)
		#
		##Frame Prompt
		self.Grp_frame = tk.LabelFrame(self.ctr_frame, text = "Log", fg = tx_color, bg = sc_color, borderwidth = 0)
		self.Grp_frame.place(anchor = 'nw', relx = 0.01, rely = 0.76, relwidth = 0.98, relheight = 0.23)
		#Create a Scrolled Text box
		self.log_wn = ScrolledText(self.Grp_frame, state='disabled', fg = bg_color, bg = tx_color)
		self.log_wn.configure(font='TkFixedFont')
		self.log_wn.place(anchor = 'nw', relx = 0.02, rely = 0.01, relwidth = 0.98, relheight = 0.98)
		# Create textLogger
		self.text_handler = scrl.TextHandler(self.log_wn)
		# Add the handler to logger
		self.logger = logging.getLogger()
		self.logger.addHandler(self.text_handler)
		self.logger.warn('Ready')

		
	def start(self):
		self.root.mainloop() #end of application