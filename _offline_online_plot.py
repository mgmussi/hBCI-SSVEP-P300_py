from tkinter import filedialog
import pandas as pd
pd.options.display.float_format = '{:20.6f}'.format
import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
import matplotlib.pyplot as plt

if __name__ == '__main__':
	offline_125 = []
	online_124 = []
	offline_ts = []
	p300ll_data = []
	p300pos_data = []
	ssvepll_data = []
	curr_id = 0
	#
	SPS = 250
	NYQ = int(SPS*0.5)
	SLICE = NYQ
	#

	filename1 = filedialog.askopenfilenames(initialdir = "C:/Users/atech/Documents/GitHub/SSVEP_P300-py/Participants",
		title = "Select <Saved Data> File", filetypes = (("Comma Separated Values", "*.csv*"), ("all files","*.*")))

	filename2 = filedialog.askopenfilenames(initialdir = "C:/Users/atech/Documents/GitHub/SSVEP_P300-py/Participants",
		title = "Select <Saved Snippet> File", filetypes = (("Comma Separated Values", "*.csv*"), ("all files","*.*")))

	##::<Saved Data> File
	print(filename1[0])
	df1 = pd.read_csv(filename1[0])

	## Get df1 indexes:
	idx1 = df1.index[df1['P300_ts'] == 0]
	idx2 = df1.index[df1['P300_ts'] == 2]

	## Extract data from DataFrame
	counter = 0
	for i, j in zip(idx1, idx2):
		## Get the last 125 samples
		offline_125.append([
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH1']].values],
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH2']].values],
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH3']].values],
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH4']].values],
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH5']].values],
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH6']].values],
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH7']].values],
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH8']].values]])

		## Get labels for each sample
		offline_ts.append([t[0] for t in list(df1.loc[j-NYQ+1:j, ['Timestamps']].values)])
		p300 = df1.loc[i, ['P300_labels']].values
		p300ll_data.append(p300[0])
		p300pos = df1.loc[i, ['P300_position']].values
		p300pos_data.append(p300pos[0])
		ssvep = df1.loc[i, ['SSVEP_labels']].values
		ssvepll_data.append(ssvep[0])

	offline_125 = np.array(offline_125)
	offline_ts = np.array(offline_ts)
	p300ll_data = np.array(p300ll_data)
	p300pos_data = np.array(p300pos_data)
	ssvepll_data = np.array(ssvepll_data)


	##::<Saved Snippet> File
	print(filename2[0])
	df2 = pd.read_csv(filename2[0])

	for i in range(0, df2.shape[0], 125):
		print(i)


	# return offline_125, offline_ts, p300ll_data, p300pos_data, ssvepll_data