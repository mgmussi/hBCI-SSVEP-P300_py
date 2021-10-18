from tkinter import filedialog
import pandas as pd
pd.options.display.float_format = '{:20.6f}'.format
import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

if __name__ == '__main__':
	offline_125 = []
	online_125 = []
	offline_ts = []
	online_ts = []
	p300ll_data = []
	p300pos_data = []
	ssvepll_data = []
	curr_id = 0
	#
	SPS = 250
	NYQ = int(SPS*0.5)
	SLICE = NYQ
	#
	#
	#
	filename = filedialog.askopenfilenames(initialdir = "C:/Users/atech/Documents/GitHub/SSVEP_P300-py/Participants",
		title = "Select \'Saved Data\' File", filetypes = (("Comma Separated Values", "*.csv*"), ("all files","*.*")))
	
	end_str = filename[0].rindex('_')
	if filename[0][end_str:] == '_savedsnippet.csv':
		filename2 = filename[0]
		filename1 = filename[0][:end_str] + '_saveddata.csv'
	elif filename[0][end_str:] == '_saveddata.csv':
		filename1 = filename[0]
		filename2 = filename[0][:end_str] + '_savedsnippet.csv'
	#
	#
	#
	##::<Saved Data> File
	print('\n>[Extracting file] ', filename1)
	df1 = pd.read_csv(filename1)

	## Get df1 indexes:
	idx1 = df1.index[df1['P300_ts'] == 0]
	idx2 = df1.index[df1['P300_ts'] == 2]

	## Extract data from DataFrame
	counter = 0
	for i, j in zip(idx1, idx2):
		## Get the last 125 samples
		offline_125.append([
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH2']].values],
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH3']].values],
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH4']].values],
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH5']].values],
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH6']].values],
			[x[0] for x in df1.loc[j-SLICE+1:j, ['CH7']].values]])

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
	print("<<OFFLINE ARRAYS>>")
	print(offline_125.shape)
	print(offline_ts.shape)
	#
	#
	#
	##::<Saved Snippet> File
	print('\n>[Extracting file] ', filename2)
	df2 = pd.read_csv(filename2, header=None)
	df2.columns = ['CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'Timestamp']
	# print(df2.head(126), df2.shape)

	for i in range(0, df2.shape[0], SLICE):
		# print("Analyzing vals from: ", i, "to", i+SLICE-1)
		online_125.append([
			[x for x in df2.iloc[i:i+SLICE, 0].values],
			[x for x in df2.iloc[i:i+SLICE, 1].values],
			[x for x in df2.iloc[i:i+SLICE, 2].values],
			[x for x in df2.iloc[i:i+SLICE, 3].values],
			[x for x in df2.iloc[i:i+SLICE, 4].values],
			[x for x in df2.iloc[i:i+SLICE, 5].values]])
		online_ts.append([t for t in df2.iloc[i:i+SLICE, 6].values])
	online_125 = np.array(online_125)
	online_ts = np.array(online_ts)
	print("<<ONLINE ARRAYS>>")
	print(online_125.shape)
	print(online_ts.shape)
	#
	#
	#
	##::Plot Offline x Online Data
	for i in range(online_125.shape[0]):
		fig, axs = plt.subplots(6)
		fig.set_size_inches(18.5,9.5)
		ch = 0
		for ax in axs:
			ax.plot(online_ts[i,:], online_125[i,ch,:].T, 'r-', linewidth=3)
			ax.plot(offline_ts[i,:], offline_125[i,ch,:].T, 'y--', linewidth=2)
			# ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))
			if not ch:
				ax.legend(['Online Data','Offline Data'], loc = 'best')
			ch += 1
		#add side labels
		fig.add_subplot(111, frame_on=False)
		plt.tick_params(labelcolor="none", bottom=False, left=False)
		plt.xlabel('Magnitude [uv]')
		plt.ylabel('Timestamps')
		plt.show()


	# return offline_125, offline_ts, p300ll_data, p300pos_data, ssvepll_data
