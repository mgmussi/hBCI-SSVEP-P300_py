# SSVEP P300 -py
Hybrid Brain-Computer Interface using synchronous SSVEP and P300 for Python. On a display, there will be three squares which will have white centre areas that flash at different frequencies for the SSVEP component, and an outline edge that will appear around the squares one at a time in a pseudo-random order for the P300 component. For the SSVEP, squares will create the flashing effect by interpolating between black and white for maximum contrast. Because of the limited frames per second provided by the monitors available for this experiment (60 frames per second), and to avoid seizure-inducing ranges (12-25 Hz ([Fisher et al., 2005](https://onlinelibrary-wiley-com.login.ezproxy.library.ualberta.ca/doi/full/10.1111/j.1528-1167.2005.31405.x); [Okudan & Özkara, 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5779309/))), the chosen frequencies were be 10, 6 and 4 Hz (_but they can be changed using different frame ratios_). These frequencies can be easily attained on the display because they are multiples of the monitor’s frames per second. When the classification is concluded, the selected square’s centre area will briefly turn green in colour to indicate the classifier chose that square as selected.

::The system is compatible with the OpenBCI and the g.tec Unicorn::

## Setting up experiment on a new computer (Windows):
### Preparation, download:
1. Python 3 (3.6.6)
2. Anaconda
3. Sublime (text editor)
4. Psychopy [.yml file](https://raw.githubusercontent.com/psychopy/psychopy/master/conda/psychopy-env.yml) (for Conda and Mini-Conda)
5. Install FTDI driver (VCP) - executable

### A. Creating the environment:
_Obs.: If you have a Linux, step 2) needs to be changed acordingly. The environment needs to be created based on the [Psychopy](https://www.psychopy.org/download.html) plug-in. Installing Psychopy on existing environents does not work consistently._

1) Download and install Anaconda Powershell Prompt (anaconda3)
2) Once installed, go to the folder with the downloaded file psychopy-env.yml create a new environment using the code below (instructions [here](https://github.com/psychopy/psychopy/blob/release/docs/source/download.rst#id8)): 
```
conda env create -n psychopy -f psychopy-env.yml 
```
3) Activate the newly created environment: (to deactivate, use `conda deactivate`)
```
conda activate psychopy
```
4) Once in the environment,  install the following:
```
conda install -c anaconda cython
conda install -c conda-forge matplotlib
conda install -c conda-forge pyglet=1.4.10
conda install -c anaconda pyserial
conda install numpy
conda install pyqtgraph
conda install scipy
conda install pandas
pip install pylsl
pip install requests
pip install Pillow
```
All the above should work seamlessly when installing. If any error aoccurs, try searching Google to see if I did any mistakes with the syntax.

#### Troubleshooting:
* _Pyglet needs to be downgraded to 1.4.10 because some later versions yield the error “WMF Failed to initialize threading: err.strerror”._
* _When running for the first time, two errors can appear saying “The program can’t start because VCRUNTIME140D.dll is missing from your computer” and “The program can’t start because ucrtbased.dll is missing from your computer”. To solve this problem, both dll’s need to be downloaded and pasted in C:/Windows/System32._
* _The error "pyglet.gl.lib.MissingFunctionException: glActiveTexture is not exported by the available OpenGL driver” can be solved by updating the driver of the current graphics card._
* _The error “ImportError: sys.meta_path is None, Python is likely shutting down” has no solution yet found_

### B. Install Brainflow
On the environment/prompt run the following code:
```
python -m pip install brainflow
```

### C. Install Scikit-Learn
On the environment/prompt run the following code:
```
conda install scikit-learn
```

### D. Remove Latency from OpenBCI
(Open File Explorer, Right Click on "My Computer” (or "This PC”, name varies) > Properties > Device Manager. Then click on the dropdown “Ports (COM & LPT)”, right click on the dongle port (e.g. COM3) > Properties, click the tab “Port Settings”,  “Advanced…” and then change “Latency Timer (msec)” to 1.

## Working with Brainflow
To stream with Brainflow, it is possible to connect and start the streaming from within the code. The minimum sequence to connect to a board is:
import brainflow
from brainflow.board_shim import BardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperation #sometimes this last object doesn’t work. Commenting it solves the problem

```
##CONNECTION
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
params.serialport = ‘COM11’ #each board needs some specific params. Check https://brainflow.readthedocs.io/en/stable/SupportedBoards.html
                            #also, the port varies from computer to computer. In mac, the format is /dev/cu.usbserial-XXXXXXX
board_id = BoardIds.CYTON_BOARD.value    #this value can be found in the link above. Each board has a number, e.g. Cyton Board is ‘0’ 
board = BoardShim(board_id, params)    #creates callable object
board.prepare_session()    #finds and connects to board
board.start_stream ()    #starts actual streaming
#board.start_stream(45000, ‘file://cyton_data.csv:w')   #this is an alternative form which specifies the ringbuffer and (automatically) saves raw data from
                                                        #board in .csv file

##GETTING SAMPLES
data = board.get_board_data() #gets ALL data from board (sample ID, samples, sensors, timestamps, other channels, etc) and cleans ringbuffer
#current_data = board.get_current_data() #gets ALL data from board and keeps it in ringbuffer
#it’s important to note that all the information is in the same array. To find out what columns contain what kind of information, use:
board.get_eeg_channels(board_id)
board.get_emg_channels(board_id)
board.get_ecg_channels(board_id)
```

#### StackOverflow questions
Some of the implementations came from ideas that people gave as I tried to solve my problem(s).
1) The callback solution, which helped me to actually plot something, but not quite in real-time: https://stackoverflow.com/questions/62273244/reading-higher-frequency-data-in-thread-and-plotting-graph-in-real-time-with-tki
2) Some ideas for profiling and making the plotting faster. I got the  set_data  idea from here: https://stackoverflow.com/questions/63529920/plotting-higher-frequency-data-using-threads-in-real-time-without-freezing-tkint
3) Trying to make set_data work in my case (with no success, unfortunately): https://stackoverflow.com/questions/63637759/why-are-my-plots-not-appearing-with-set-data-using-tkinter

### D. Streaming the LSL (simulation)
0) In your Anaconda prompt, make sure you activated the created environment
1) Navigate to the OpenBCI LSL folder

---
### Problems with Mac not recognizing OpenBCI Dongle:
Sometimes works, sometimes don’t: https://gist.github.com/technobly/97cb576957f0a701580984c6edc0433f

### Problems With FTDI Buffer FIx on OS X:
1) Did not solve the problem, but here is the resource: https://docs.openbci.com/docs/10Troubleshooting/FTDI_Fix_Mac
2) A thread that ended being the resource: https://openbci.com/forum/index.php?p=/discussion/199/latencytimer-inbuffersize-for-os-x-new-info-plist




------
## Old Program Achitecture - DEPRECATED
### Download the OpenBCI LSL reader

1) Go to https://github.com/openbci-archive/OpenBCI_LSL and clone the directory to a folder of your choice

To run this directory, the system will also need the following packages:
```
pip install pyserial
pip install pylsl
```
### A.2. Creating a MatLab simulator for the EEG board (Optional)
If you do have a board that can be connected, disconsider this section and connect the board and jump to section D. You'll need MatLab.
0) If you don't have MatLab, download the free license available here for UofA students.
1) Download the Lab Streaming Layer (LSL) Matlab Library https://github.com/labstreaminglayer/liblsl-Matlab
2) Open the file SendData.m
3) Run the file
LSL, in case you don't know, it's a sort of data-transferring protocol.
If error appears to simulate signal with SendData.m:
- Make sure `liblsl-Matlab` repository is installed properly
- Download `MinGW-w 64` from within MatLab Add-ons
- Move some files inside the liblsl release package:
> move `liblsl64.lib` & `liblsl64.dll` to the bin folder
> run `build-mex` from MatLab
- If error reading `library.dll` appears, use version 1.13.0-b13 instead of b14 (found on releases)

## B.2. Streaming with OpenBCI LSL (simulation)
0) In your Anaconda prompt, make sure you activated the created environment
2) Navigate to the OpenBCI LSL folder
3) Run the following `python openbci_lsl.py --stream`
This will only work if you have the simulator or the (OpenBCI) board plugged in, so make sure to do that first
3) Once the 'board' is identified, run `/start` to initiate streaming.
4) To stop streaming, simply run `/stop`
5) To quit (stop before quitting) `/exit`
