import sys
sys.path.insert(1, './tobii_udp')
# -- Similar to addpath --

import tobii
import os
import wx
import time

# TOBII SETUP
ROOT_PATH = os.getcwd()
SERVER_PATH = ROOT_PATH + "/tobii_server"
chk_drivers = tobii.check_drivers(SERVER_PATH)
if not (chk_drivers):
    print("driver not found!!\n")
    sys.exit()
print("Driver found!!\n")
        
## Initialize Server (Run executable)
tobii_server = tobii.tobiiServer(SERVER_PATH) 
tobii_server.connect()
chk_server = tobii_server.check_server()
if not (chk_server):
    print("UDP Server can't be open!!\n")
    sys.exit()
print("UDP Server is open!!\n")
            
##Initialize Client
tobii_client = tobii.tobiiClient()
tobii_client.connect()
#commands = ['init', 'start', 'pause', 'stop']
tobii_client.command('init')



# User 
app = wx.App()
frame = wx.Frame(None, -1, 'win.py')
frame.SetSize(0,0,200,50)
dlg = wx.TextEntryDialog(frame, 'Patient ID','AT-Lab')
dlg.SetValue("Luis")
if dlg.ShowModal() == wx.ID_OK:
    pass

# Create Directory
save_dir = "./DATA/"
USER_DIR =  save_dir + dlg.GetValue()
if not (os.path.exists(USER_DIR)):
    os.mkdir(USER_DIR)


# Check if "CalibrationTobii exist"
USER_TEST = user_dir + "/calibDATA"
if not (os.path.exists(USER_TEST)):
    #tobii.calibration(subject)    
    pass


