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
#commands = ['init', 'start', 'stop']
print("-----------------\n")
print("init command:\n")
if (tobii_client.command('init')):
    print("UDP sucesfully initiated:\n")
else:
    print("UDP Server can't be initiated!!\n")
    #sys.exit()
print("-----------------\n")
print("start command:\n")
if (tobii_client.command('start')):
    print("UDP sucesfully started:\n")
else:
    print("UDP Server can't be started!!\n")
    
    #sys.exit()

'''
#Receive eye position from udp server.
initial_time = time.time()
sim_time = 0
while(sim_time<15):
    sim_time = time.time()-initial_time
    pos = tobii_client.read_position()
    print("sim time: {} | eye pos: {}".format(sim_time, pos))
'''

## Extract DATA
print("-----------------\n")
print("Data1\n")
tobii_client.command('receive')
print("-----------------\n")
print("Data1\n")
tobii_client.command('receive')