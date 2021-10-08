import os.path
import socket
import time
import signal
import struct
from subprocess import Popen, PIPE

class tobiiServer:
    '''
    self.process - subprocess open
    '''
    def __init__(self, SERVER_PATH):
        self.SERVER_PATH = SERVER_PATH
        self.chk_server = False

    def connect(self):
        # Launch server through Python
        DRIVER_PATH = self.SERVER_PATH + "/EyeXMatlabServer.exe"
        try:
            self.process = Popen(DRIVER_PATH, stdout=PIPE, stdin=PIPE, shell=True)
            self.chk_server = True    
            
        except:
            self.chk_server = False
    
    def check_server(self):
        return self.chk_server 

    def disconnect(self):
        self.process.terminate()
        self.chk_server = False

class tobiiClient:
    '''
    self.sock - python udp object
    '''
    def __init__(self, UDP_IP="127.0.0.1"):
        self.UDP_IP = UDP_IP
        self.UDP_PORT = 50102
        self.pos = (0,0)

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #self.sock.settimeout(1/20)
        self.sock.connect((self.UDP_IP, self.UDP_PORT))
        
    def command(self, command):
        if command=='init':
            byte = self.communicate(64)
            
        elif command=='start':
            byte = self.communicate(65)

        elif command=='receive':
            byte = self.communicate(66)  
            print(byte)

        elif command=='stop':
            byte = self.communicate(67)
        else:
            print("Insert a valid command!")
            return False
        print(len(byte))
        if (len(byte)==192):
            return True
        else:
            return False

    def read_position(self):
        byte = self.communicate(66)
        self.pos = self.unpack_bin(byte)
        return self.pos

    def communicate(self, command):
        data = chr(command)
        b_data = data.encode('utf-8')
        self.sock.send(b_data)
        b_rcv_data = self.sock.recvfrom(1024)
        return b_rcv_data[0]

    def unpack_bin(self, byte):
        L_eye = [0,0]
        R_eye = [0,0]
        size_data = int(len(byte)/8)
        if size_data == 24:
            L_eye[0] = (struct.unpack('>d', byte[88:96])[0])/1920
            L_eye[1] = (struct.unpack('>d', byte[96:104])[0])/1080
            R_eye[0] = (struct.unpack('>d', byte[-8:])[0])/1920
            R_eye[1] = (struct.unpack('>d', byte[-16:-8])[0])/1080
            x_pos = (L_eye[0] + R_eye[0])/2
            y_pos = (L_eye[1] + R_eye[1])/2
            return (x_pos,y_pos)
        else:
            return self.pos
            

        
def check_drivers(folder):
    path_dll = folder + "/Tobii.EyeX.Client.dll"
    if (os.path.exists(path_dll)):
        return True
    else:
        return False


