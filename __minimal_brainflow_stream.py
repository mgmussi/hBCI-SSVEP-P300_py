import time
import numpy as np
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes#, AggOperation

def main ():
    BoardShim.enable_dev_board_logger ()
    params = BrainFlowInputParams ()
    params.serial_port = 'COM11'
    board_id = BoardIds.CYTON_BOARD.value
    board = BoardShim (board_id, params)
    board.prepare_session() #connect to board
    board.start_stream()    #starts stream #45000, 'file://cyton_daisy_data.csv:w')
    time.sleep (10)

    data = board.get_board_data()
    eegChan = board.get_eeg_channels(board_id)
    sampleRate = board.get_sampling_rate(board_id)
    otherChan = board.get_other_channels(board_id)
    accelChan = board.get_accel_channels(board_id)
    emgChan = board.get_emg_channels(board_id)
    ecgChan = board.get_ecg_channels(board_id)
    timeChan = board.get_timestamp_channel(board_id)

    board.stop_stream()
    board.release_session()
    
    print('|Data Shape = [', data.shape, ']')
    print('|Sample Rate =', sampleRate)
    print('|EEG Channels =', eegChan)
    print('|EMG Channels =', emgChan)
    print('|ECG Channels =', ecgChan)
    print('|Accel Channels =', accelChan)
    print('|Other Channels =', otherChan)
    print('|TimeStamp Channels =', timeChan)

if __name__ == "__main__":
    main ()