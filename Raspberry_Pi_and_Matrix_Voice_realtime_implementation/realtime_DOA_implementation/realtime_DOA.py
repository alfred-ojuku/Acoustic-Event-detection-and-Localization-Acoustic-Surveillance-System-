"""
 Estimate the realtime Direction of Arrrival using two microphones from the matrix voice (Microphone 2 and Microphone 6)
 
 Raspbian Buster OS is used for implementation of the project due to easy integration of its kernel with the matrix kernel 
 modules.
 To implement this, first install the matrix kernel modules on the Raspberry Pi as well as the matrix hal programming layer
 Make sure the matrix device is first detected by running the Everloop program, the matrix LEDs should light on 
 successful detection
 

"""

import pyaudio
import webrtcvad
import numpy as np
import collections
import multiprocessing
import threading
import signal
import sys
import math
import audioop
from gcc_phat import gcc_phat
from vad import vad

# Check the Matrix Voice device number by running the check_Matrix_Voice_device_index program
# Set the corresponding device index number in the program, in our case device index number is 3
class Microphone:

    def __init__(self, rate=48000, channels=8):
        self.pyaudio_instance = pyaudio.PyAudio()
        self.queue = multiprocessing.Queue()
        self.quit_event = threading.Event()
        self.channels = channels
        self.sample_rate = rate

    def _callback(self, in_data, frame_count, time_info, status):
        self.queue.put(in_data)
        return None, pyaudio.paContinue

    def read_chunks(self, size):
        device_index = 3
        
        stream = self.pyaudio_instance.open(
            input=True,
            format=pyaudio.paInt32,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=size,
            stream_callback=self._callback,
            input_device_index = device_index,
        )

        while not self.quit_event.is_set():
            frames = self.queue.get()
            if not frames:
                break
            yield frames
        
        stream.close()

    def close(self):
        self.quit_event.set()
        self.queue.put('')


def main():
    sample_rate = 44100
    channels = 8
    N = 4096 * 4

    mic = Microphone(sample_rate, channels)
    window = np.hanning(N)

    sound_speed = 343.2
    
    # Exact distance between microphone 2 and microphone 6 in matrix voice
    distance = 0.07469354055

    max_tau = distance / sound_speed

    def signal_handler(sig, num):
        print('Quit')
        mic.close()

    signal.signal(signal.SIGINT, signal_handler)

    for data in mic.read_chunks(N):
    
        buf = np.frombuffer(data, dtype='int32')
        mono = buf[1::channels].tobytes()
        if sample_rate != 16000:
            mono, _ = audioop.ratecv(mono, 2, 1, sample_rate, 16000, None)
        print('silence')
        if vad.is_speech(mono):
#from here  
            
            data = mic.read_chunks(N)
#to here         
            tau, _ = gcc_phat(buf[1::channels]*window, buf[5::channels]*window, fs=sample_rate, max_tau=max_tau)
            theta = math.asin(tau / max_tau) * 180 / math.pi
            if -11 > theta > -90 :
                print('ACOUSTIC EVENT DETECTED FROM THE NORTH!!! Angle of arrival: {}'.format(int(theta)))
            else:
                print('ACOUSTIC EVENT DETECTED FROM THE SOUTH!!! Angle of Arrival: {}'.format(int(theta)))
  
        
if __name__ == '__main__':
    main()
