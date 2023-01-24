"""
    real_time_acoustic_classification.py

    This programs collects audio data from a single channel (channel 1)
    in the matrix voice mic on the Raspberry Pi and runs the TensorFlow
    Lite interpreter on a per-build model. 
    
    Raspbian Buster OS is used for implementation of the project due to easy integration of its kernel with the matrix kernel 
	modules.
	To implement this, first install the matrix kernel modules on the Raspberry Pi as well as the matrix hal programming layer
	Make sure the matrix device is first detected by running the Everloop program, the matrix LEDs should light on 
	successful detection
    
"""

from scipy.io import wavfile
from scipy import signal
import numpy as np
import argparse 
import pyaudio
import wave
import time


from tflite_runtime.interpreter import Interpreter

VERBOSE_DEBUG = False

# Check the Matrix Voice device number by running the check_Matrix_Voice_device_index program
# Set the corresponding device index number in the program, in our case device index number is 3

def get_live_input():
    CHUNK = 4096
    FORMAT = pyaudio.paInt32
    CHANNELS = 8
    RATE = 44100 
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "test.wav"
    NFRAMES = int((RATE * RECORD_SECONDS) / CHUNK)

    # initialize pyaudio
    p = pyaudio.PyAudio()

    print('opening stream...')
    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = CHUNK,
                    input_device_index = 3)

    # discard first 1 second
    for i in range(0, NFRAMES):
        data = stream.read(CHUNK, exception_on_overflow = False)

    try:
        while True:
            print("Listening...")
            frames = []
            for i in range(0, NFRAMES):
                data = stream.read(CHUNK, exception_on_overflow = False)
                frames.append(data)

            # process data
            # 4096 * 3 frames * 2 channels * 4 bytes = 98304 bytes 
            # CHUNK * NFRAMES * 2 * 4 
            buffer = b''.join(frames)
            audio_data = np.frombuffer(buffer, dtype=np.int32)
            
            #Pick channel one from the eight channels of the eight microphones in matrix voice
            audio_data = audio_data[0::8]
            nbytes = CHUNK * NFRAMES 
            # reshape for input 
            audio_data = audio_data.reshape((nbytes, 1))
            # run inference on audio data 
            run_inference(audio_data)
    except KeyboardInterrupt:
        print("exiting...")
           
    stream.stop_stream()
    stream.close()
    p.terminate()

def process_audio_data(waveform):
    """Process audio input.

    This function takes in raw audio data from a WAV file and does scaling 
    and padding to 44100 length.

    """

    if VERBOSE_DEBUG:
        print("waveform:", waveform.shape, waveform.dtype, type(waveform))
        print(waveform[:5])


    if VERBOSE_DEBUG:
        print("After scaling:")
        print("waveform:", waveform.shape, waveform.dtype, type(waveform))
        print(waveform[:5])

    # normalise audio
    wabs = np.abs(waveform)
    wmax = np.max(wabs)
    waveform = waveform / wmax

    PTP = np.ptp(waveform)


    # return None if too silent 
    # the value here can be adjusted to the suitable threshold according to the noise in the environment
    if PTP < 1:
        return []

    if VERBOSE_DEBUG:
        print("After normalisation:")
        print("waveform:", waveform.shape, waveform.dtype, type(waveform))
        print(waveform[:5])

    # scale and center
    waveform = 2.0*(waveform - np.min(waveform))/PTP - 1

    # extract 44100 len (1 second) of data   
    max_index = np.argmax(waveform)  
    start_index = max(0, max_index-22050)
    end_index = min(max_index+22050, waveform.shape[0])
    waveform = waveform[start_index:end_index]

    # Padding for files with less than 44100 samples
    if VERBOSE_DEBUG:
        print("After padding:")

    waveform_padded = np.zeros((44100,))
    waveform_padded[:waveform.shape[0]] = waveform

    if VERBOSE_DEBUG:
        print("waveform_padded:", waveform_padded.shape, waveform_padded.dtype, type(waveform_padded))
        print(waveform_padded[:5])

    return waveform_padded

def get_spectrogram(waveform):
    
    waveform_padded = process_audio_data(waveform)

    if not len(waveform_padded):
        return []

    # compute spectrogram 
    f, t, Zxx = signal.stft(waveform_padded, fs=44100, nperseg=512, 
        noverlap = 256, nfft=512)
    # Output is complex, so take abs value
    spectrogram = np.abs(Zxx)

    if VERBOSE_DEBUG:
        print("spectrogram:", spectrogram.shape, type(spectrogram))
        print(spectrogram[0, 0])
        
    return spectrogram


def run_inference(waveform):

    # get spectrogram data 
    spectrogram = get_spectrogram(waveform)

    if not len(spectrogram):
        print("Silence...")
        time.sleep(1)
        return 

    spectrogram1= np.reshape(spectrogram, (-1, spectrogram.shape[0], spectrogram.shape[1], 1))
    
    if VERBOSE_DEBUG:
        print("spectrogram1: %s, %s, %s" % (type(spectrogram1), spectrogram1.dtype, spectrogram1.shape))

    # load TF Lite model
    interpreter = Interpreter('acoustic_classification.tflite')
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    input_shape = input_details[0]['shape']
    input_data = spectrogram1.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    print('Acoustic Event Detected')
    print("Predicting...")
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    yvals = output_data[0]
    commands = ['clapping', 'footfall']

    if VERBOSE_DEBUG:
        print(output_data[0])
    print(commands[np.argmax(output_data[0])].upper() + ' detected!!!')

    time.sleep(1)

def main():

    # create parser
    descStr = """
    This program does ML inference on audio data.
    """
    parser = argparse.ArgumentParser(description=descStr)
    # add a mutually exclusive group of arguments
    group = parser.add_mutually_exclusive_group()

    # add expected arguments
    group .add_argument('--input', dest='wavfile_name', required=False)
    
    # parse args
    args = parser.parse_args()


    
    # test WAV file
    if args.wavfile_name:
        wavfile_name = args.wavfile_name
        # get audio data 
        rate, waveform = wavfile.read(wavfile_name)
        # run inference
        run_inference(waveform)
    else:
        get_live_input()

    print("done.")

# main method
if __name__ == '__main__':
    main()
