# AEDL_project
## Acoustic Event Detection and Localisation
In this project, we perform:

* Real-time implementation of Acoustic Event classification 
* Real-time implementaion of Acoustic Event Localisation.

We implement the project using:
* Raspberry Pi model 3B+
* MATRIX Voice development board.

We use the **Raspbian buster** OS version downloaded from [here](https://downloads.raspberrypi.org/raspios_oldstable_armhf/images/raspios_oldstable_armhf-2022-09-26/2022-09-22-raspios-buster-armhf.img.xz).

We setup the MATRIX Voice device on the Raspberry Pi with instructions from [here](https://www.hackster.io/matrix-labs/direction-of-arrival-for-matrix-voice-creator-using-odas-b7a15b) by downloading the MATRIX software and installing the MATRIX kernel modules. The MATRIX kernel modules install well on the Raspbian buster OS.

Run the 'light' compiled cpp file to ensure the MATRIX voice device is being detected on the Raspberry Pi where the MATRIX voice LEDs light green.
Check for the MATRIX voice index number on the Raspberry Pi by running the python script 'check_Matrix_Voice_device_index.py' in 'Raspberry_Pi_and_Matrix_Voice_realtime_implementation' folder.

We train a model first using two sound classes 'footfall' and 'clapping'. The dataset is obtained from the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset. We then convert the model to a [.tflite](https://www.tensorflow.org/tutorials/audio/simple_audio) model from which our program runs inferences from on the Raspberry Pi. We implement the .tflite model on the RPi by running the 'real_time_acoustic_classification.py' which proceses realtime audio data from one channel in MATRIX voice and predicts the acoustic event captured.

We implement realtime localisation using GCC-PHAT algorithm and Voice Activity Detector based on [WebRTC VAD](https://github.com/wiseman/py-webrtcvad). We do this by using two microphones from the MATRIX Voice, MIC 2 and MIC 6. To perform realtime localisation, make sure MATRIX device is first detected by the Raspberry Pi then run 'realtime_DOA.py' in realtime_DOA_implementation folder.
