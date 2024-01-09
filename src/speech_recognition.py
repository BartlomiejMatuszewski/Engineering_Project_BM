# -*- coding: utf-8 -*-
import os
import pyaudio

from vosk import Model, KaldiRecognizer

from VoiceCommandNumber import VoiceCommandNumber


# module for speech inference
def recognise_speech_command(model_path: str = os.path.join("../frameworks_dnn_models/polish")):
    # initializing model
    model = Model(model_path=os.path.join(model_path))
    recognizer = KaldiRecognizer(model, 16000)

    # reading voice signal from microphone
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()

    print("Nasłuchuję!")

    #  checking if command was said
    while True:
        # reading speech data real time
        data = stream.read(4096)

        if recognizer.AcceptWaveform(data):
            print("Zaczynam...")

            text: str = recognizer.Result()

            if text[14:-3].lower() == "ogień":
                print("Komenda: " + str(VoiceCommandNumber.FIRE))

            elif text[14:-3].lower() == "woda":
                print("Komenda: " + str(VoiceCommandNumber.WATER))

            elif text[14:-3].lower() == "wiatr":
                print("Komenda: " + str(VoiceCommandNumber.WIND))

            elif text[14:-3].lower() == "ziemia":
                print("Komenda: " + str(VoiceCommandNumber.EARTH))

            elif text[14:-3].lower() == "leczenie":
                print("Komenda: " + str(VoiceCommandNumber.HEAL))

            else:
                print("Komenda: " + str(VoiceCommandNumber.NEUTRAL))


if __name__ == '__main__':
    recognise_speech_command(model_path=os.path.join("../frameworks_dnn_models/polish"))
