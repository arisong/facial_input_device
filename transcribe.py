import pyautogui
import speech_recognition as sr # debug:https://github.com/Uberi/speech_recognition/issues/294 (same for pyaudio)

def transcribe():
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        try:
            transcription = r.recognize_google(audio, language='en')
        except:
            transcription = ''
        print(transcription)
    return transcription
