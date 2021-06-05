import pyttsx3
from gtts import gTTS
from time import strftime

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 150)


def text_to_speech(text):
    print("Playing Audio")
    engine.say(text)
    engine.runAndWait()
    audio = gTTS(text)
    print("Saving Audio!!")
    audio.save("Results/audio"+strftime("%H-%M-%S_%d-%m-%Y")+".mp3")
    
def run_text(input_word):
    text = input_word
    if 'end' == text: 
        return
    else:
        text_to_speech(text)
