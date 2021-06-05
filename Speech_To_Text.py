import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source,duration=1)
        audio = r.listen(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(query)
    except sr.UnknownValueError:
        speak("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        speak("Could not request results from Google Speech Recognition service; {0}".format(e))
    except Exception as e:
        print(e,'\n',"say again")
        return "None"
    return query

def run_speech():
    text_full = ""
    while(True):
        text = takeCommand()
        text_full += text+" "
        if 'stop' in text:
            break
        else:
            continue
    return text_full
    