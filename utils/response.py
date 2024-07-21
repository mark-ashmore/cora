"""Main functions for assistant responses."""

import webbrowser

from playsound import playsound
from google_speech import Speech

import speech_recognition as sr


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class Fulfillment:
    """Fulfilment class, used for response logic based on model parse."""

    def __init__(self, intent):
        self.intent = intent
        # print(self.intent)
        self.get()

    def get(self):
        def func_not_found():  # just in case we don't have the function
            print("No Function " + self.intent + " Found!")

        func = getattr(self, self.intent, func_not_found)
        func()

    @staticmethod
    def search():
        r1 = sr.Recognizer()
        r1.energy_threshold = 50
        url = "https://www.google.com/search?q="
        with sr.Microphone(device_index=0) as source:
            print("\nWhat would you like to search for?")
            say("What would you like to search for")
            audio = r1.listen(source, timeout=6, phrase_time_limit=6)

            try:
                get = r1.recognize_google(audio)
                print("\nSearching for", get)
                webbrowser.open(url + get)
                say("Searching for" + get)
            except sr.UnknownValueError:
                print("error")
            except sr.RequestError as e:
                print("failed".format(e))

    @staticmethod
    def set_alarm():
        r1 = sr.Recognizer()
        r1.energy_threshold = 50
        with sr.Microphone(device_index=0) as source:
            print("\nWhat time would you like to set your alarm for?")
            say("What time would you like to set your alarm for")
            audio = r1.listen(source, timeout=6, phrase_time_limit=6)

            try:
                get = r1.recognize_google(audio)
                print("\nSetting an alarm for", get)
                time = get
                am_pm = None
                if "p.m." in time.lower():
                    am_pm = "pm"
                    time = time[:-5]
                elif "a.m." in time.lower():
                    am_pm = "am"
                    time = time[:-5]
                if time[1] == ":":
                    if am_pm == "am" or None:
                        if is_number(time[0]):
                            print("added 0")
                            time = "0" + time
                    else:
                        if is_number(time[0]):
                            num = int(time[0])
                            mil_num = num + 12
                            time = str(mil_num) + time[1:]
                url = f"https://vclock.com/#time={time}&title=Alarm&sound=bells&loop=1"
                webbrowser.open(url)
                say("Setting an alarm for " + get)
            except sr.UnknownValueError:
                print("error")
            except sr.RequestError as e:
                print("failed".format(e))

    @staticmethod
    def youtube():
        r1 = sr.Recognizer()
        r1.energy_threshold = 50
        url = "https://www.youtube.com/results?search_query="
        with sr.Microphone(device_index=0) as source:
            print("\nWhat would you like to search for on YouTube?")
            say("What would you like to search for on YouTube")
            audio = r1.listen(source, timeout=6, phrase_time_limit=6)

            try:
                get = r1.recognize_google(audio)
                print("\nSearching YouTube for", get)
                webbrowser.open(url + get)
                say("Searching for " + get + " on YouTube")
            except sr.UnknownValueError:
                print("error")
            except sr.RequestError as e:
                print("failed".format(e))
