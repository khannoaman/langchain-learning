import requests
from bs4 import BeautifulSoup
import pandas as pd
from threading import *
import pyttsx3
import speech_recognition as sr
import datetime
import re
import numpy as np
import cv2
from plyer import notification
import time

#Get Country Data
class Country(Thread):

    countriesData=None
    countriesName=None

    def run(self):
        try:
            url = "https://www.worldometers.info/coronavirus/?utm_campaign=homeAdUOA?Si9"

            r = requests.get(url)
            htmlcontent = r.content

            soup = BeautifulSoup(htmlcontent, "lxml")
            table = soup.find("table")
            tr = table.find_all("tr")
            data = []
            for i in tr:
                def process(i):
                    i = i.text
                    i = i.strip()
                    i = re.sub(r'\+', "", i)
                    i = re.sub(r',', "", i)
                    return i

                row = [j for j in map(process, i.find_all(["th", "td"]))]
                data.append(row)

            df = pd.DataFrame(data)
            df.columns = df.iloc[0, :]
            df = df.iloc[:, [1, 2, 4, 6, 8]]
            df = df.drop(0)
            df = df.replace("", np.nan)
            df = df.replace("N/A", np.nan)
            df = df.dropna(subset=["CountryOther"])
            df.columns = ["Location", "Confirmed", "Deceased", "Recovered", "Active"]
            df.iloc[6, 0] = "world"
            df.set_index("Location", drop=True, inplace=True)
            df = df.drop("Total:")
            df.fillna(0, inplace=True)
            df = df.reindex(columns=["Confirmed", "Active", "Recovered", "Deceased"])
            df.to_csv("CountryData.csv")
            Country.countriesData = df
            Country.countriesName = list(df.index)



        except:
            df=pd.read_csv("CountryData.csv",index_col="Location")
            Country.countriesData = df
            Country.countriesName = list(df.index)




#Country Map
class CountryMap(Thread):
    @classmethod
    def run(cls):
        df = pd.read_csv("CountryMap.csv", index_col=0)
        df1 = pd.read_csv("CountryData.csv", index_col=0)

        img = cv2.imread("worldMap.jpg")
        part = cv2.imread("countryPart.jpg")

        window = "WORLD"
        cv2.namedWindow(window)

        def details(s):
            box = cv2.imread("box.png")
            p, q, r, t = df1.loc[s, :]
            text = "{}".format(s)
            cv2.putText(box, text, (25, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (35, 35, 35), 2)
            text = "Total Cases: {}".format(p)
            cv2.putText(box, text, (25, 65), cv2.FONT_ITALIC, 0.6, (35, 35, 35), 2)
            text = "Active: {}".format(q)
            cv2.putText(box, text, (25, 95), cv2.FONT_ITALIC, 0.6, (35, 35, 35), 2)
            text = "Recovered: {}".format(r)
            cv2.putText(box, text, (25, 125), cv2.FONT_ITALIC, 0.6, (35, 35, 35), 2)
            text = "Death: {}".format(t)
            cv2.putText(box, text, (25, 155), cv2.FONT_ITALIC, 0.6, (35, 35, 35), 2)

            return box

        def location(x, y):
            country = df.loc[(df.xmin <= x) & (df.xmax >= x) & (df.ymin <= y) & (df.ymax >= y), ["Location"]]
            if len(country) == 0:
                return None
            else:
                return country.iloc[0][0]

        def Data(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                if location(x, y) != None:
                    box = details(location(x, y))

                    img[445:615, 1300:1550] = box
                else:
                    img[445:615, 1300:1550] = part

        cv2.setMouseCallback(window, Data)

        while True:
            cv2.imshow(window, img)
            if cv2.waitKey(10) == 27 or cv2.waitKey(10)==13:
                break
        cv2.destroyAllWindows()





#Get States Data
class States(Thread):

    statesData=None
    statesName=None
    def run(self):
        try:
            url = "https://www.mygov.in/covid-19"

            r = requests.get(url)
            htmlcontent = r.content

            soup = BeautifulSoup(htmlcontent, "lxml")

            df = pd.read_html(soup.prettify())[0]
            df.columns = ["Location", "Confirmed", "Active" , "Recovered","Deceased"]
            df = df.set_index("Location", drop=True)
            df.to_csv("StateData.csv")
            States.statesData = df
            States.statesName = list(df.index)


        except:
            df=pd.read_csv("StateData.csv",index_col="Location")
            States.statesData = df
            States.statesName = list(df.index)

class StateMap(Thread):
    @classmethod
    def run(cls):
        df = pd.read_csv("IndiaMap.csv", index_col=0)
        df1 = pd.read_csv("StateData.csv", index_col=0)

        img = cv2.imread("India1.png")
        part = cv2.imread("part.jpg")

        window = "INDIA"
        cv2.namedWindow(window)

        def details(s):
            box = cv2.imread("box.png")
            p, q, r, t = df1.loc[s, :]
            text = "{}".format(s)
            cv2.putText(box, text, (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (35, 35, 35), 2)
            text = "Total Cases: {}".format(p)
            cv2.putText(box, text, (20, 65), cv2.FONT_ITALIC, 0.6, (35, 35, 35), 2)
            text = "Active: {}".format(q)
            cv2.putText(box, text, (20, 95), cv2.FONT_ITALIC, 0.6, (35, 35, 35), 2)
            text = "Recovered: {}".format(r)
            cv2.putText(box, text, (20, 125), cv2.FONT_ITALIC, 0.6, (35, 35, 35), 2)
            text = "Death: {}".format(t)
            cv2.putText(box, text, (20, 155), cv2.FONT_ITALIC, 0.6, (35, 35, 35), 2)

            return box

        def location(x, y):
            state = df.loc[(df.xmin <= x) & (df.xmax >= x) & (df.ymin <= y) & (df.ymax >= y), ["Location"]]
            if len(state) == 0:
                return None
            else:
                return state.iloc[0][0]

        def Data(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                if location(x, y) != None:
                    box = details(location(x, y))

                    img[53:223, 420:670] = box
                else:
                    img[53:223, 420:670] = part

        cv2.setMouseCallback(window, Data)

        while True:
            cv2.imshow(window, img)
            if cv2.waitKey(10) == 27 or cv2.waitKey(10)==13:
                break
        cv2.destroyAllWindows()


#Myth Buster
class MythBuster(Thread):
    @classmethod
    def run(cls):
        img_urls = ["https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-16.jpg",
                    "https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-14-1.jpg",
                    "https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-12.jpg",
                    "https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-14.jpg",
                    "https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-11.jpg",
                    "https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-10.jpg",
                    "https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-9.jpg",
                    "https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-8.jpg",
                    "https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-7.jpg",
                    "https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-6.jpg",
                    "https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-5.jpg",
                    "https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-4.jpg",
                    "https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-3.jpg",
                    "https://transformingindia.mygov.in/wp-content/uploads/2020/05/Myth-1.jpg"]

        for i in img_urls:
            res = requests.get(i)
            img = cv2.imdecode(np.asarray(bytearray(res.content)), -1)
            img = cv2.resize(img, (700, 700))
            img[20:80, 600:680] = img[20:80, 507:587]
            cv2.imshow("Myth Buster", img)
            if cv2.waitKey(5000) == 27:
                break
        cv2.destroyAllWindows()

#Symptoms
class Symptoms(Thread):
    @classmethod
    def run(cls):
        img = cv2.imread("Symptoms.png")
        img = img[:530, :590]
        cv2.imshow("Symptoms", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#Spread & Prevention
class SpreadPrevention(Thread):
    @classmethod
    def run(cls):
        img = cv2.imread("Spread & Prevention.png")
        img = img[:532, 5:640]
        cv2.imshow("Spread & Prevention", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#Do's & Dont's
class DoDont(Thread):
    @classmethod
    def run(cls):
        img_urls = [
            "https://transformingindia.mygov.in/wp-content/uploads/2020/06/WhatsApp-Image-2020-06-09-at-7.53.39-PM.jpeg",
            "https://transformingindia.mygov.in/wp-content/uploads/2020/06/WhatsApp-Image-2020-06-09-at-7.53.41-PM.jpeg",
            "https://transformingindia.mygov.in/wp-content/uploads/2020/06/WhatsApp-Image-2020-06-09-at-7.54.32-PM.jpeg",
            "https://transformingindia.mygov.in/wp-content/uploads/2020/06/WhatsApp-Image-2020-06-09-at-7.54.33-PM.jpeg",
            "https://transformingindia.mygov.in/wp-content/uploads/2020/06/1.jpeg",
            "https://transformingindia.mygov.in/wp-content/uploads/2020/06/2.jpeg",
            "https://transformingindia.mygov.in/wp-content/uploads/2020/05/WhatsApp-Image-2020-05-10-at-10.04.57-PM-291x350.jpeg",
            "https://transformingindia.mygov.in/wp-content/uploads/2020/05/WhatsApp-Image-2020-05-28-at-10.29.31-PM-1.jpeg",
            "https://transformingindia.mygov.in/wp-content/uploads/2020/05/WhatsApp-Image-2020-05-28-at-10.29.32-PM.jpeg",
            "https://transformingindia.mygov.in/wp-content/uploads/2020/04/WhatsApp-Image-2020-03-31-at-12.35.59-PM.jpeg",
            "https://transformingindia.mygov.in/wp-content/uploads/2020/05/WhatsApp-Image-2020-05-01-at-4.06.31-PM.jpeg",
            "https://transformingindia.mygov.in/wp-content/uploads/2020/05/WhatsApp-Image-2020-04-29-at-4.35.25-PM.jpeg",
            "https://transformingindia.mygov.in/wp-content/uploads/2020/03/Tips-On-Work-From-Home.jpg"]
        for i in img_urls:
            res = requests.get(i)
            img = cv2.imdecode(np.asarray(bytearray(res.content)), -1)
            img = cv2.resize(img, (700, 700))
            cv2.imshow("Do's & Don'ts", img)
            if cv2.waitKey(5000) == 27:
                break
        cv2.destroyAllWindows()


#Notification
class notifi(Thread):
    def run(self):

            file = open("CoronaData.txt")
            lines = file.readlines()

            while True:
               for i in range(22):

                    if i < 3 :
                        if i == 0:
                            s = ""
                            for j in lines[11:16]:
                                s += j
                            notification.notify(title=lines[8].strip(), message=s, app_name='Covid', app_icon="icon.ico",
                                            timeout=5)
                            time.sleep(8)
                        elif i == 1:
                            s = ""
                            for j in lines[16:24]:
                                s += j
                            notification.notify(title=lines[8].strip(), message=s, app_name='Covid', app_icon="icon.ico",
                                            timeout=5)
                            time.sleep(8)
                        elif i == 2:
                            s = ""
                            for j in lines[25:29]:
                                s += j
                            notification.notify(title=lines[8].strip(), message=s, app_name='Covid', app_icon="icon.ico",
                                            timeout=5)
                            time.sleep(8)



                    if i < 7 :
                        notification.notify(title=lines[35].strip(), message=lines[38 + i], app_name='Covid',
                                            app_icon="icon.ico", timeout=5)

                        time.sleep(8)



                    notification.notify(title=lines[46].strip(), message=lines[48 + i], app_name='Covid',
                                    app_icon="icon.ico", timeout=5)

                    time.sleep(8)
            print("out ")





def speak(audio):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 250)
    #engine.setProperty('volume', 100)
    engine.say(audio)
    engine.runAndWait()



def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning!")

    elif hour>=12 and hour<18:
        speak("Good Afternoon!")

    else:
        speak("Good Evening!")

    speak("I am covid-19 voice assistant Sir. Please tell me how may I help you")


#It takes microphone input from the user and returns string output
def takeCommand(ob):
    query="None"
    if ob.status:

        r = sr.Recognizer()
        with sr.Microphone() as source:
            ob.last_sent_label("Listening...")
            r.pause_threshold = 1
            r.energy_threshold=500
            audio = r.listen(source)

        try:
            if ob.status:
                ob.last_sent_label("Recognizing...")
                query = r.recognize_google(audio, language='en-in')
                ob.user(query)

        except Exception as e:
            # print(e)
            if ob.status:
                ob.last_sent_label("Say that again please...")
                time.sleep(1)
                return "None"
        return query


def find_place(query,s,c):
   a=[]
   for i in s.statesName:
       if i in query:
           a.append(i)

   for i in c.countriesName:
       if i in query:
           a.append(i)


   return a



def Exit(query):
    for i in ["close","exit","bye"]:
        if i in query:
            return 1


def data():
    file=open("CoronaData.txt")
    return file.readlines()


