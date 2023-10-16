import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import tkinter as tk
from tkinter import *


pred = ""

# funtion to predict emotion


def predict():
    model = load_model("model.h5")
    label = np.load("labels.npy")
    holistic = mp.solutions.holistic
    hands = mp.solutions.hands
    holis = holistic.Holistic()
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    lst = []

    _, frm = cap.read()

    frm = cv2.flip(frm, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1, -1)

        global pred
        pred = label[np.argmax(model.predict(lst))]

        print(pred)
        cv2.putText(frm, pred, (50, 50),
                    cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

    drawing.draw_landmarks(frm, res.face_landmarks,
                           holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks,
                           hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks,
                           hands.HAND_CONNECTIONS)
    emotion_label.config(text=f"Your Recognised Emotion: {pred}")
    cv2.imshow("window", frm)
    cv2.destroyAllWindows()
    cap.release()


# funtion to search song on youtube
def search():
    import webbrowser

    predictedEmotion = pred
    lang = textField1.get()
    singer = textField2.get()
    if lang and singer and predictedEmotion:

        # Define the search query
        search_query = f"https://www.youtube.com/results?search_query={predictedEmotion}+{lang}+songs+by+{singer}"

# Perform the search using the default web browser
        webbrowser.open(search_query)


# main function
if __name__ == "__main__":
    root = Tk()
    root.title("DarpanSur")
    root.geometry("900x470+300+200")
    root.resizable(False, False)
    logo_img = PhotoImage(file="emoji1.png")
    root.iconphoto(False, logo_img)
    language_label = Label(root, text="Your preferred language:", font=(
        "times new roman", 12, "bold"), bg="skyblue")
    language_label.place(x=8, y=170, width=250, height=30)
    textField1 = tk.Entry(root, justify='center', width=14, font=(
        "poppins", 12, 'bold'), bg="#203243", border=0, fg="white")
    textField1.place(x=260, y=170, width=100, height=30)
    textField1.focus()
    singer_label = Label(root, text="Enter your preferred singer: ", font=(
        "times new roman", 12, "bold"), bg="skyblue")
    singer_label.place(x=8, y=240, width=250, height=30)
    textField2 = tk.Entry(root, justify='center', width=14, font=(
        "poppins", 12, 'bold'), bg="#203243", border=0, fg="white")
    textField2.place(x=260, y=240, width=100, height=30)
    textField2.focus()
    recognise_button = tk.Button(
        root, text="Recognise emotion", command=predict, background="skyblue")
    recognise_button.place(x=530, y=120, width=120, height=100)
    submit_button = tk.Button(
        root, text="Search Music", command=search, background="green")
    submit_button.place(x=530, y=270, width=120, height=100)
    emotion_label = Label(root, font=(
        "times new roman", 12, "bold"), bg="skyblue")
    emotion_label.place(x=8, y=300, width=354, height=30)
    root.mainloop()
