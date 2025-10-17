import os
import time
from tkinter import messagebox, filedialog, simpledialog, Label, Button, Text, Scrollbar, Tk, END, Frame
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import speech_recognition as sr
from gtts import gTTS
import assemblyai as aai
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Environment setup for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress most TensorFlow log messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for better compatibility

aai.settings.api_key = f"8179381008804b77ad3712a93edc0892"  


sia = SentimentIntensityAnalyzer()
recognizer = sr.Recognizer()


detection_model_path = 'models/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprise", "neutral"]


main = Tk()
#main.title("Detecting Depression from Video and Audio")
main.geometry("1400x700")
main.config(bg='light blue')

title = Label(main, text='Detecting Depression From Video and Audio')
title.config(bg='dark blue', fg='white')
title.config(font=('times', 22, 'bold'))
title.config(height=3, width=80)
title.place(relwidth=1)

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="images")
    pathlabel.config(text=filename)

def detectExpression():
    global filename
    text.delete("1.0", END)
    c_img = cv2.imread(filename)
    gray_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_detection.detectMultiScale(gray_img, 1.32, 5)
    text.insert(END, f"Total number of faces detected: {len(faces_detected)}\n\n")
    if len(faces_detected) > 0:
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(c_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img = roi_gray.reshape((1, 48, 48, 1)) / 255.0

            max_index = np.argmax(emotion_classifier.predict(img), axis=-1)[0]
            predicted_emotion = EMOTIONS[max_index]
            cv2.putText(c_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            resized_img = cv2.resize(c_img, (1000, 700))
            emoji_img = cv2.imread(f'Emoji/{predicted_emotion}.png')
            emoji_img = cv2.resize(emoji_img, (600, 400))
            cv2.putText(emoji_img, f"Facial Expression Detected As: {predicted_emotion}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            text.insert(END, f"Facial Expression Detected as: {predicted_emotion}\n")
            cv2.imshow('Facial Emotion Analysis', resized_img)
            cv2.imshow('Emoji Display', emoji_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        messagebox.showinfo("Facial Expression Prediction Screen", "No face detected in uploaded image")

def detect_depression_from_audio():
    text.delete('1.0', END)
    audio_file = filedialog.askopenfilename(initialdir="audio_files", filetypes=[("Audio Files", "*.wav")])
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        recognized_text = recognizer.recognize_google(audio_data)
        text.insert(END, f"\nRecognized text: {recognized_text}\n\n")
        tts = gTTS(recognized_text)
        tts.save("output.mp3")
        os.system("start output.mp3")
    except sr.UnknownValueError:
        print("Could not understand the audio")
    except sr.RequestError as e:
        print(f"Error with the API request: {e}")
    else:
        text_scoring(recognized_text)

def text_scoring(recognized_text):
    sentiment_score = sia.polarity_scores(recognized_text)['compound']
    if sentiment_score <= -0.5:
        text.insert(END, "Detected as: High risk of depression\n")
    elif -0.5 < sentiment_score < 0.5:
        text.insert(END, "Detected as: Mild risk of depression\n")
    else:
        text.insert(END, "Detected as: No depression detected\n")

def detect_depression_from_audio1():
    text.delete('1.0', END)
    text.insert(END, "Don't press any key; it is under processing...\n\n\n")
    audio_file = filedialog.askopenfilename(initialdir="", filetypes=[("Audio Files", "*.wav")])
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file, config=config)
    for utterance in transcript.utterances:
        text.insert(END, f"Speaker {utterance.speaker}: {utterance.text}\n")
        if utterance.speaker == "C":
            text_scoring(utterance.text)

def detectfromvideo(img):
    """
    Detects emotion from a video frame image.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_detection.detectMultiScale(gray_img,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces_detected)))
    if len(faces_detected) > 0:
        for (x, y, w, h) in faces_detected:
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_array = roi_gray.reshape((1, 48, 48, 1)) / 255.0

            max_index = np.argmax(emotion_classifier.predict(img_array), axis=-1)[0]
            predicted_emotion = EMOTIONS[max_index]

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return predicted_emotion
    return "none"

# Update the detectWebcamExpression function if necessary to reflect usage of detectfromvideo.
def detectWebcamExpression():
    text.delete('1.0', END)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img = cap.read()
        
        # Check if frame was successfully captured, break if not (indicating the camera is closed)
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        
        height, width, channels = img.shape
        result = detectfromvideo(img)

        if result != 'none':
            img1 = cv2.imread('Emoji/' + result + ".png")
            img1 = cv2.resize(img1, (width, height))
            cv2.putText(img1, "Facial Expression Detected As : " + result, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Emoji Output", img1)

        cv2.putText(img, "Facial Expression Detected As : " + result, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Facial Expression Output", img)

        # Check if 'q' key is pressed or if the window is closed
        if cv2.waitKey(650) & 0xFF == ord('q') or cv2.getWindowProperty("Facial Expression Output", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_depression_from_text():
    text.delete('1.0', END)
    
    input_text = simpledialog.askstring("Input", "Please enter the text to analyze:")
    
    if input_text:
        text.insert(END, "Input Text: " + input_text + "\n\n")
        
        # Perform sentiment scoring on the input text
        sentiment_score = sia.polarity_scores(input_text)['compound']
        
        # Display the depression level based on the sentiment score
        if sentiment_score <= -0.5:
            result = "High risk of depression"
        elif -0.5 < sentiment_score < 0.5:
            result = "Mild risk of depression"
        else:
            result = "No depression detected"
        
        # Insert the result into the text widget
        text.insert(END, "Detected as: " + result + "\n")
    else:
        text.insert(END, "No text entered.\n")


# UI setup
#font = ('times', 22, 'bold')
#title = Label(main, text='Detecting Depression From Video and Audio', bg='light pink', fg='blue', font=font)
#title.place(x=0, y=5)

'''font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Image With Face", command=upload, font=font1, bg="black", fg='white')
upload.place(x=50, y=150)

pathlabel = Label(main, bg='light blue', fg='black', font=font1)
pathlabel.place(x=300, y=150)

emotion = Button(main, text="Detect Facial Expression", command=detectExpression, font=font1, bg="black", fg='white')
emotion.place(x=50, y=230)

webcam_emotion = Button(main, text="Detect Facial Expression from WebCam", command=detectWebcamExpression, font=font1, bg="black", fg='white')
webcam_emotion.place(x=300, y=230)

audio = Button(main, text="Detect Depression Through Single Voice", command=detect_depression_from_audio, font=font1, bg="black", fg='white')
audio.place(x=670, y=230)

audio1 = Button(main, text="Detect Depression Through Multi Voices", command=detect_depression_from_audio1, font=font1, bg="black", fg='white')
audio1.place(x=1035, y=230)

# Add button for text-based depression detection
text_detection_button = Button(main, text="Detect Depression from Text", command=detect_depression_from_text)
text_detection_button.place(x=670, y=300)
text_detection_button.config(font=font1, bg="black", fg='white')


text = Text(main, height=15, width=137, font=font1)
text.place(x=10, y=310)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)'''


# Define the font variables before using them
font = ('times', 22, 'bold')  # Title font size and style
font1 = ('times', 14, 'bold')  # Font for buttons and text

'''# Title label with the defined font
title = Label(main, text='Detecting Depression From Video and Audio')
title.config(fg='white')  # Modify title background and foreground colors
title.config(font=font)  # Use the font defined earlier
title.config(height=3, width=80)       '''

# Create a frame to hold the title label so the background can be centered with it
'''title_frame = Frame(main, bg='dark blue')
title_frame.place(relwidth=1)  # This will make the background color span across the full width

# Title label inside the title_frame
title_label = Label(title_frame, text='Detecting Depression From Video and Audio')
title_label.config(bg='dark blue', fg='white')  # Modify title background and foreground colors
title_label.config(font=font)  # Use the font defined earlier
title_label.place(relx=0.5, rely=0.5, anchor="center")  # Center title text within the title_frame
'''

#def center_title(event=None):
#    title_frame.place(x=(main.winfo_width() - title_frame.winfo_width()) // 2, y=5)

# Call the function when the window is resized to update the title position
#main.bind("<Configure>", center_title)

# Set initial x and y coordinates for the buttons
button_x = 50
button_y = 150
button_spacing = 80  

upload_button = Button(main, text="Upload Image With Face", command=upload)
upload_button.place(x=button_x, y=button_y)
upload_button.config(font=font1, bg="black", fg='white')  

button_y += button_spacing
emotion_button = Button(main, text="Detect Facial Expression", command=detectExpression)
emotion_button.place(x=button_x, y=button_y)
emotion_button.config(font=font1, bg="black", fg='white')  

button_y += button_spacing
webcam_button = Button(main, text="Detect Facial Expression from WebCam", command=detectWebcamExpression)
webcam_button.place(x=button_x, y=button_y)
webcam_button.config(font=font1, bg="black", fg='white') 

button_y += button_spacing
audio_button = Button(main, text="Detect Depression Through Single Voice", command=detect_depression_from_audio)
audio_button.place(x=button_x, y=button_y)
audio_button.config(font=font1, bg="black", fg='white') 

button_y += button_spacing
multi_audio_button = Button(main, text="Detect Depression Through Multiple Voices", command=detect_depression_from_audio1)
multi_audio_button.place(x=button_x, y=button_y)
multi_audio_button.config(font=font1, bg="black", fg='white') 

button_y += button_spacing
text_detection_button = Button(main, text="Detect Depression from Text", command=detect_depression_from_text)
text_detection_button.place(x=button_x, y=button_y)
text_detection_button.config(font=font1, bg="black", fg='white') 

pathlabel = Label(main)
pathlabel.config(bg='light blue', fg='black')  
pathlabel.config(font=font1)           
pathlabel.place(x=300, y=150)

text_display_x = 450  
text_display_width = main.winfo_width()  

# Create a text box that starts from X=450 and extends till the full window width
#text = Text(main, height=15, width=100, bg='white')  # Set bg='white' for a white background in the text box
#text.place(x=text_display_x, y=160, width=text_display_width)
#text.config(font=font1)

font1 = ('times', 14, 'bold')
text=Text(main,height=17,width=90, bg = 'white')
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=230)
text.config(font=font1)
# Update the title and buttons after the window is drawn (to handle dynamic width)
#main.config(bg='light blue')
main.mainloop()


