import cv2
import os
import numpy as np
import pyttsx3
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time


sender_email = "squardblack7@gmail.com"
password = "sdjvlcekuvzynacf"
receiver_email = "mathiy379@gmail.com"
smtp_server = "smtp.gmail.com"
smtp_port = 587


cooldown_seconds = 60

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 130)

def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# Load Haar cascade and LBPH recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
def prepare_training_data(data_folder_path):
    faces, labels = [], []
    label_map = {}
    label_count = 0

    for file_name in os.listdir(data_folder_path):
        if file_name.lower().endswith((".jpg", ".png")):
            path = os.path.join(data_folder_path, file_name)
            gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            faces_rect = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3)

            for (x, y, w, h) in faces_rect:
                face = gray_img[y:y+h, x:x+w]
                label = file_name.split("_")[0]

                if label not in label_map:
                    label_map[label] = label_count
                    label_count += 1

                faces.append(face)
                labels.append(label_map[label])

    return faces, labels, {v: k for k, v in label_map.items()}

# Train model
print("Training model...")
faces, labels, label_map_rev = prepare_training_data("faces/mathi")
recognizer.train(faces, np.array(labels))
print("Training complete.")

# Start webcam
video_capture = cv2.VideoCapture(0)
announcement_timestamps = {}  # Store last announcement time per person

print("Starting face detection...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(gray, None, fx=1.5, fy=1.5)
    faces_rect = face_cascade.detectMultiScale(resized_gray, scaleFactor=1.1, minNeighbors=3)

    current_time = time.time()
    found_names = set()

    for (x, y, w, h) in faces_rect:
        # Convert back to original resolution
        x, y, w, h = int(x / 1.5), int(y / 1.5), int(w / 1.5), int(h / 1.5)
        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)
        name = label_map_rev.get(label, "Unknown")
        confidence_text = f"{int(confidence)}"

        if confidence < 75:
            found_names.add(name)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence_text})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            last_announcement = announcement_timestamps.get(name, 0)
            if current_time - last_announcement > cooldown_seconds:
                # Speak
                speak(f"{name} is in front of you")
                # Update timestamp
                announcement_timestamps[name] = current_time
                try:
                    email_message = MIMEMultipart()
                    email_message["From"] = sender_email
                    email_message["To"] = receiver_email
                    email_message["Subject"] = f"Face Detected: {name}"
                    email_message.attach(MIMEText(f"{name} is in front of you", "plain"))

                    with smtplib.SMTP(smtp_server, smtp_port) as server:
                        server.starttls()
                        server.login(sender_email, password)
                        server.sendmail(sender_email, receiver_email, email_message.as_string())
                    print(f"Email sent to {receiver_email}")
                except Exception as e:
                    print(f"Failed to send email: {e}")

    # Clean up timestamps for faces not seen anymore
    for key in list(announcement_timestamps.keys()):
        if key not in found_names:
            # Keep last time, but you could delete if you want shorter memory
            pass

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
