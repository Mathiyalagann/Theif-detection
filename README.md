🔐 Real-Time Thief Detection using Python and OpenCV

This project is a "real-time face recognition-based thief detection system" developed in Python using OpenCV. It identifies known thief faces by comparing live webcam feed with a pre-stored "faces" database and sends an "email alert" when a match is found.

---

📌 Features

- ✅ Real-time face detection via laptop webcam
- ✅ Face comparison with known "thief" images
- ✅ Email alert with matched face when a thief is detected
- ✅ Easy-to-update face database (just add images to the `faces/` folder)

---

🛠️ Tech Stack

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- pyttsx3 (for optional voice alerts)
- smtplib (for sending email alerts)

---

📁 Project Structure

Real-time-thief-detection/
1. faces/ # Folder containing thief images

   thief1.jpg
   
   thief2.jpg
3. thief_detector.py # Main Python script
4. requirements.txt # Python dependencies

   README.md # Project documentation


---

📷 How It Works

1. Laptop webcam captures real-time frames.
2. Detected faces are compared with images in the "faces/" folder.
3. If a match is found:
   - Voice alert (optional).
   - Email is sent to the registered address with the intruder's image.

---

🚀 Getting Started

1. Clone the Repository

```bash
git clone https://github.com/your-username/real-time-thief-detection.git

cd real-time-thief-detection

2. Create Virtual Environment

python -m venv venv
source - venv\Scripts\activate

3. Install Requirements 
 
pip install -r requirements.txt
opencv-python
numpy
pyttsx3

4. Add Thief Images

Place all known thief faces into the faces/ folder. Make sure filenames are clear (e.g., thief1.jpg, thief2.jpg).

5. Configure Email Alert

Open thief_detector.py and update the email credentials:

EMAIL_SENDER = "your-email@gmail.com"
EMAIL_PASSWORD = "your-app-password"
EMAIL_RECEIVER = "destination-email@gmail.com"

Note: Use an App Password if using Gmail.

6. Run the Program

python thief_detector.py

📬 Sample Email Alert

Subject: ⚠️ Thief Detected
Body: "A thief has been detected in front of the camera."
Attachment: Image of detected face

✉️ Contact

For any queries or support, contact:

📧 gmail-mathiy379@gmail.com
📌 GitHub: @mathiyalagann

