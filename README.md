Here’s a `README.md` file for our project that adjusts screen brightness and system volume using hand gestures via webcam, utilizing MediaPipe, OpenCV, and other libraries:


Hand Gesture Controller for Volume and Brightness

This Python-based project uses your webcam and hand gestures to control system volume and screen brightness in real-time. It uses MediaPipe for hand tracking, OpenCV for video processing, PyCAW for volume control, and screen-brightness-control for brightness adjustment.

 Features

* 🖐️ Detects two hands and distinguishes between left and right.
* 🔊 Right hand controls system volume (index finger and thumb distance).
* 💡 Left hand controls screen brightness (index finger and thumb distance).
* 👆 Real-time gesture tracking using your webcam.
* 🔄 Flipped (mirrored) display for natural interaction.


Requirements

Install the following dependencies before running the program:

```bash
pip install opencv-python mediapipe screen-brightness-control pycaw comtypes numpy
```

Also, ensure you are running on **Windows**, since `pycaw` and `screen-brightness-control` are Windows-specific.

📁 File Structure

```
.
├── hand_gesture_controller.py   # Main script
├── README.md                    # This file
```

---

## 🚀 How to Run

1. Clone this repository or save the script to your local machine.

2. Run the script:

```bash
python hand_gesture_controller.py
```

3. Allow access to your webcam.

4. Use hand gestures:

   Right Hand (your left on screen): Control Volume.
   Left Hand (your right on screen): Control Brightness.
   Move index finger and thumb** closer or further apart to adjust levels.

5. Press `Q` to quit.

How It Works

* Uses MediaPipe Hands to detect 21 hand landmarks.
* Tracks landmark 4 (thumb tip) and landmark 8 (index tip).
* Calculates the distance between them.
* Interpolates this distance to system brightness (0–100%) or volume range (min–max).
* Adjusts:
  * Brightness using `screen_brightness_control`
  * Volume using `pycaw`

Troubleshooting

*No camera access?Make sure no other application is using the webcam.
*Brightness not changing? Some monitors (especially external) might not support brightness control.
*Volume control not working?Ensure you're running it on Windows and that `pycaw` is properly installed.

📌 Notes

* This project is designed for **Windows** only due to `pycaw` and `screen-brightness-control` compatibility.
* It mirrors the webcam feed for intuitive gesture interaction.
* Accuracy may vary depending on lighting and webcam quality.

Made with using Python, OpenCV, and MediaPipe.