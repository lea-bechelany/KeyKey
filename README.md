# ⌨️ KeyKey

KeyKey is an interactive gesture-based typing game that uses your webcam to let you type letters by "tapping" on a virtual keyboard with your fingers. 
It's a fun way to practice spelling and test your speed, without ever touching your keyboard!

---

## 🧠 How It Works

- 🖐️ Your webcam detects your index finger and thumb using MediaPipe
- 🧲 Hover your index over a key, then tap your thumb and index together to select it
- 🎯 Try to type the target word shown on screen — get it right to score a point!
- 💾 Your score is saved between sessions!

---

## 🚀 Features

- Gesture-based typing using hand landmarks
- Full A-Z virtual QWERTY keyboard
- Backspace support (`<` key)
- Real-time word matching and scoring
- Audio feedback for correct, incorrect, and key taps
- Random word fetch via public API
- Score persistence using `score.txt`

---

## 📦 Dependencies

- Python 3.7+
- `opencv-python`
- `mediapipe`
- `numpy`
- `requests`
- `playsound`

---

## 🛠️ Installation

## Clone the repo

git clone https://github.com/yourusername/KeyKey.git
cd KeyKey


## Install the requirements

pip install opencv-python mediapipe numpy requests playsound


## Run the game

python keykey.py

