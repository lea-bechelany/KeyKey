import cv2
import numpy as np
import mediapipe as mp
import requests
import random
import time
import os
from playsound import playsound
import threading

# --- Sound helper ---
def play_sound(path):
    threading.Thread(target=playsound, args=(path,), daemon=True).start()

# --- Word API ---
def get_random_word():
    try:
        response = requests.get("https://random-word-api.herokuapp.com/word", params={"number": 1})
        if response.status_code == 200:
            return response.json()[0]
    except:
        pass
    return "hello"  # fallback

# --- Score persistence ---
score_file = "score.txt"
def load_score():
    if os.path.exists(score_file):
        with open(score_file, "r") as f:
            try:
                return int(f.read())
            except:
                return 0
    return 0

def save_score(score):
    with open(score_file, "w") as f:
        f.write(str(score))

# --- MediaPipe Hands Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# --- Screen Setup ---
canvas_width, canvas_height = 960, 540
cap = cv2.VideoCapture(0)
cap.set(3, canvas_width)
cap.set(4, canvas_height)

# --- Keyboard Setup ---
keys = list("QWERTYUIOPASDFGHJKLZXCVBNM") + ["<"]  # Add backspace key
key_positions = []
start_x, start_y = 50, 300
key_w, key_h = 60, 60
key_h = 60

# Arrange QWERTY layout
rows = [
    keys[0:10],
    keys[10:19],
    keys[19:26] + ["<"]
]
for row_idx, row in enumerate(rows):
    for col_idx, key in enumerate(row):
        x = start_x + col_idx * (key_w + 10) + (30 if row_idx == 1 else 0)
        y = start_y + row_idx * (key_h + 10)
        key_positions.append({"char": key, "pos": (x, y)})

# --- State ---
current_word = get_random_word().upper()
typed_word = ""
score = load_score()
key_pressed = False

# --- Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    index_tip = None
    thumb_tip = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            index = hand_landmarks.landmark[8]
            thumb = hand_landmarks.landmark[4]
            index_tip = int(index.x * w), int(index.y * h)
            thumb_tip = int(thumb.x * w), int(thumb.y * h)

            cv2.circle(frame, index_tip, 10, (0, 255, 255), -1)
            cv2.circle(frame, thumb_tip, 8, (255, 0, 255), -1)

    hovering_key = None

    # --- Draw Keyboard ---
    for key in key_positions:
        x, y = key["pos"]
        char = key["char"]
        cv2.rectangle(frame, (x, y), (x + key_w, y + key_h), (255, 255, 255), 2)
        display_char = "<" if char == "<" else char
        cv2.putText(frame, display_char, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if index_tip and x < index_tip[0] < x + key_w and y < index_tip[1] < y + key_h:
            cv2.rectangle(frame, (x, y), (x + key_w, y + key_h), (0, 255, 0), 3)
            hovering_key = key

    # --- Detect Tap: index and thumb close together ---
    if hovering_key and index_tip and thumb_tip:
        dist = np.linalg.norm(np.array(index_tip) - np.array(thumb_tip))
        if dist < 40 and not key_pressed:
            key_char = hovering_key["char"]
            if key_char == "<":
                if typed_word:
                    typed_word = typed_word[:-1]
                    play_sound("sounds/select.mp3")
            else:
                typed_word += key_char
                play_sound("sounds/select.mp3")

                # --- Check for correct word ---
                if typed_word == current_word:
                    score += 1
                    save_score(score)
                    typed_word = ""
                    current_word = get_random_word().upper()
                    play_sound("sounds/erase.mp3")  # success sound
                elif not current_word.startswith(typed_word):
                    play_sound("sounds/wrong.mp3")
            key_pressed = True
    elif hovering_key is None:
        key_pressed = False

    # --- Display word and score ---
    cv2.putText(frame, f"Type this: {current_word}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
    cv2.putText(frame, f"Typed: {typed_word}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 0), 2)
    cv2.putText(frame, f"Score: {score}", (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("KeyKey", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
