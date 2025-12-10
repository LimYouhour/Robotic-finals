# file: motor_control.py
import os
import time
import cv2
import torch
import numpy as np
from torchvision import transforms
from auppbot import AUPPBot

# === Config ===
MODEL_PATH = "gesture_model.pt"
CLASS_NAMES_FILE = "class_names.txt"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Motor speeds (tweak to your robot)
FWD_SPEED = 25
TURN_SPEED = 20
REV_SPEED = -20

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---- Robot control helpers ----
class RobotController:
    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        self.bot = AUPPBot(port, baud, auto_safe=True)

    def move_forward(self):
        self.bot.motor1.speed(FWD_SPEED)
        self.bot.motor2.speed(FWD_SPEED)
        self.bot.motor3.speed(FWD_SPEED)
        self.bot.motor4.speed(FWD_SPEED)

    def reverse(self):
        self.bot.motor1.speed(REV_SPEED)
        self.bot.motor2.speed(REV_SPEED)
        self.bot.motor3.speed(-REV_SPEED)  # keep orientation consistent
        self.bot.motor4.speed(-REV_SPEED)

    def turn_left(self):
        self.bot.motor1.speed(-TURN_SPEED)
        self.bot.motor2.speed(-TURN_SPEED)
        self.bot.motor3.speed(TURN_SPEED)
        self.bot.motor4.speed(TURN_SPEED)

    def turn_right(self):
        self.bot.motor1.speed(TURN_SPEED)
        self.bot.motor2.speed(TURN_SPEED)
        self.bot.motor3.speed(-TURN_SPEED)
        self.bot.motor4.speed(-TURN_SPEED)

    def stop_all(self):
        self.bot.stop_all()

# ---- Gesture mapping ----
def map_gesture_to_action(gesture: str) -> str:
    name = gesture.lower()
    if "ok" in name:
        return "FORWARD"
    if "palm" in name or "stop" in name:
        return "STOP"
    if "fist" in name:
        return "REVERSE"
    if "left" in name:
        return "LEFT"
    if "right" in name:
        return "RIGHT"
    return "NONE"

# ---- Model loading ----
def load_class_names(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Class names file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_model():
    from train_gesture_robot import build_model  # reuse architecture
    class_names = load_class_names(CLASS_NAMES_FILE)
    model = build_model(len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, class_names

# ---- Main loop ----
def main():
    model, class_names = load_model()
    robot = RobotController()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started. Press 'q' to quit.")
    last_action = "STOP"
    robot.stop_all()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, idx = torch.max(probs, dim=1)
            idx = idx.item()
            confidence = confidence.item()

        gesture = class_names[idx] if 0 <= idx < len(class_names) else "UNKNOWN"
        action = map_gesture_to_action(gesture)

        # Execute action
        if action == "FORWARD":
            robot.move_forward()
        elif action == "REVERSE":
            robot.reverse()
        elif action == "LEFT":
            robot.turn_left()
        elif action == "RIGHT":
            robot.turn_right()
        elif action == "STOP":
            robot.stop_all()
        # else "NONE" -> keep previous command

        if action != "NONE":
            last_action = action

        # Overlay info
        cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(frame, f"Action: {action} (last: {last_action})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        cv2.imshow("Gesture-Controlled Robot", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    robot.stop_all()
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed; robot stopped.")

if __name__ == "__main__":
    main()