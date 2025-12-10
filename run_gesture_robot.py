import os
import cv2
import numpy as np
import torch
from torchvision import transforms

MODEL_PATH = "gesture_model.pt"
CLASS_NAMES_FILE = "class_names.txt"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class VirtualRobot:
    """Simple virtual robot on a 2D grid."""

    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.x = grid_size // 2
        self.y = grid_size // 2
        self.direction = "UP"   # UP, DOWN, LEFT, RIGHT

    def move_forward(self):
        if self.direction == "UP":
            self.y = max(0, self.y - 1)
        elif self.direction == "DOWN":
            self.y = min(self.grid_size - 1, self.y + 1)
        elif self.direction == "LEFT":
            self.x = max(0, self.x - 1)
        elif self.direction == "RIGHT":
            self.x = min(self.grid_size - 1, self.x + 1)

    def reverse(self):
        if self.direction == "UP":
            self.y = min(self.grid_size - 1, self.y + 1)
        elif self.direction == "DOWN":
            self.y = max(0, self.y - 1)
        elif self.direction == "LEFT":
            self.x = min(self.grid_size - 1, self.x + 1)
        elif self.direction == "RIGHT":
            self.x = max(0, self.x - 1)

    def turn_left(self):
        if self.direction == "UP":
            self.direction = "LEFT"
        elif self.direction == "LEFT":
            self.direction = "DOWN"
        elif self.direction == "DOWN":
            self.direction = "RIGHT"
        elif self.direction == "RIGHT":
            self.direction = "UP"

    def turn_right(self):
        if self.direction == "UP":
            self.direction = "RIGHT"
        elif self.direction == "RIGHT":
            self.direction = "DOWN"
        elif self.direction == "DOWN":
            self.direction = "LEFT"
        elif self.direction == "LEFT":
            self.direction = "UP"

    def stop(self):
        pass  # no movement

    def execute_action(self, action: str):
        if action == "MOVE FORWARD":
            self.move_forward()
        elif action == "REVERSE":
            self.reverse()
        elif action == "TURN LEFT":
            self.turn_left()
        elif action == "TURN RIGHT":
            self.turn_right()
        elif action == "STOP":
            self.stop()

    def state_string(self):
        return f"Pos=({self.x},{self.y}) Dir={self.direction}"


def load_class_names(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Class names file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    print("Loaded class names:", names)
    return names


def map_gesture_to_action(gesture: str) -> str:
    name = gesture.lower()
    if "ok" in name:
        return "MOVE FORWARD"
    if "palm" in name or "stop" in name:
        return "STOP"
    if "fist" in name:
        return "REVERSE"
    if "left" in name:
        return "TURN LEFT"
    if "right" in name:
        return "TURN RIGHT"
    return "NO ACTION"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    from train_gesture_robot import build_model  # reuse architecture
    class_names = load_class_names(CLASS_NAMES_FILE)
    model = build_model(len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, class_names


def run_webcam(model, class_names):
    robot = VirtualRobot()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Webcam started. Press 'q' to quit.")

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
        if action != "NO ACTION":
            robot.execute_action(action)

        cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(frame, f"Action:  {action}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.putText(frame, robot.state_string(),
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

        cv2.imshow("Gesture-Controlled Virtual Robot", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


def main():
    model, class_names = load_model()
    run_webcam(model, class_names)


if __name__ == "__main__":
    main()