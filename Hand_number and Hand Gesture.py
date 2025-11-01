import cv2
import mediapipe as mp

# Initialize MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert BGR to RGB
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms, handType in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            # Hand label (Left/Right)
            hand_label = handType.classification[0].label  # "Left" or "Right"

            # Store landmark positions
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                lmList.append((int(lm.x * w), int(lm.y * h)))

            # Finger tip IDs (Thumb, Index, Middle, Ring, Pinky)
            tipIds = [4, 8, 12, 16, 20]
            fingers = []

            # Thumb
            if hand_label == "Right":
                fingers.append(1 if lmList[tipIds[0]][0] > lmList[tipIds[0] - 1][0] else 0)
            else:  # Left hand
                fingers.append(1 if lmList[tipIds[0]][0] < lmList[tipIds[0] - 1][0] else 0)

            # Other four fingers
            for id in range(1, 5):
                fingers.append(1 if lmList[tipIds[id]][1] < lmList[tipIds[id] - 2][1] else 0)

            totalFingers = fingers.count(1)

            # Display hand type and finger count
            cv2.putText(frame, f"{hand_label} Hand - Fingers: {totalFingers}",
                        (10, 50 if hand_label == "Right" else 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

