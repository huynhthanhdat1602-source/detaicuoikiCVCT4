import cv2
import mediapipe as mp
import time
from scipy.spatial import distance as dist
import pygame  

# --- KHỞI TẠO ---
pygame.mixer.init() # Khởi tạo âm thanh
mp_face_mesh = mp.solutions.face_mesh 
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


try:
    alert_sound = pygame.mixer.Sound("alert.mp3")
except:
    print("Lỗi: Không tìm thấy file alert.mp3!")
    alert_sound = None

# --- CẤU HÌNH ---
EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.65 
EYE_CLOSED_SECONDS = 3.0
MOUTH_OPEN_SECONDS = 1.0

eye_start_time = None
mouth_start_time = None

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_INNER = [13, 312, 311, 78, 308, 317, 14, 87] 

def calculate_ratio(landmarks, points):
    v1 = dist.euclidean(landmarks[points[1]], landmarks[points[5]])
    v2 = dist.euclidean(landmarks[points[2]], landmarks[points[4]])
    h = dist.euclidean(landmarks[points[0]], landmarks[points[3]])
    return (v1 + v2) / (2.0 * h)

def draw_bbox(frame, landmarks, points, label, color):
    x_coords = [landmarks[p][0] for p in points]
    y_coords = [landmarks[p][1] for p in points]
    cv2.rectangle(frame, (min(x_coords)-5, min(y_coords)-5), (max(x_coords)+5, max(y_coords)+5), color, 2)
    cv2.putText(frame, label, (min(x_coords), min(y_coords)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status_text = "BINH THUONG"
    status_color = (0, 255, 0)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            img_h, img_w, _ = frame.shape
            coords = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in face_landmarks.landmark]

            avg_ear = (calculate_ratio(coords, LEFT_EYE) + calculate_ratio(coords, RIGHT_EYE)) / 2.0
            mar = calculate_ratio(coords, MOUTH_INNER)

            # --- NGỦ GẬT ---
            if avg_ear < EAR_THRESHOLD:
                if eye_start_time is None:
                    eye_start_time = time.time()

                if time.time() - eye_start_time >= EYE_CLOSED_SECONDS:
                    status_text = "NGU GAT"
                    status_color = (0, 0, 255)

                    if alert_sound and not pygame.mixer.get_busy():
                        alert_sound.play()
            else:
                eye_start_time = None

            # --- NGÁP ---
            if mar > MAR_THRESHOLD:
                if mouth_start_time is None:
                    mouth_start_time = time.time()

                if time.time() - mouth_start_time >= MOUTH_OPEN_SECONDS:
                    status_text = "MAT TAP TRUNG"
                    status_color = (0, 165, 255)
            else:
                mouth_start_time = None

    # ================= UI =================

    # Header
    cv2.rectangle(frame, (0, 0), (640, 50), (30, 30, 30), -1)
    cv2.putText(frame, "DROWSINESS DETECTION SYSTEM", (120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Status Box
    cv2.rectangle(frame, (20, 70), (300, 150), (50, 50, 50), -1)
    cv2.putText(frame, "STATUS", (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.putText(frame, status_text, (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    # EAR Bar
    bar_x = int(avg_ear * 300)  # scale
    cv2.rectangle(frame, (20, 180), (320, 210), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 180), (20 + bar_x, 210), (255, 0, 0), -1)

    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    

    

    cv2.imshow("Drowsiness Detection Pro", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()