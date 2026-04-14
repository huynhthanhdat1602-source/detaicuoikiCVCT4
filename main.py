import cv2
import mediapipe as mp
import time
from scipy.spatial import distance as dist
import pygame  # SỬA LỖI 1: Thêm import pygame

# --- KHỞI TẠO ---
pygame.mixer.init() # Khởi tạo âm thanh
mp_face_mesh = mp.solutions.face_mesh 
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# SỬA LỖI 2: Đảm bảo có file alert.mp3 trong thư mục
try:
    alert_sound = pygame.mixer.Sound("alert.mp3")
except:
    print("Lỗi: Không tìm thấy file alert.mp3!")
    alert_sound = None

# --- CẤU HÌNH ---
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.65 
EYE_CLOSED_SECONDS = 4.0
MOUTH_OPEN_SECONDS = 2.0

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
    if not success: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            img_h, img_w, _ = frame.shape
            coords = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in face_landmarks.landmark]

            avg_ear = (calculate_ratio(coords, LEFT_EYE) + calculate_ratio(coords, RIGHT_EYE)) / 2.0
            mar = calculate_ratio(coords, MOUTH_INNER)

            draw_bbox(frame, coords, LEFT_EYE, "Eye L", (0, 255, 0))
            draw_bbox(frame, coords, RIGHT_EYE, "Eye R", (0, 255, 0))
            draw_bbox(frame, coords, MOUTH_INNER, f"Mouth MAR:{mar:.2f}", (0, 255, 255))

            # Logic Cảnh báo Nhắm mắt
            if avg_ear < EAR_THRESHOLD:
                if eye_start_time is None: eye_start_time = time.time()
                if time.time() - eye_start_time >= EYE_CLOSED_SECONDS:
                    cv2.putText(frame, "CANH BAO: NGU GAT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    # SỬA LỖI 3: Thêm ngoặc () và kiểm tra để không phát đè âm thanh
                    if alert_sound and not pygame.mixer.get_busy():
                        alert_sound.play()
            else:
                eye_start_time = None

            # Logic Cảnh báo Ngáp
            if mar > MAR_THRESHOLD:
                if mouth_start_time is None: mouth_start_time = time.time()
                if time.time() - mouth_start_time >= MOUTH_OPEN_SECONDS:
                    cv2.putText(frame, "CANH BAO: DANG NGAP!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
            else:
                mouth_start_time = None

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), 1, 1.2, (255, 255, 0), 2)

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()