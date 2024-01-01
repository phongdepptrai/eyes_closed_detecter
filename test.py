import cv2
import dlib

# Sử dụng detector của dlib để nhận diện khuôn mặt
detector = dlib.get_frontal_face_detector()

# Sử dụng bộ predictor để dự đoán điểm landmarks trên khuôn mặt
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

# Hàm để xác định trạng thái mắt (mở hoặc nhắm)
def eye_aspect_ratio(eye):
    # Tính toán tỷ lệ chiều dài và chiều rộng của mắt
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    # Tính tỷ lệ chiều dài và chiều rộng trung bình
    ear = (A + B) / (2.0 * C)
    return ear

# Khởi tạo video stream từ webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Nhận diện khuôn mặt trong hình ảnh
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = []
        right_eye = []

        # Lấy tọa độ các điểm landmarks của mắt trái và mắt phải
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))

        # Vẽ hình chữ nhật xung quanh mắt
        left_eye_pts = np.array(left_eye, np.int32)
        right_eye_pts = np.array(right_eye, np.int32)
        cv2.polylines(frame, [left_eye_pts], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye_pts], True, (0, 255, 0), 1)

        # Tính toán tỷ lệ chiều dài và chiều rộng của mắt
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Xác định trạng thái mắt (mở hoặc nhắm)
        threshold = 0.2 # Giá trị ngưỡng có thể thay đổi
        if left_ear < threshold and right_ear < threshold:
            cv2.putText(frame, "CLOSED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "OPEN", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Eye Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()