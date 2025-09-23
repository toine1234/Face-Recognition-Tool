import cv2
import pickle
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import datetime, csv

# ======================
# Load dữ liệu đã train (pkl chứa embeddings + names)
# ======================
with open("encodings/face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

TOLERANCE = data.get("tolerance", 0.48)  # ngưỡng cosine similarity
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================
# Load model (pretrained hoặc fine-tuned nếu có .pth)
# ======================
model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)

# Nếu có model fine-tune thì load, nếu không thì dùng pretrained
try:
    model.load_state_dict(torch.load("models/facenet_triplet.pth", map_location=device))
    print("✅ Loaded fine-tuned model facenet_triplet.pth")
except FileNotFoundError:
    print("⚠️ Không tìm thấy facenet_triplet.pth, dùng pretrained FaceNet.")

model.eval()

# Detector MTCNN
mtcnn = MTCNN(image_size=160, margin=20, device="cpu")

# ======================
# Hàm trích embedding
# ======================
def get_embedding(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    face = mtcnn(pil)
    if face is None:
        return None
    with torch.no_grad():
        emb = model(face.unsqueeze(0).to(device)).cpu().numpy()[0]
    # chuẩn hóa L2
    n = np.linalg.norm(emb)
    if n > 0:
        emb = emb / n
    return emb

# ======================
# Ghi điểm danh
# ======================
def mark_attendance(name):
    now = datetime.datetime.now()
    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), "Present"])

# ======================
# Chụp và nhận diện
# ======================
def capture_and_recognize():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không mở được camera")
        return

    print("📸 Nhấn SPACE để chụp, ESC để thoát")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Press SPACE to Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC thoát
            break
        elif key == 32:  # SPACE để chụp
            emb = get_embedding(frame)
            if emb is None:
                print("⚠️ Không phát hiện khuôn mặt. Vui lòng thử lại.")
                continue

            sims = np.dot(data["encodings"], emb)  # cosine similarity vì đã L2 normalize
            idx = np.argmax(sims)
            best_score = sims[idx]

            print(f"[DEBUG] Similarity scores: {sims}")   # in toàn bộ điểm số
            print(f"[DEBUG] Best match: {data['names'][idx]} với score={best_score:.4f}")

            if best_score > TOLERANCE:
                name = data["names"][idx]
                print(f"✅ Điểm danh thành công: {name}")
                mark_attendance(name)
            else:
                print("❌ Không khớp trong cơ sở dữ liệu. Người dùng chưa đăng ký.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_recognize()
