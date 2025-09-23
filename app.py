from flask import Flask, render_template, Response, jsonify
import cv2, os, pickle, face_recognition, numpy as np
from datetime import datetime
import csv
import face_detection

app = Flask(__name__)

ENCODINGS_FILE = "encodings/face_encodings.pkl"
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
else:
    data = {"encodings": [], "names": [], "tolerance": 0.4}

TOLERANCE = data.get("tolerance", 0.4)

attendance_log = {}   # lưu tạm thời để hiển thị trên web
seen_counter = {}     # đếm số lần liên tiếp

CSV_FILE = "attendance.csv"

# Hàm ghi điểm danh vào CSV
def save_to_csv(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    # Nếu file chưa có thì tạo với header
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    # Kiểm tra đã ghi hôm nay chưa
    already_marked = False
    with open(CSV_FILE, mode="r") as f:
        for row in csv.reader(f):
            if row and row[0] == name and row[1] == today:
                already_marked = True
                break

    if not already_marked:
        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, today, time_now])
        print(f"📌 Đã lưu điểm danh: {name} - {today} {time_now}")

def mark_attendance(name):
    if name != "Unknown":
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attendance_log[name] = now
        save_to_csv(name)

def generate_frames():
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 5 != 0:  # skip frame để giảm lag
            continue

        # Resize frame để xử lý nhanh hơn
        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, boxes)

        for (top, right, bottom, left), encoding in zip(boxes, encs):
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=TOLERANCE)
            name = "Unknown"

            if True in matches:
                distances = face_recognition.face_distance(data["encodings"], encoding)
                idx = np.argmin(distances)
                name = data["names"][idx]

            # Đếm liên tiếp để chắc chắn
            seen_counter[name] = seen_counter.get(name, 0) + 1
            if seen_counter[name] >= 3:
                mark_attendance(name)
                seen_counter[name] = 0

            # Scale box về size gốc
            top, right, bottom, left = [v*2 for v in (top, right, bottom, left)]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/attendance')
def attendance():
    mark_attendance()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
