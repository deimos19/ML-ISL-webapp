from flask import Flask, render_template, Response, request, send_file
import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import os

app = Flask(__name__)


IMAGE_FOLDER = 'eng2sign'


camera = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)


model_path = 'model/ISL_model1_Full.h5'
model = load_model(model_path)

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

offset = 20
imgSize = 300

def get_sign_image(letter):
    file_name = f'{letter.upper()}.jpg'
    img_path = os.path.join(IMAGE_FOLDER, file_name)
    img = Image.open(img_path)
    return img

def text_to_sign_language(text):
    img_list = []
    for char in text:
        if char.isalpha():
            img = get_sign_image(char)
            if img:
                img_list.append(img)

    total_width = sum(img.width for img in img_list)
    max_height = max(img.height for img in img_list)

    combined_img = Image.new('RGB', (total_width, max_height), (0, 0, 0))
    x_offset = 0
    for img in img_list:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return combined_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/convert_text', methods=['POST'])
def convert_text():
    text = request.form.get('text', '').strip()
    if text:
        img = text_to_sign_language(text)
        if img:
            img_io = BytesIO()
            img.save(img_io, 'JPEG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')
    return 'Error converting text to sign language.', 400

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        imgOut = frame.copy()
        hands, frame = detector.findHands(frame)
        landmark_image = np.zeros_like(frame)
        
        if hands:
            for hand in hands:
                lmList = hand["lmList"]
                for i, lm in enumerate(lmList):
                    cv2.circle(landmark_image, (lm[0], lm[1]), 5, (0, 255, 0), cv2.FILLED)

                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20),
                    (5, 9), (9, 13), (13, 17)
                ]

                for start, end in connections:
                    cv2.line(landmark_image, (lmList[start][0], lmList[start][1]),
                            (lmList[end][0], lmList[end][1]), (0, 255, 0), 2)

            x_min = min(hand['bbox'][0] for hand in hands)
            y_min = min(hand['bbox'][1] for hand in hands)
            x_max = max(hand['bbox'][0] + hand['bbox'][2] for hand in hands)
            y_max = max(hand['bbox'][1] + hand['bbox'][3] for hand in hands)

            x = x_min - offset
            y = y_min - offset
            w = x_max - x_min + 2 * offset
            h = y_max - y_min + 2 * offset

            imgCropped = landmark_image[y:y + h, x:x + w]

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            imgCropShape = imgCropped.shape
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                widthCalculated = math.ceil(k * w)
                imgResize = cv2.resize(imgCropped, (widthCalculated, imgSize))
                widthGap = math.ceil((imgSize - widthCalculated) / 2)
                imgWhite[:, widthGap:widthCalculated + widthGap] = imgResize
            else:
                k = imgSize / w
                heightCalculated = math.ceil(k * h)
                imgResize = cv2.resize(imgCropped, (imgSize, heightCalculated))
                heightGap = math.ceil((imgSize - heightCalculated) / 2)
                imgWhite[heightGap:heightCalculated + heightGap, :] = imgResize

            frame_resized = cv2.resize(imgWhite, (300, 300))
            frame_normalized = frame_resized / 255.0

            frame_expanded = np.expand_dims(frame_normalized, axis=-1)
            frame_expanded = np.expand_dims(frame_expanded, axis=0)

            predictions = model.predict(frame_expanded)
            predicted_class = np.argmax(predictions)
            predicted_label = labels[predicted_class]
            confidence = np.max(predictions)

            cv2.putText(imgWhite, f'{predicted_label}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)

            cv2.rectangle(imgOut, (x - offset, y - offset - 50),
                        (x - offset + 90, y - offset - 50 + 50), (230, 230, 0), cv2.FILLED)
            cv2.putText(imgOut, predicted_label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOut, (x - offset, y - offset),
                        (x + w + offset, y + h + offset), (230, 230, 0), 4)

        ret, buffer = cv2.imencode('.jpg', imgOut)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    app.run(debug=True)
