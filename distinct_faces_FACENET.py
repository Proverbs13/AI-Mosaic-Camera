# 필요한 라이브러리 불러오기
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from pathlib import Path

# FaceNet 모델 로드
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Yolo 모델 로드
yolo = YOLO("yolo/final_best.pt")

# 인물 이름에 따른 임베딩 벡터를 저장할 딕셔너리
embeddings_dict = {}

# L2 Norm을 계산하는 함수
def L2Norm(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# 얼굴 임베딩을 계산하는 함수
def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img / 255.0
    face_img = np.transpose(face_img, (2, 0, 1))
    face_img = torch.tensor(face_img).float().unsqueeze(0)

    with torch.no_grad():
        face_embedding = facenet(face_img)

    return face_embedding[0].numpy()

# 'aligned_faces' 디렉토리의 모든 인물 폴더에 대해
for person_folder in Path('aligned_faces').iterdir():
    if person_folder.is_dir():
        person_name = person_folder.name
        person_embeddings = []

        for face_img_path in person_folder.glob("*.jpg"):
            face_img = cv2.imread(str(face_img_path))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_embedding = get_embedding(face_img)
            person_embeddings.append(face_embedding)

        embeddings_dict[person_name] = person_embeddings

# 블러 처리 함수
def apply_blur(image, top_left, bottom_right, kernel_size=(50, 50)):
    x1, y1 = top_left
    x2, y2 = bottom_right
    roi = image[y1:y2, x1:x2]
    roi = cv2.blur(roi, kernel_size)
    image[y1:y2, x1:x2] = roi
    return image

# 웹캠 캡처 객체 초기화
cap = cv2.VideoCapture(0)

cv2.namedWindow('YOLOv8 Face Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLOv8 Face Detection', 1500, 1200)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # YOLO 모델을 사용하여 얼굴 및 객체 탐지
    results = yolo(frame)

    # 각 클래스('face', 'knife', 'cigarette')의 인덱스 찾기
    class_indices = {}
    for idx, class_name in results[0].names.items():
        if class_name in ['face', 'knife', 'cigarette']:
            class_indices[class_name] = idx

    # 각 클래스에 대한 바운딩 박스 정보 추출 및 처리
    for class_name, class_idx in class_indices.items():
        boxes = results[0].boxes.data[results[0].boxes.cls == class_idx]
        for box in boxes:
            x1, y1, x2, y2, conf, cls_idx = map(int, box)

            # 얼굴 인식 및 처리
            if class_name == 'face':
                crop_img = frame[y1:y2, x1:x2]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                crop_img_embedding = get_embedding(crop_img)

                avg_embeddings_dict = {}

                for name, embs in embeddings_dict.items():
                    embeddings_list = []
                    for emb in embs:
                        dist = L2Norm(emb, crop_img_embedding)
                        embeddings_list.append(dist)
                    avg_embeddings_dict[name] = np.mean(embeddings_list)

                min_dist = float('inf')
                min_name = None
                for name, dist in avg_embeddings_dict.items():
                    if dist < min_dist:
                        min_dist = dist
                        min_name = name

                if min_dist < 0.7:
                    label = f"{min_name}, dist: {min_dist:.2f}"
                else:
                    label = f"unknown, dist: {min_dist:.2f}"
                    frame = apply_blur(frame, (x1, y1), (x2, y2))
            else:  # 'knife'나 'cigarette' 클래스의 경우
                frame = apply_blur(frame, (x1, y1), (x2, y2))
                label = class_name

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
