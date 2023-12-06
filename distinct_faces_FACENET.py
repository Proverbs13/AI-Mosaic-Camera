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
yolo = YOLO("yolo/knife_face_cigarette-detection_3000_11301007_best.pt")

# 인물 이름에 따른 임베딩 벡터를 저장할 딕셔너리
embeddings_dict = {}

# L2 Norm을 계산하는 함수
def L2Norm(x1, x2):  # L2Norm: 벡터간 유클리드 거리 계산
    return np.sqrt(np.sum((x1 - x2) ** 2))

# 얼굴 임베딩을 계산하는 함수
def get_embedding(face_img):
    # 이미지 전처리 및 임베딩 계산
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img / 255.0  # 이미지 정규화
    face_img = np.transpose(face_img, (2, 0, 1))
    face_img = torch.tensor(face_img).float().unsqueeze(0)

    # 얼굴 이미지의 임베딩 벡터 계산
    with torch.no_grad():
        face_embedding = facenet(face_img)

    return face_embedding[0].numpy()

# 'aligned_faces' 디렉토리의 모든 인물 폴더에 대해
for person_folder in Path('aligned_faces').iterdir():
    if person_folder.is_dir():
        # 인물의 이름을 디렉토리 이름에서 가져옴
        person_name = person_folder.name

        # 해당 인물의 얼굴 이미지에 대한 임베딩 벡터를 저장할 리스트 초기화
        person_embeddings = []

        # 인물 폴더의 모든 얼굴 이미지에 대해
        for face_img_path in person_folder.glob("*.jpg"):
            # 얼굴 이미지 불러오기
            face_img = cv2.imread(str(face_img_path))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # 이미지에서 임베딩 계산
            face_embedding = get_embedding(face_img)

            # 임베딩 벡터를 리스트에 추가
            person_embeddings.append(face_embedding)

        # 인물 이름에 따른 임베딩 벡터를 딕셔너리에 저장
        embeddings_dict[person_name] = person_embeddings

# 웹캠 캡처 객체 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 화면을 좌우로 반전
    frame = cv2.flip(frame, 1)

    # YOLO 모델을 사용하여 얼굴 탐지
    results = yolo(frame)

    # 'results[0].names' 사전 반전
    names_inv = {v: k for k, v in results[0].names.items()}

    # 'face' 클래스의 인덱스 찾기
    face_idx = None
    for key, value in results[0].names.items():
        if value == 'face':
            face_idx = key
            break

    if face_idx is None:
        print("'face' 클래스를 찾을 수 없습니다.")
    else:
        # 'face' 클래스의 바운딩 박스 정보 추출
        face_boxes = results[0].boxes.data[results[0].boxes.cls == face_idx]

        for i, (x1, y1, x2, y2, conf, cls_idx) in enumerate(face_boxes):
            # 바운딩 박스 좌표를 정수로 변환
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # 바운딩 박스의 너비와 높이 계산
            width = x2 - x1
            height = y2 - y1

            # 바운딩 박스의 너비나 높이가 3 픽셀 미만인 경우 처리를 건너뜀
            if width < 3 or height < 3:
                continue

            # 바운딩 박스로 얼굴 크롭
            crop_img = frame[y1:y2, x1:x2]

            # 이미지 리사이징
            crop_img = cv2.resize(crop_img, (160, 160))

            # 얼굴 이미지의 임베딩 벡터 계산
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            crop_img_embedding = get_embedding(crop_img)

            # 웹캠 얼굴 임베딩 출력
            # print(len(crop_img_embedding))
            # print(crop_img_embedding)

            avg_embeddings_dict = {}

            # 기존의 얼굴 임베딩과 비교
            for name, embs in embeddings_dict.items():
                embeddings_list = []
                for emb in embs:
                    dist = L2Norm(emb, crop_img_embedding)
                    embeddings_list.append(dist)
                # 각 인물에 대한 평균 거리 저장
                avg_embeddings_dict[name] = np.mean(embeddings_list)

            # 각 인물별 평균 거리
            # print(avg_embeddings_dict)

            # 평균 거리가 최소인 인물 찾기
            min_dist = float('inf')
            min_name = None
            for name, dist in avg_embeddings_dict.items():
                if dist < min_dist:
                    min_dist = dist
                    min_name = name

            # 가장 작은 거리를 갖는 인물의 이름과 거리를 출력
            if min_dist < 10.0:
                label = f"{min_name}, dist: {min_dist:.2f}"
            else:
                label = f"unknown, dist: {min_dist:.2f}"

            # 웹캠 영상에 바운딩 박스와 레이블 출력
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 이미지 표시
    cv2.imshow('YOLOv8 Face Detection', frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
