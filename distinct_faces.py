# 필요한 라이브러리 불러오기
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from ultralytics import YOLO
from pathlib import Path

# FaceNet 및 YOLO 모델 로드
facenet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=True)
yolo = YOLO("yolo/best.pt")  # YOLOv8 'best.pt' 가중치 사용

# 인물 이름에 따른 임베딩 벡터를 저장할 딕셔너리
embeddings_dict = {}

# 'aligned_faces' 디렉토리의 모든 인물 폴더에 대해
for person_folder in Path('aligned_faces').iterdir():
    if person_folder.is_dir():
        # 인물의 이름을 디렉토리 이름에서 가져옴
        person_name = person_folder.name

        # 해당 인물의 얼굴 이미지에 대한 임베딩 벡터를 저장할 리스트
        person_embeddings = []

        # 인물 폴더의 모든 얼굴 이미지에 대해
        for face_img_path in person_folder.glob("*.jpg"):
            # 얼굴 이미지 불러오기
            face_img = cv2.imread(str(face_img_path))

            # BGR 이미지를 RGB로 변환
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # 이미지를 PyTorch 텐서로 변환하고 float 형식으로 변환
            face_img = torch.from_numpy(face_img).permute(2, 0, 1).float()

            # 얼굴 이미지의 임베딩 벡터 계산
            face_embedding = facenet(face_img.unsqueeze(0))

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
            crop_img = torch.from_numpy(crop_img).permute(2, 0, 1).float()
            crop_img_embedding = facenet(crop_img.unsqueeze(0))

            # 기존의 얼굴 임베딩과 비교
            min_dist = float('inf')
            min_name = None
            for name, embs in embeddings_dict.items():
                for emb in embs:
                    dist = torch.norm(emb - crop_img_embedding)
                    if dist < min_dist:
                        min_dist = dist
                        min_name = name

            # 가장 작은 거리를 갖는 인물의 이름과 거리를 출력
            if min_dist < 0.2:
                label = f"{min_name}, dist: {min_dist:.2f}"
            else:
                label = f"unsigned person, dist: {min_dist:.2f}"

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
