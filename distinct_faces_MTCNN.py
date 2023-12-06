# 필요한 라이브러리 불러오기
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN  # MTCNN 추가
from pathlib import Path

# FaceNet 모델 로드
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# MTCNN 모델 로드
mtcnn = MTCNN()

# 인물 이름에 따른 임베딩 벡터를 저장할 딕셔너리
embeddings_dict = {}

# L2 Norm을 계산하는 함수
def L2Norm(x1, x2):
    return np.sqrt(np.sum(np.square(x1.detach().numpy() - x2.detach().numpy())))



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
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # MTCNN을 사용하여 얼굴 탐지
            faces = mtcnn.detect_faces(face_img_rgb)

            if faces:
                # 얼굴이 탐지되었다면 첫 번째 얼굴 사용
                box = faces[0]['box']
                x1, y1, w, h = map(int, box)
                x2, y2 = x1 + w, y1 + h

                # 얼굴 부분만 잘라내기
                crop_img = face_img[y1:y2, x1:x2]

                # 이미지 리사이징
                crop_img = cv2.resize(crop_img, (160, 160))

                # 얼굴 이미지의 임베딩 벡터 계산
                crop_img = torch.from_numpy(crop_img).permute(2, 0, 1).float()
                face_embedding = facenet(crop_img.unsqueeze(0))

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

    # BGR 이미지를 RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MTCNN을 사용하여 얼굴 탐지
    faces = mtcnn.detect_faces(frame_rgb)

    if faces:
        for face in faces:
            box = face['box']
            x1, y1, w, h = map(int, box)
            x2, y2 = x1 + w, y1 + h

            # 얼굴 부분만 잘라내기
            crop_img = frame[y1:y2, x1:x2]

            # 이미지 리사이징
            crop_img = cv2.resize(crop_img, (160, 160))

            # 얼굴 이미지의 임베딩 벡터 계산
            crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            crop_img = torch.from_numpy(crop_img_rgb).permute(2, 0, 1).float()
            face_embedding = facenet(crop_img.unsqueeze(0))

            # 기존의 얼굴 임베딩과 비교
            avg_embeddings_dict = {}
            for name, embs in embeddings_dict.items():
                embeddings_list = []
                for emb in embs:
                    dist = L2Norm(emb, face_embedding)
                    embeddings_list.append(dist)
                avg_embeddings_dict[name] = np.mean(embeddings_list)

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
                label = f"unsigned person, dist: {min_dist:.2f}"

            # 웹캠 영상에 바운딩 박스와 레이블 출력
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 이미지 표시
    cv2.imshow('MTCNN + FaceNet Face Recognition', frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
