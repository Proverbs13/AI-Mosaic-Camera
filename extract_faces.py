import cv2
import torch
import os
import sys
from pathlib import Path
from ultralytics import YOLO

# YOLOv8 모델 로드 ('best.pt' 가중치 사용)
# model = YOLO("yolo/best.pt")
# YOLOv8 모델 로드 (pretrained 가중치 사용)
# model = YOLO("yolo/yolov8n-face.pt")
# YOLOv8 모델 로드 (11/30 3000 학습 가중치 사용)
model = YOLO("yolo/real_final_best.pt")

# 폴더 경로 설정
unaligned_folder = Path("unaligned_faces")
aligned_folder = Path("aligned_faces")
aligned_folder.mkdir(exist_ok=True)

# 각 인물 폴더를 순회하며 얼굴 크롭 및 저장
for person_folder in unaligned_folder.iterdir():
    if person_folder.is_dir():
        aligned_person_folder = aligned_folder / person_folder.name
        aligned_person_folder.mkdir(exist_ok=True)

        for image_path in person_folder.iterdir():
            if image_path.is_file():
                # 파일 확장자를 가져옵니다.
                file_extension = image_path.suffix.lower()

            if file_extension in [".png", ".jpg", ".jpeg", ".gif"]:
                # 이미지 파일인 경우 처리합니다.
                image = cv2.imread(str(image_path))
                results = model(image)
            else:
                print("file_extension error")
                sys.exit()

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

                    # 바운딩 박스로 얼굴 크롭
                    crop_img = image[y1:y2, x1:x2]

                    # 이미지 리사이징
                    crop_img = cv2.resize(crop_img, (160, 160))

                    # 크롭된 이미지 저장
                    save_path = aligned_person_folder / f"{image_path.stem}_face{i}.jpg"
                    cv2.imwrite(str(save_path), crop_img)
