import cv2
import torch
from facenet_pytorch import InceptionResnetV1
from pathlib import Path
from ultralytics import YOLO

# FaceNet 모델 로드
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# YOLO 모델 로드 ('best.pt' 가중치 사용)
yolo = YOLO("yolo/best.pt")

# 인물별 얼굴 임베딩 딕셔너리 초기화
embeddings_dict = {}

# 'aligned_faces' 디렉토리의 각 인물 폴더에 대해
for person_folder in Path('aligned_faces').iterdir():
    if person_folder.is_dir():
        # 인물의 이름을 디렉토리 이름에서 가져오기
        person_name = person_folder.name

        # 해당 인물의 얼굴 임베딩을 저장할 리스트
        person_embeddings = []

        # 인물 폴더의 모든 얼굴 이미지에 대해
        for face_img_path in person_folder.glob("*.jpg"):
            # 얼굴 이미지 로드
            face_img = cv2.imread(str(face_img_path))

            # YOLO로 얼굴 탐지
            results = yolo(face_img)

            # 얼굴 부분 추출
            for box in results:
                if box[-1] == 0:  # 0: face class ID
                    x1, y1, x2, y2 = map(int, box[:4])
                    cropped_face = face_img[y1:y2, x1:x2]

                    # 얼굴 이미지 전처리 및 FaceNet에 입력할 준비
                    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    cropped_face = torch.from_numpy(cropped_face).permute(2, 0, 1).float()

                    # 얼굴 이미지의 임베딩 계산
                    face_embedding = facenet(cropped_face.unsqueeze(0))
                    person_embeddings.append(face_embedding)

        # 각 인물의 임베딩 리스트를 딕셔너리에 저장
        embeddings_dict[person_name] = person_embeddings


# 두 얼굴 임베딩 간의 거리를 계산하여 인물을 구분하는 함수
def compare_faces(embedding1, embedding2, threshold=0.6):
    distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)
    return distance < threshold


# 웹캠 캡처 객체 초기화
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 1080
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('mysupervideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20,
                         (width, height))  # mac or linux *'XVID'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    writer.write(frame)
    # YOLO 모델을 사용하여 얼굴 탐지
    results = yolo(frame)

    # 얼굴 부분 추출
    for box in results:
        if box[-1] == 0:  # 0: face class ID
            x1, y1, x2, y2 = map(int, box[:4])
            cropped_face = frame[y1:y2, x1:x2]

            # 얼굴 이미지 전처리 및 FaceNet에 입력할 준비
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            cropped_face = torch.from_numpy(cropped_face).permute(2, 0, 1).float()

            # 얼굴 이미지의 임베딩 계산
            face_embedding = facenet(cropped_face.unsqueeze(0))

            # 두 얼굴 임베딩 간의 거리 계산 및 확인하는 코드
            for person_name, person_embeddings in embeddings_dict.items():
                for embedding1 in person_embeddings:
                    same_person = compare_faces(embedding1.squeeze(), face_embedding.squeeze())
                    print(f"두 얼굴은 동일한 사람입니까? {same_person}")

            # 결과 이미지 표시
            cv2.imshow('Face Recognition', frame)

            # 'q' 키를 눌러 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
