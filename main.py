# 필요한 라이브러리 불러오기
import cv2
import torch

# YOLOv5 커스텀 모델 로드 (사용자가 훈련시킨 모델 'best.pt' 사용)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo/best.pt')  # 여기서 'best.pt'는 사용자가 제공하는 파일 경로로 변경해야 함

# 웹캠 캡처 객체 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델을 사용하여 얼굴 탐지
    results = model(frame)
    faces = results.xyxy[0]  # 탐지된 객체의 바운딩 박스

    for x1, y1, x2, y2, conf, cls in faces:
        # 바운딩 박스와 클래스 이름(여기서는 'person') 그리기
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'person {conf:.2f}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 이미지 표시
    cv2.imshow('YOLOv5 Face Detection', frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
