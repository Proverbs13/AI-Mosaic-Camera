from ultralytics import YOLO
import cv2
import math

# 터미널에서 pip install ultralytics 이후 진행해야함
# model 아직 학습이 다 안되서 정확도 측면에서는 떨어짐

# model = 개인 경로에 맞게 수정해야함
# model = YOLO('C:/Code/PROJECT/yolo_mosaic_test/models/best1027_4.pt')
# model = YOLO('C:/Code/PROJECT/yolo_mosaic_test/models/bestknife_face_cigarette-detection_3000_11301007.pt')
# model = YOLO('C:/Code/PROJECT/yolo_mosaic_test/models/best_hoi_9000.pt')
# model = YOLO('C:/Code/PROJECT/yolo_mosaic_test/models/새로운8000.pt')
model = YOLO('yolo/real_final_best.pt')



# model = YOLO('C:/Code/PROJECT/yolo_mosaic_test/models/bestknife_face_cigarette-detecti ton_3000_11301007.pt')

# model = YOLO('C:/Code/PROJECT/yolo_mosaic_test/models/best3.pt')
# model = YOLO('C:/Code/PROJECT/yolo_mosaic_test/models/best.pt')
# model = YOLO('C:/Code/PROJECT/yolo_mosaic_test/models/yolov8n.pt')
# model = YOLO('C:/Code/PROJECT/yolo_mosaic_test/models/yolov8n-face.pt')


# object classes (표기 클래스)
# classNames = ["cigarette","face","knife"]
classNames = ["cigarette","face","knife"]
# classNames = ["1","2","3"]
#classNames = ["face"]
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat","traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat","dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella","handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush"]



#start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

while True:
    success, img = cap.read()
    results = model(img, stream=True)


    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

#####
            # 박스 영역 모자이크 처리 코드 / 주석 처리하면 모자이크 사라짐
            face_roi = img[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            img[y1:y2, x1:x2] = blurred_face

#####
            # 박스 그리기 / 주석 처리하면 박스 사라짐
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
#####


            # 정확도 정보와 클래스 분류 print

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

#####
            # 클래스이름 글씨 모양 설정, 캠에서 표시하는 코드 / 주석 처리하면 사라짐
            # 글씨 모양 설정
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            # 클래스 이름 표시 코드
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
#####



    cv2.imshow('Webcam', img)

    ## Q 누르면 캠 종료
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# # 사진 한개 테스트용 코드 = 개인 경로에 맞게 수정해야함
# result = model.predict("C:/Code/PROJECT/yolo_mosaic_test/models/test_data/k.jpg", save=True, conf=0.5)
# plots = result[0].plot()
# cv2.imshow("plot", plots)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
