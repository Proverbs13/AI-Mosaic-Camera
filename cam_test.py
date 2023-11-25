import cv2

cap = cv2.VideoCapture(0)  # 컴퓨터에 연결된 디폴트 웹캠 캡쳐

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 1080
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('mysupervideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20,
                         (width, height))  # mac or linux *'XVID'

# 20 은 초당 찍는 프레임 수로, 20~30 사이. 숫자가 높을 수록 파일 크기는 커진다.
# VideoWriter_fourcc : mp4 파일을 입력하기 위해 사용하는 실제 비디오 코덱을 의미한다. 이 변수의 중요한 점은 운영체제에 따라 다르다는 점.
# 코덱을 명시하기 위해 사용된 4바이트 코드. fourcc.org에 있다.
while True:
    # 기본적으로 비디오 캡쳐가 보내온 싱글 이미지를 프레임으로 사용.
    ret, frame = cap.read()

    # OPERATION (DRAWING)
    writer.write(frame)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', gray)
    cv2.imshow('frame', frame)  # 컬러

    if cv2.waitKey(1) & 0xFFF == 27:  # ord('q') q를 누를 때
        break

cap.release()
writer.release()
cv2.destroyAllWindows()  # 모든 윈도우를 없애줌.