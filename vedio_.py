import cv2
import time


def __main__():
    path = 'love/1.mp4'
    cap = cv2.VideoCapture(path)  # path가 0, 1이면 웹캠, 경로면 파일 읽기
    frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
    frame_num = (cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_time = 0
    FPS = 800

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # cv2.VideoWriter params
    # (path, fourcc, fps, (w, h))
    out = cv2.VideoWriter('test/0.avi', fourcc, 60.0, (int(frame_size[1]), int(frame_size[0])))

    for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, img = cap.read()  # ret은 프레임 읽은 결괏값: True or False, img는 프레임
        current_time = time.time() - prev_time  # 현재 시간 설정

        if (ret is True) and (current_time > 1. / FPS):  # 현재 시간이 FPS에 합당한지?

            prev_time = time.time()  # prev_time 초기화
            out.write(img)  # 프레임 쓰기
    cap.release()


def vedio_cap(path, save_path='test/0.avi', FPS=300):
    cap = cv2.VideoCapture(path)  # path가 0, 1이면 웹캠, 경로면 파일 읽기
    frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
    frame_num = (cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_time = 0
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # cv2.VideoWriter params
    # (path, fourcc, fps, (w, h))
    out = cv2.VideoWriter(save_path, fourcc, 60.0, (int(frame_size[1]), int(frame_size[0])))

    for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, img = cap.read()  # ret은 프레임 읽은 결괏값: True or False, img는 프레임
        current_time = time.time() - prev_time  # 현재 시간 설정

        if (ret is True) and (current_time > 1. / FPS):  # 현재 시간이 FPS에 합당한지?

            prev_time = time.time()  # prev_time 초기화
            out.write(img)  # 프레임 쓰기
    cap.release()
