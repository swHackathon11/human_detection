import cv2
import os
from detecter import Detecter
import schedule
import time
import datetime
import matplotlib.pyplot as plt
from config import SAMPLING_CYCLE,TOTAL_MIN,CAPTURE_CYCLE

Detecter = Detecter()

def job():
    stack=[]

    target = datetime.datetime.now().strftime("%Y-%m-%d")+".mp4"
    # target='2021-02-06.mp4'
    cam = cv2.VideoCapture("data/videos/" + target)
    minute=datetime.datetime.now().minute
    FPS = 1
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print('Error: Creating directory of data')
    currentframe = 0
    capture_rate = 60#int(cam.get(cv2.CAP_PROP_FPS))
    end_of_frame = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(end_of_frame)

    stat_of_frame =0# minute*(cam.get(cv2.CAP_PROP_FPS) * 60)

    # print(stat_of_frame)
    for i in range(TOTAL_MIN//SAMPLING_CYCLE):
        for j in range(SAMPLING_CYCLE):

            cam.set(cv2.CAP_PROP_POS_FRAMES, CAPTURE_CYCLE * (currentframe*FPS) + stat_of_frame)
            ret, frame = cam.read()

            if ret:
                # if currentframe % capture_rate == 0:
                name = 'data/frames/frame' + str(j) + '.jpg'
                cv2.imwrite(name, frame)
            else:
                break
            currentframe += 1


        result = Detecter.detect()
        output=round(result/CAPTURE_CYCLE,2)
        print(datetime.datetime.now().strftime("%Y-%m-%d"),i,": Avg =",output)

        stack.append(output)
    plt.plot(range(TOTAL_MIN//SAMPLING_CYCLE),stack)
    plt.savefig(target+'result.png')
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # schedule.every(10).seconds.do(job)
    job()
    # schedule.every().minute.do(job)
    #
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)