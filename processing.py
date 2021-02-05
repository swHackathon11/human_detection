import cv2
import os
from detecter import Detecter
from mosaicer import mosaicer
import schedule
import time
import datetime
import matplotlib.pyplot as plt
from config import SAMPLING_CYCLE,TOTAL_MIN,CAPTURE_CYCLE

Detecter = Detecter()
mosaicer = mosaicer()

def job():
    stack=[]
    PATH="data/images/"
    date=datetime.datetime.now().strftime("%Y-%m-%d")
    target =date+".mp4"
    os.mkdir(PATH+date)


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
                if j == 0:
                    # print(PATH+date+'/'+str(i))
                    cv2.imwrite(PATH+date+'/'+str(i)+'.jpg', frame)
            else:
                break
            currentframe += 1


        result = Detecter.detect()
        output=round(result/CAPTURE_CYCLE,2)

        # print(datetime.datetime.now().strftime("%Y-%m-%d"),i,": Avg =",output)

        stack.append(output)

    plt.plot(range(TOTAL_MIN//SAMPLING_CYCLE),stack)
    plt.savefig(target+'result.png')

    mosaicer.process(PATH+date+'/')

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # scheduler for server

    # schedule.every(10).seconds.do(job)

    #local option
    job()

    #run

    # schedule.every().minute.do(job)
    #
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)