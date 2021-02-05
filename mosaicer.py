import sys
import os
import requests
from PIL import Image

API_URL = 'https://dapi.kakao.com/v2/vision/face/detect'
MYAPP_KEY = 'cf2e6be05d472f52d98fe597730f279a'

class mosaicer:
    def __init__(self):

        self.headers = {'Authorization': 'KakaoAK {}'.format(MYAPP_KEY)}

    def detect_face(self,filepath):
        file_list = os.listdir(filepath)

        for path in file_list:
            print(filepath+path)

            try:
                files = { 'image' : open(filepath+path, 'rb')}

                resp = requests.post(API_URL, headers=self.headers, files=files)
                resp.raise_for_status()
                self.mosaic(filepath+path,resp.json())

            except Exception as e:
                print(str(e))
                sys.exit(0)

    def mosaic(self,filename, detection_result):
        image = Image.open(filename)

        for face in detection_result['result']['faces']:

            x = int(face['x']*image.width)
            w = int(face['w']*image.width)
            y = int(face['y']*image.height)
            h = int(face['h']*image.height)
            box = image.crop((x,y,x+w, y+h))
            box = box.resize((4,4), Image.NEAREST).resize((w,h), Image.NEAREST)
            image.paste(box, (x,y,x+w, y+h))

        image.save(filename)



if __name__ == "__main__":
    mo=mosaicer()
    mo.detect_face("./data/frames/")
#     parser = argparse.ArgumentParser(description='Mosaic faces.')
#     parser.add_argument('image_file', type=str, nargs='?', default="./data/frames/frame22.jpg",
#                         help='image file to hide faces')
#
#     args = parser.parse_args()
#
#     detection_result = detect_face(args.image_file)
#     image = mosaic(args.image_file, detection_result)
#     image.show()