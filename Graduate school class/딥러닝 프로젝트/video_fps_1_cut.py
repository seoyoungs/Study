  
import cv2
from PIL import Image
import numpy as np

# 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class
# 영상이 있는 경로
path = "D:/company_python/ML_post_training_code/000000652_000034_2022_08_03_22_09_20.mp4"
imagePath = "D:/company_python/ML_post_training_code/test/frame%d.png"
vidcap = cv2.VideoCapture(path)
# imagePath ="D:/공유폴더/서영/전처리작업에_필요한_코드/test/frame%d.png"

count = 1

while(vidcap.isOpened()):
    try:
        ret, image = vidcap.read()
        # 이미지 사이즈 960x540으로 변경
        # if image is None:
        #     print('Wrong path:', path)
        # else:
        #     image1 = cv2.resize(image, dsize=(1280,720))

        
        # 30프레임당 하나씩 이미지 추출
        if(int(vidcap.get(1)) % 30 == 0):
            print('Saved frame number : ' + str(int(vidcap.get(1))))
            # 추출된 이미지가 저장되는 경로
            
            image = cv2.resize(image, (1280, 720))
            
            cv2.imwrite(imagePath % count, image)
            #print('Saved frame%d.jpg' % count)
            count += 1
            
    except Exception as e:
        break
    

        
vidcap.release()