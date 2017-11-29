import cv2 
import numpy as np
import os
from datetime import datetime

## 
cascade_path = "./haarcascade_frontalface_alt2.xml"
cascade=cv2.CascadeClassifier(cascade_path)

def capture_camera(scale, crop_face, save_path, num):
    print(cascade)
    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        raise("IO Error")
    for i in range(num):
        ret, frame = cap.read()
        orgHeight, orgWidth = frame.shape[:2]
        """
        while (orgWidth > 1000 or orgHeight > 1000):
            frame = cv2.resize(frame, (int(orgWidth/2), int(orgHeight/2)))
            orgHeight, orgWidth = frame.shape[:2]
        """
        facerect=cascade.detectMultiScale(
            cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),
            scaleFactor=1.11,
            minNeighbors=3,
            minSize=(100,100)
        )
        img=frame.copy()
            
        if len(facerect)!=0:
            for rect in facerect:
                org_length=rect[3]/2.
                face_center=np.array((rect[0:2]+rect[2:4]/2.)-np.array([0,org_length*0.25])).astype(np.int32)
                af_length=int(min([min(face_center[0]+org_length*scale,img.shape[1])-face_center[0],face_center[0]-max(face_center[0]-org_length*scale,0),min(face_center[1]+org_length*scale,img.shape[0])-face_center[1],face_center[1]-max(face_center[1]-org_length*scale,0)]))
                cv2.rectangle(
                    img,
                    tuple(face_center-af_length),
                    tuple(face_center+af_length),
                    (0,300,300),
                    thickness=2
                ) 
                
                if crop_face:
                    cv2.imwrite(save_path + datetime.now().strftime("%f%m%d-%H%M%S") +".png", frame[face_center[1]-af_length:face_center[1]+af_length,face_center[0]+af_length:face_center[0]-af_length:-1])
                else:
                    cv2.imwrite(save_path + datetime.now().strftime("%f%m%d-%H%M%S") +".png", frame[:,::-1])
        cv2.imshow("camera", img[:,::-1])
        k = cv2.waitKey(1) # 1msec待つ
               
        if k == 27: # ESCキーで終了
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ### bbox scale
    scale = 1.8
    ### save whole images or cropped images
    crop_face = True
    save_path = 'images/capture'
    capture_num = 1000
    capture_camera(scale, crop_face, save_path,capture_num)

