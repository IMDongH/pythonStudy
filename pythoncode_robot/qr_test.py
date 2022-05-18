import os
import time
import urllib.request
import cv2
import numpy as np  # pip install numpy
from pyzbar.pyzbar import decode  # pip install pyzbar

capture = cv2.VideoCapture(0, cv2.CAP_V4L)
# capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

second = 5

while True:
    url_links = []
    success, img = capture.read()

    for barcode in decode(img):
        # print(barcode.data)
        myUrl = barcode.data.decode('utf-8')
        print(myUrl)
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 0, 255), 5)
        url_links.append(myUrl)

        set(url_links)
        # print(url_links)
        # url접속
        if url_links:
            web_url = urllib.request.urlopen(url_links[0])
            print("result code: " + str(web_url.getcode()))
            time.sleep(second)
            os.system("python cp_test.py")
            # url_links 비어있지않는 경우, 체크포인트 스크립트 실행

    cv2.imshow('Result', img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
capture.release()
cv2.destroyAllWindows()
