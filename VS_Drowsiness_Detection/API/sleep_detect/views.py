from django.shortcuts import render
from django.http import HttpResponse
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import cv2
import threading

# Create your views here.
def home(request):
    context = {}
    return render(request,'sleep_detect/drowsiness.html', context)

# CUSTOM WEBCAM ORIGINAL
# @gzip.gzip_page
# def WebCam(request):
#
#     try:
#         cam = VideoCamera()
#         return StreamingHttpResponse(gen(cam),content_type="multipart/x-mixed-replace;boundary=frame")
#     except Exception as e:
#         raise
#
#     context = {}
#     return render(request,'sleep_detect/test.html', context)
#
# # To capture video class
# class VideoCamera(object):
#     """docstring forVideoCamera."""
#
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#         (self.grabbed, self.frame) = self.video.read()
#         threading.Thread(target=self.update,args=()).start()
#
#     def __del__(self):
#         self.video.release()
#
#     def get_frame(self):
#         image = self.frame
#         _,jpeg = cv2.imencode('.jpg',image)
#         return jpeg.tobytes()
#
#     def update(self):
#         while True:
#             (self.grabbed, self.frame) = self.video.read()
#
# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         yield(b'--frame\r\n'
#               b'Content-Type: image /jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def webcam_second(request):
    capture = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    videoWriter = cv2.VideoWriter('/vibhu/test_video/video.avi', fourcc, 30.0, (640,480))

    while (True):

        ret, frame = capture.read()

        if ret:
            cv2.imshow('video', frame)
            videoWriter.write(frame)

        if cv2.waitKey(1) == 27:
            break

    capture.release()
    videoWriter.release()

    cv2.destroyAllWindows()

    context = {}
    return render(request, 'sleep_detect/drowsiness.html', context)
