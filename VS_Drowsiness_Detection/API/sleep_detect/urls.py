from django.urls import path
from . import views

urlpatterns=[
    path('',views.home, name = "home"),
    path('open_webcam/', views.WebCam, name = "webcam"),
    path('drowsiness_detect/',views.DrowsinessDetect,name="drowsiness_detect"),
    path('under_construction/',views.UnderConstruction,name="under_construction"),
    # path('webcam/', views.webcam_second, name = "webcam")
]
