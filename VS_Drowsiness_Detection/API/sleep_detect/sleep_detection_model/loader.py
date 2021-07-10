import dlib
import numpy as np
from .config import *

face_detector_model=None
face_landmark_model=None

def load_model():
    global face_detector_model
    global face_landmark_model

    path_predictor=face_landmark_model_path
    face_detector_model=dlib.get_frontal_face_detector()
    face_landmark_model=dlib.shape_predictor(path_predictor)

def initialize_parameters(width,height,fps):
    size = height,width
    focal_length=size[1]
    center = (size[1]/2, size[0]/2)

    K = [focal_length, 0.0, center[0],
        0.0, focal_length, center[1],
        0.0, 0.0, 1.0]

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left cornerq
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner

    ])

    cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    param_dict={
        'K':K,
        'model_points':model_points,
        'cam_matrix':cam_matrix,
        'dist_coeffs':dist_coeffs
    }
    return param_dict
