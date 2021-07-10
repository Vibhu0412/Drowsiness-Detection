from .base import *

class gaze_tracking():

  def __init__(self):
    self.fps=0
    self.EYE_AR_THRESH = 0.25
    self.SLEEP_COUNTER_THRESH = 20
    self.DROWSY_COUNTER_THRESH=20
    self.BLINK_COUNTER_THRESH=3
    self.YAWN_THRESH = 25
    self.DROWSY_COUNTER=0
    self.SLEEP_COUNTER = 0
    self.BLINK_COUNTER=0
    self.YAWN_COUNTER=0from .base import *

class gaze_tracking():

  def __init__(self):
    self.fps=0
    self.EYE_AR_THRESH = 0.25
    self.SLEEP_COUNTER_THRESH = 20
    self.DROWSY_COUNTER_THRESH=20
    self.BLINK_COUNTER_THRESH=3
    self.YAWN_THRESH = 25
    self.DROWSY_COUNTER=0
    self.SLEEP_COUNTER = 0
    self.BLINK_COUNTER=0
    self.YAWN_COUNTER=0
    self.NO_FACE_COUNTER=0
    self.NO_FACE_THRESH=5*self.fps

    self.BLINK_COUNTER_FLAG=False
    self.TOTAL_BLINK_COUNT=0
    self.YAWN_COUNTER_FLAG=False
    self.TOTAL_YAWN_COUNT=0

    self.landmarks=None
    self.blink_duration=0
    self.last_blink_frame_num=0
    self.yawn_status='no'
    self.eye_status='open'
    self.LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    self.RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
    self.GAZE_POINTS = [33, 8, 36, 45, 48,54]
    self.gaze_direction=[]
    self.blink_tracker=[]

  def eye_aspect_ratio(self,eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

  def final_ear(self):

      leftEye = np.array([(self.landmarks[point][0],self.landmarks[point][1]) for point in self.LEFT_EYE_POINTS])
      rightEye = np.array([(self.landmarks[point][0],self.landmarks[point][1]) for point in self.RIGHT_EYE_POINTS])

      leftEAR = self.eye_aspect_ratio(leftEye)
      rightEAR = self.eye_aspect_ratio(rightEye)

      ear = (leftEAR + rightEAR) / 2.0
      eye_dist=distance.euclidean(leftEye[3],rightEye[0])
      return (ear, leftEye, rightEye, eye_dist)

  def lip_distance(self):

      top_lip = np.array([(self.landmarks[point][0],self.landmarks[point][1]) for point in range(50,53)])
      top_lip_merge=np.array([(self.landmarks[point][0],self.landmarks[point][1]) for point in range(61,64)])
      top_lip = np.concatenate((top_lip, top_lip_merge))

      low_lip = np.array([(self.landmarks[point][0],self.landmarks[point][1]) for point in range(56,59)])
      low_lip_merge=np.array([(self.landmarks[point][0],self.landmarks[point][1]) for point in range(65,68)])
      low_lip = np.concatenate((low_lip, low_lip_merge))

      top_mean = np.mean(top_lip, axis=0)
      low_mean = np.mean(low_lip, axis=0)

      distance = abs(top_mean[1] - low_mean[1])
      return distance

  def get_eye_status(self,frame,c):

      eye = self.final_ear()
      ear = eye[0]
      distance = self.lip_distance()

      self.yawn_status='no'
      if (distance > self.YAWN_THRESH):
              self.YAWN_COUNTER+=1

              if self.YAWN_COUNTER>10:
                # print('Yawn Alert')
                self.yawn_status='yes'
                self.SLEEP_COUNTER=0
                self.DROWSY_COUNTER=0
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
      else:
        self.YAWN_COUNTER=0

      eye_status=None
      if ear <= self.EYE_AR_THRESH:
          # print('EAR',c,ear)
          self.BLINK_COUNTER += 1
          self.DROWSY_COUNTER += 1

          if ear<=0.22:
            self.SLEEP_COUNTER += 1
          else:
            self.SLEEP_COUNTER=0

          if self.SLEEP_COUNTER >= self.SLEEP_COUNTER_THRESH:
              # print('wake up')
              eye_status='sleepy'
              self.blink_tracker=np.append(self.blink_tracker,'sleep')
              cv2.putText(frame, "SLEEP ALERT!", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
          elif self.DROWSY_COUNTER >= self.DROWSY_COUNTER_THRESH:
              # print('drowsy alert')
              eye_status='drowsy'
              self.blink_tracker=np.append(self.blink_tracker,'drowsy')
              cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
          elif (self.BLINK_COUNTER >= self.BLINK_COUNTER_THRESH) and self.SLEEP_COUNTER>=1:
              # print('blink')
              eye_status='blinking'
              self.blink_tracker=np.append(self.blink_tracker,'blink')
              # self.BLINK_COUNTER=0
              # self.blink_duration=(c-self.last_blink)/fps
              # print(self.blink_duration)
              # self.last_blink=c
          elif self.SLEEP_COUNTER>=1:
              # print('blink')
              eye_status='blinking'
              self.blink_tracker=np.append(self.blink_tracker,'blink')
              # self.BLINK_COUNTER=0
              # self.blink_duration=(c-self.last_blink)/fps
              # print(self.blink_duration)
              # self.last_blink=c
          else:
            eye_status='open'
            self.blink_tracker=np.append(self.blink_tracker,'no_blink')


      else:
          eye_status='open'
          self.blink_tracker=np.append(self.blink_tracker,'no_blink')
          self.SLEEP_COUNTER = 0
          self.BLINK_COUNTER = 0
          self.DROWSY_COUNTER = 0

      self.eye_status=eye_status

      return [ear,distance]

  def get_head_direction(self):
    i=30
    p=(self.landmarks[i][0],self.landmarks[i][1])
    avg_ratio=[]
    for i in range(1,5):
      p1=(self.landmarks[i][0],self.landmarks[i][1])
      p2=(self.landmarks[17-i][0],self.landmarks[17-i][1])
      dist1=distance.euclidean(p1,p)
      dist2=distance.euclidean(p2,p)
      avg_ratio.append(dist1/dist2)

    if np.average(avg_ratio)<0.6:
      self.gaze_direction=np.append(self.gaze_direction,'right')
      return 'right'
    elif np.average(avg_ratio)>1.5:
      self.gaze_direction=np.append(self.gaze_direction,'left')
      return 'left'
    else:
      self.gaze_direction=np.append(self.gaze_direction,'front')
      return 'front'

  def image_preprocessing(self,new_frame,thresh):
      kernel = np.ones((3, 3), np.uint8)
      new_frame = cv2.bilateralFilter(new_frame, 10, 15, 15)
      new_frame = cv2.erode(new_frame, kernel, iterations=3)
      new_frame = cv2.threshold(new_frame, thresh, 255, cv2.THRESH_BINARY)[1]
      return new_frame


  def best_threshold(self,eye_frame,margin):

    trials={}
    for thresh in range(5, 100, 5):

      #img processing
      new_frame=self.image_preprocessing(eye_frame,thresh)

      #iris size
      new_frame=new_frame[margin:-margin,margin:-margin]
      h,w=new_frame.shape[:2]
      total_pixel=h*w
      black_pixel=total_pixel-cv2.countNonZero(new_frame)
      trials[thresh]=black_pixel/total_pixel

    iris_size=0.50
    best_thresh=min( trials.items() , key= (lambda x: abs(x[1]-iris_size) ) )
    return best_thresh

  def get_pupil_coords(self,side,gray,height, width):

    if side=='left':
      region=np.array([ (self.landmarks[point][0],self.landmarks[point][1]) for point in self.LEFT_EYE_POINTS])
    else:
      region=np.array([ (self.landmarks[point][0],self.landmarks[point][1]) for point in self.RIGHT_EYE_POINTS])

    mask = np.full((height, width), 0, np.uint8)
    cv2.polylines(mask, [region], True,255,2)
    cv2.fillPoly(mask, [region], 255)
    eye = cv2.bitwise_and(gray.copy(), gray.copy(), mask=mask)

    margin = 5
    min_x = np.min(region[:, 0]) - margin
    max_x = np.max(region[:, 0]) + margin
    min_y = np.min(region[:, 1]) - margin
    max_y = np.max(region[:, 1]) + margin

    eye_frame = eye[min_y:max_y, min_x:max_x]
    origin = (min_x, min_y)

    height, width = eye_frame.shape[:2]
    center = (width / 2, height / 2)

    threshold=self.best_threshold(eye_frame,margin)
    iris_frame = self.image_preprocessing(eye_frame,threshold[1])
    contours, _ = cv2.findContours(iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    contours = sorted(contours, key=cv2.contourArea)

    x=0
    y=0
    try:
        moments = cv2.moments(contours[-1])
        x = int(moments['m10'] / moments['m00'])
        y = int(moments['m01'] / moments['m00'])
    except (IndexError, ZeroDivisionError):
        x=0
        y=0

    return (x,y), origin, center

  def get_pupil_location(self,pupil_left_coords,center_left,pupil_right_coords,center_right):
    pupil_left = pupil_left_coords[0] / (center_left[0] * 2 - 10)
    pupil_right = pupil_right_coords[0] / (center_right[0] * 2 - 10)
    horizontal_ratio = (pupil_left + pupil_right) / 2
    if horizontal_ratio<0.55 and horizontal_ratio>0.45:
      return 'center'
    elif horizontal_ratio <= 0.55:
      return 'left'
    elif horizontal_ratio >= 0.45:
      return 'right'

  def mark_pupil(self,img,origin_left,origin_right,pupil_left_coords,pupil_right_coords):
    pupil_left=(int(origin_left[0]+pupil_left_coords[0]), int(origin_left[1]+pupil_left_coords[1]) )
    pupil_right=(int(origin_right[0]+pupil_right_coords[0]), int(origin_right[1]+pupil_right_coords[1]) )
    img=cv2.circle(img,pupil_left,2,(0,255,0),2)
    img=cv2.circle(img,pupil_right,2,(0,255,0),2)
    return img


  def get_gaze(self,model_points,cam_matrix,dist_coeffs):

    image_points = []
    for n in self.GAZE_POINTS:
        x = self.landmarks[n][0]
        y = self.landmarks[n][1]
        image_points += [(x, y)]

    image_points = np.array(image_points, dtype="double")

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                  cam_matrix, dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                      translation_vector,
                                                      cam_matrix, dist_coeffs)

    return nose_end_point2D,image_points,rotation_vector,translation_vector

  def plot_gaze(self,img,image_points,nose_end_point2D):

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    cv2.line(img, p1, p2, (255, 0, 0), 2)
    return img

  def get_head_orientation(self,model_points, rotation_vector, translation_vector, cam_matrix, dist_coeffs):
      modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, cam_matrix, dist_coeffs)
      rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
      proj_matrix = np.hstack((rvec_matrix, translation_vector))
      eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
      pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
      pitch = math.degrees(math.asin(math.sin(pitch)))
      roll = -math.degrees(math.asin(math.sin(roll)))
      yaw = math.degrees(math.asin(math.sin(yaw)))
      return pitch,roll,yaw

  def gaze_ball_detection(self):
      fps=self.fps
      status='active'
      if len(self.gaze_direction)==15*fps:
        gaze_left=len(self.gaze_direction[self.gaze_direction=='left'])
        gaze_right=len(self.gaze_direction[self.gaze_direction=='right'])
        gaze_front=len(self.gaze_direction[self.gaze_direction=='front'])
        if gaze_left==15*fps or gaze_right==15*fps:
           status='looking_left_or_right'
        elif gaze_front==15*fps:
           no_blink=len(self.blink_tracker[self.blink_tracker=='no_blink'])
           if no_blink==15*fps:
              status='alert'
        else:
           status='active'

        self.gaze_direction=np.delete(self.gaze_direction,0)
        self.blink_tracker=np.delete(self.blink_tracker,0)

      else:
        status='active'

      return status  
    self.NO_FACE_COUNTER=0
    self.NO_FACE_THRESH=5*self.fps

    self.BLINK_COUNTER_FLAG=False
    self.TOTAL_BLINK_COUNT=0
    self.YAWN_COUNTER_FLAG=False
    self.TOTAL_YAWN_COUNT=0

    self.landmarks=None
    self.blink_duration=0
    self.last_blink_frame_num=0
    self.yawn_status='no'
    self.eye_status='open'
    self.LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    self.RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
    self.GAZE_POINTS = [33, 8, 36, 45, 48,54]
    self.gaze_direction=[]
    self.blink_tracker=[]

  def eye_aspect_ratio(self,eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

  def final_ear(self):

      leftEye = np.array([(self.landmarks[point][0],self.landmarks[point][1]) for point in self.LEFT_EYE_POINTS])
      rightEye = np.array([(self.landmarks[point][0],self.landmarks[point][1]) for point in self.RIGHT_EYE_POINTS])

      leftEAR = self.eye_aspect_ratio(leftEye)
      rightEAR = self.eye_aspect_ratio(rightEye)

      ear = (leftEAR + rightEAR) / 2.0
      eye_dist=distance.euclidean(leftEye[3],rightEye[0])
      return (ear, leftEye, rightEye, eye_dist)

  def lip_distance(self):

      top_lip = np.array([(self.landmarks[point][0],self.landmarks[point][1]) for point in range(50,53)])
      top_lip_merge=np.array([(self.landmarks[point][0],self.landmarks[point][1]) for point in range(61,64)])
      top_lip = np.concatenate((top_lip, top_lip_merge))

      low_lip = np.array([(self.landmarks[point][0],self.landmarks[point][1]) for point in range(56,59)])
      low_lip_merge=np.array([(self.landmarks[point][0],self.landmarks[point][1]) for point in range(65,68)])
      low_lip = np.concatenate((low_lip, low_lip_merge))

      top_mean = np.mean(top_lip, axis=0)
      low_mean = np.mean(low_lip, axis=0)

      distance = abs(top_mean[1] - low_mean[1])
      return distance

  def get_eye_status(self,frame,c):

      eye = self.final_ear()
      ear = eye[0]
      distance = self.lip_distance()

      self.yawn_status='no'
      if (distance > self.YAWN_THRESH):
              self.YAWN_COUNTER+=1

              if self.YAWN_COUNTER>10:
                # print('Yawn Alert')
                self.yawn_status='yes'
                self.SLEEP_COUNTER=0
                self.DROWSY_COUNTER=0
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
      else:
        self.YAWN_COUNTER=0

      eye_status=None
      if ear <= self.EYE_AR_THRESH:
          # print('EAR',c,ear)
          self.BLINK_COUNTER += 1
          self.DROWSY_COUNTER += 1

          if ear<=0.22:
            self.SLEEP_COUNTER += 1
          else:
            self.SLEEP_COUNTER=0

          if self.SLEEP_COUNTER >= self.SLEEP_COUNTER_THRESH:
              # print('wake up')
              eye_status='sleepy'
              self.blink_tracker=np.append(self.blink_tracker,'sleep')
              cv2.putText(frame, "SLEEP ALERT!", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
          elif self.DROWSY_COUNTER >= self.DROWSY_COUNTER_THRESH:
              # print('drowsy alert')
              eye_status='drowsy'
              self.blink_tracker=np.append(self.blink_tracker,'drowsy')
              cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
          elif (self.BLINK_COUNTER >= self.BLINK_COUNTER_THRESH) and self.SLEEP_COUNTER>=1:
              # print('blink')
              eye_status='blinking'
              self.blink_tracker=np.append(self.blink_tracker,'blink')
              # self.BLINK_COUNTER=0
              # self.blink_duration=(c-self.last_blink)/fps
              # print(self.blink_duration)
              # self.last_blink=c
          elif self.SLEEP_COUNTER>=1:
              # print('blink')
              eye_status='blinking'
              self.blink_tracker=np.append(self.blink_tracker,'blink')
              # self.BLINK_COUNTER=0
              # self.blink_duration=(c-self.last_blink)/fps
              # print(self.blink_duration)
              # self.last_blink=c
          else:
            eye_status='open'
            self.blink_tracker=np.append(self.blink_tracker,'no_blink')


      else:
          eye_status='open'
          self.blink_tracker=np.append(self.blink_tracker,'no_blink')
          self.SLEEP_COUNTER = 0
          self.BLINK_COUNTER = 0
          self.DROWSY_COUNTER = 0

      self.eye_status=eye_status

      return [ear,distance]

  def get_head_direction(self):
    i=30
    p=(self.landmarks[i][0],self.landmarks[i][1])
    avg_ratio=[]
    for i in range(1,5):
      p1=(self.landmarks[i][0],self.landmarks[i][1])
      p2=(self.landmarks[17-i][0],self.landmarks[17-i][1])
      dist1=distance.euclidean(p1,p)
      dist2=distance.euclidean(p2,p)
      avg_ratio.append(dist1/dist2)

    if np.average(avg_ratio)<0.6:
      self.gaze_direction=np.append(self.gaze_direction,'right')
      return 'right'
    elif np.average(avg_ratio)>1.5:
      self.gaze_direction=np.append(self.gaze_direction,'left')
      return 'left'
    else:
      self.gaze_direction=np.append(self.gaze_direction,'front')
      return 'front'

  def image_preprocessing(self,new_frame,thresh):
      kernel = np.ones((3, 3), np.uint8)
      new_frame = cv2.bilateralFilter(new_frame, 10, 15, 15)
      new_frame = cv2.erode(new_frame, kernel, iterations=3)
      new_frame = cv2.threshold(new_frame, thresh, 255, cv2.THRESH_BINARY)[1]
      return new_frame


  def best_threshold(self,eye_frame,margin):

    trials={}
    for thresh in range(5, 100, 5):

      #img processing
      new_frame=self.image_preprocessing(eye_frame,thresh)

      #iris size
      new_frame=new_frame[margin:-margin,margin:-margin]
      h,w=new_frame.shape[:2]
      total_pixel=h*w
      black_pixel=total_pixel-cv2.countNonZero(new_frame)
      trials[thresh]=black_pixel/total_pixel

    iris_size=0.50
    best_thresh=min( trials.items() , key= (lambda x: abs(x[1]-iris_size) ) )
    return best_thresh

  def get_pupil_coords(self,side,gray,height, width):

    if side=='left':
      region=np.array([ (self.landmarks[point][0],self.landmarks[point][1]) for point in self.LEFT_EYE_POINTS])
    else:
      region=np.array([ (self.landmarks[point][0],self.landmarks[point][1]) for point in self.RIGHT_EYE_POINTS])

    mask = np.full((height, width), 0, np.uint8)
    cv2.polylines(mask, [region], True,255,2)
    cv2.fillPoly(mask, [region], 255)
    eye = cv2.bitwise_and(gray.copy(), gray.copy(), mask=mask)

    margin = 5
    min_x = np.min(region[:, 0]) - margin
    max_x = np.max(region[:, 0]) + margin
    min_y = np.min(region[:, 1]) - margin
    max_y = np.max(region[:, 1]) + margin

    eye_frame = eye[min_y:max_y, min_x:max_x]
    origin = (min_x, min_y)

    height, width = eye_frame.shape[:2]
    center = (width / 2, height / 2)

    threshold=self.best_threshold(eye_frame,margin)
    iris_frame = self.image_preprocessing(eye_frame,threshold[1])
    contours, _ = cv2.findContours(iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    contours = sorted(contours, key=cv2.contourArea)

    x=0
    y=0
    try:
        moments = cv2.moments(contours[-1])
        x = int(moments['m10'] / moments['m00'])
        y = int(moments['m01'] / moments['m00'])
    except (IndexError, ZeroDivisionError):
        x=0
        y=0

    return (x,y), origin, center

  def get_pupil_location(self,pupil_left_coords,center_left,pupil_right_coords,center_right):
    pupil_left = pupil_left_coords[0] / (center_left[0] * 2 - 10)
    pupil_right = pupil_right_coords[0] / (center_right[0] * 2 - 10)
    horizontal_ratio = (pupil_left + pupil_right) / 2
    if horizontal_ratio<0.55 and horizontal_ratio>0.45:
      return 'center'
    elif horizontal_ratio <= 0.55:
      return 'left'
    elif horizontal_ratio >= 0.45:
      return 'right'

  def mark_pupil(self,img,origin_left,origin_right,pupil_left_coords,pupil_right_coords):
    pupil_left=(int(origin_left[0]+pupil_left_coords[0]), int(origin_left[1]+pupil_left_coords[1]) )
    pupil_right=(int(origin_right[0]+pupil_right_coords[0]), int(origin_right[1]+pupil_right_coords[1]) )
    img=cv2.circle(img,pupil_left,2,(0,255,0),2)
    img=cv2.circle(img,pupil_right,2,(0,255,0),2)
    return img


  def get_gaze(self,model_points,cam_matrix,dist_coeffs):

    image_points = []
    for n in self.GAZE_POINTS:
        x = self.landmarks[n][0]
        y = self.landmarks[n][1]
        image_points += [(x, y)]

    image_points = np.array(image_points, dtype="double")

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                  cam_matrix, dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                      translation_vector,
                                                      cam_matrix, dist_coeffs)

    return nose_end_point2D,image_points,rotation_vector,translation_vector

  def plot_gaze(self,img,image_points,nose_end_point2D):

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    cv2.line(img, p1, p2, (255, 0, 0), 2)
    return img

  def get_head_orientation(self,model_points, rotation_vector, translation_vector, cam_matrix, dist_coeffs):
      modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, cam_matrix, dist_coeffs)
      rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
      proj_matrix = np.hstack((rvec_matrix, translation_vector))
      eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
      pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
      pitch = math.degrees(math.asin(math.sin(pitch)))
      roll = -math.degrees(math.asin(math.sin(roll)))
      yaw = math.degrees(math.asin(math.sin(yaw)))
      return pitch,roll,yaw

  def gaze_ball_detection(self):
      fps=self.fps
      status='active'
      if len(self.gaze_direction)==15*fps:
        gaze_left=len(self.gaze_direction[self.gaze_direction=='left'])
        gaze_right=len(self.gaze_direction[self.gaze_direction=='right'])
        gaze_front=len(self.gaze_direction[self.gaze_direction=='front'])
        if gaze_left==15*fps or gaze_right==15*fps:
           status='looking_left_or_right'
        elif gaze_front==15*fps:
           no_blink=len(self.blink_tracker[self.blink_tracker=='no_blink'])
           if no_blink==15*fps:
              status='alert'
        else:
           status='active'

        self.gaze_direction=np.delete(self.gaze_direction,0)
        self.blink_tracker=np.delete(self.blink_tracker,0)

      else:
        status='active'

      return status
