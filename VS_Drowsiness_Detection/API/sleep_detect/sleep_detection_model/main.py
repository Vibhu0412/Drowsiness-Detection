from .base import *
from .helper import gaze_tracking
from .config import *
from . import loader
from PIL import ImageFont, ImageDraw, Image

loader.load_model()

BASE_DIR='videos/'

def get_angle(org_points):
  a=np.array([org_points[1][0],-org_points[1][1]])
  b=np.array([org_points[0][0],-org_points[0][1]])
  c=np.array([b[0]-10,b[1]])
  ab = a - b
  bc = c - b
  cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
  angle = np.degrees(np.arccos(cosine_angle))
  return angle

def transform_points(origin_point,angle,magnitude):
  x=origin_point[0]
  y=origin_point[1]
  width=-magnitude*np.cos(angle*np.pi/180)
  height=magnitude*np.sin(angle*np.pi/180)
  new_x=x+width
  new_y=y+height
  return (new_x,new_y)

def adding_right_bar(size,data_dict,c):

    font_type='arial.ttf'
    width=size[0]//3
    height1=(size[1]//8)
    height2=size[1]-height1-width
    height3=width

    img=np.zeros((size[1],width,3),np.uint8)
    '''frame rect'''
    img=cv2.rectangle(img,(0,0),(width,height1),color_frame_background,-1)
    '''text rect'''
    img=cv2.rectangle(img,(0,height1),(width,height1+height2),color_cream,-1)
    '''gaze box rect'''
    img=cv2.rectangle(img,(0,height1+height2-15),(width,height1+height2+height3),color_gaze_outer_box,-1)
    img=cv2.rectangle(img,(0+10,height1+height2+10),(width-10,height1+height2+height3-10),color_gaze_inner_box,-1)
    gaze_box_x1,gaze_box_y1=0+10,height1+height2+10
    gaze_box_x2,gaze_box_y2=width-10,height1+height2+height3-10


    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    th=30
    temp_width=(width//2)+5
    # font = ImageFont.truetype('/usr/share/fonts/truetype/roboto/hinted/Roboto-Black.ttf', 20)
    font = ImageFont.truetype(f'{font_type}', 20)

    text1=f'Frame: '
    draw.text((10,10),text1,fill=color_frame_text,font=font, align='center') #,0.8,(255,255,255),2)
    text2=f'{c}'
    draw.text((temp_width,10),text2,fill=color_frame_text,font=font, align='center')

    th=20
    font = ImageFont.truetype(f'{font_type}', 15)
    font1 = ImageFont.truetype(f'{font_type}', 17)

    gaze_header_font = ImageFont.truetype(f'{font_type}', 15)

    for key,values in data_dict.items():
        if key=='gaze_line_points':
          horizontal_line=[(width//2,gaze_box_y1),(width//2,gaze_box_y2)]
          vertical_line=[(gaze_box_x1,height1+height2+(width//2)),(gaze_box_x2,height1+height2+(width//2))]
          draw.line(horizontal_line,fill=color_black,width=1)
          draw.line(vertical_line,fill=color_black,width=1)
          draw.text((gaze_box_x1,height1+height2-12),"          Gaze Direction",font=gaze_header_font,align='left',fill=gaze_box_header)

          if values[0]!=None:
            angle=get_angle(values)
            origin_point=(width//2,height1+height2+(height3//2))
            magnitude=width//2
            end_points=transform_points(origin_point,angle,magnitude)
            line_points=[origin_point,end_points]
            draw.line(line_points,fill=color_direction_line,width=2)
          continue

        if key=='head_direction' or key=='face_status' or key=='blink_duration' or key=='pupil_points' or key=='gaze_line_points' or key=='head_orientation' or key=='eye_lip_ratio':
          continue

        if data_dict['face_status'][0]=='alert':
           text1='Unusual Movements Detected!'
           draw.text((5,height1+th),text1,font =font,align='left',fill=color_alert)
        else:
          if key=='yawn_status':
              text1 ='Yawn_status:'
              draw.text((5,height1+th),text1,font =font,align='left',fill=side_panel_text)

              if values[0]==None:
                values[0]=''

              text2=f'{values[0]}'
              if text2.lower()=='yes':
                draw.text((temp_width,height1+th),'Yes',font =font,align='right',fill=color_alert)
              else:
                draw.text((temp_width,height1+th),'No',font =font,align='right',fill=side_panel_value)


          if key=='eye_status':
              text1 ='Eye_status:'
              draw.text((5,height1+th),text1,font =font,align='left',fill=side_panel_text)

              if values[0]==None:
                continue

              text2=f'{values[0]}'
              yawn_status=data_dict['yawn_status'][0]
              print(text2,yawn_status)
              if text2.lower()=='drowsy' and yawn_status!='yes':
                draw.text((temp_width,height1+th),'Drowsy eyes',font =font1,align='right',fill=color_alert)
              elif text2.lower()=='sleepy' and yawn_status!='yes':
                draw.text((temp_width,height1+th),'!!Sleeping!!',font =font1,align='right',fill=color_alert)
              elif yawn_status=='yes':
                text2='Drowsy eyes'
                draw.text((temp_width,height1+th),text2,font =font,align='right',fill=side_panel_value)
              else:
                draw.text((temp_width,height1+th),text2,font =font,align='right',fill=side_panel_value)


          # if key=='gaze_ball_status':
          #     text1 ='Eye_ball_status:'
          #     draw.text((5,height1+th+15),text1,font =font,align='left',fill=side_panel_text)

          #     if values[0]==None:
          #       values[0]=''

          #     text2=f'{values[0]}'
          #     if text2.lower()=='alert':
          #       draw.text((5,height1+th+30),'!!Lost!!',font =font,align='right',fill=color_alert)
          #     elif text2.lower()=='active':
          #       draw.text((5,height1+th+30),text2,font =font,align='right',fill=side_panel_value)
          #     else:
          #       draw.text((5,height1+th+30),'Please! look at the road',font =font,align='right',fill=color_alert)
          th=th+25


    img = np.array(img_pil)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def adding_bottom_bar(frame,data_dict,c):
    # print(data_dict['face_status'][0])
    font_type='arial.ttf'
    loop_key={
        0:['eye_lip_ratio',0,'   Eye_ratio'],
        1:['eye_lip_ratio',1,' Mouth_ratio'],
        2:['blink_duration',0,'Blink_duration'],
        3:['head_orientation',0,'Yaw'],
        4:['head_orientation',1,'Roll'],
        5:['head_orientation',2,'Pitch'],
    }

    w,h=frame.shape[1],frame.shape[0]
    box_height=h//4
    box_width=w//6

    img=np.zeros((box_height,w,3),np.uint8)

    inner_rect_th=10

    for i in range(3):
      x1,y1=i*box_width,0,
      x2,y2=x1+box_width,box_height
      img=cv2.rectangle(img,(x1,y1),(x2,y2),color_light_orange,-1)
      x1,y1=x1+inner_rect_th,inner_rect_th
      x2,y2=x2-inner_rect_th,y2-inner_rect_th
      img=cv2.rectangle(img,(x1,y1),(x2,y2),color_cream,-1)

    for i in range(3,6):
      x1,y1=i*box_width,0,
      x2,y2=x1+box_width,box_height
      img=cv2.rectangle(img,(x1,y1),(x2,y2),color_light_orange,-1)
      x1,y1=x1+inner_rect_th,30
      x2,y2=x2-inner_rect_th,y2-inner_rect_th
      img=cv2.rectangle(img,(x1,y1),(x2,y2),color_cream,-1)

    x1,y1=3*box_width+inner_rect_th,inner_rect_th
    x2,y2=6*box_width-inner_rect_th,25
    img=cv2.rectangle(img,(x1,y1),(x2,y2),color_cream,-1)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    font1 = ImageFont.truetype(f'{font_type}', 15)
    font2 = ImageFont.truetype(f'{font_type}', 20)

    for i in range(3):
      x1,y1=i*box_width+inner_rect_th,inner_rect_th

      y1=y1+15
      text1=f'{loop_key[i][2]}'
      draw.text((x1+3,y1),text1,font=font1,align='center',fill=color_box_heading)

      values=data_dict[loop_key[i][0]]
      if data_dict['face_status'][0]=='yes':
        # print(values)
        values=values[loop_key[i][1]]
        text2=f'{values:.2f}'
        draw.text((x1+10,y1+25),text2,font=font2,align='center',fill=color_box_number)

    font1 = ImageFont.truetype(f'{font_type}', 15)
    font2 = ImageFont.truetype(f'{font_type}', 20)
    font3 = ImageFont.truetype(f'{font_type}', 20)

    x1,y1=3*box_width+inner_rect_th,inner_rect_th
    if data_dict['face_status'][0]=='yes':
      text=data_dict['head_direction'][0]
    else:
      text=''
    draw.text((x1+(box_width//2),y1-1),'Head Orientation:',font=font3,align='center',fill=color_box_heading)
    draw.text((x1+(box_width*2)-10,y1-1),text,font=font3,align='center',fill=color_box_number)

    for i in range(3,6):
      x1,y1=i*box_width+inner_rect_th,30
      # x2,y2=x1+box_width-inner_rect_th,box_height-inner_rect_th
      text1=f'{loop_key[i][2]}'
      y1=y1+15
      draw.text((x1+23,y1),text1,font=font1,align='center',fill=color_box_heading)

      values=data_dict[loop_key[i][0]]
      if data_dict['face_status'][0]=='yes':
        values=values[loop_key[i][1]]
        text2=f'{values:.2f}'
        draw.text((x1+15,y1+25),text2,font=font2,align='center',fill=color_box_number)

    img = np.array(img_pil)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def adding_bar(frame,size,data_dict,frame_num):

    h_img=adding_right_bar(size,data_dict,frame_num)
    h_frame=np.hstack([frame,h_img])
    v_img=adding_bottom_bar(h_frame,data_dict,frame_num)
    img=np.vstack([h_frame,v_img])

    return img

def inference(frame,frame_num,gaze,params,width,height):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = loader.face_detector_model(gray, 0)

    if len(rects)!=0:
      face_status='yes'
      gaze.NO_FACE_COUNTER=0
      landmarks = loader.face_landmark_model(gray, rects[0])
      # print(landmarks.part(3).x,landmarks.part(3).y)
      arr=[]
      for pt in range(68):
        arr.append([landmarks.part(pt).x,landmarks.part(pt).y])
      arr=np.array(arr,dtype='int16')

      gaze.landmarks=np.array(arr,dtype='int16')

      del landmarks,arr,rects
      gc.collect();

      head_pose = gaze.get_head_direction()

      # pupil_left_coords,origin_left,center_left = gaze.get_pupil_coords('left',gray,height, width)
      # pupil_right_coords,origin_right,center_right = gaze.get_pupil_coords('right',gray, height, width)
      # pupil_loc = gaze.get_pupil_location(pupil_left_coords,center_left,pupil_right_coords,center_right)
      # frame=gaze.mark_pupil(frame,origin_left,origin_right,pupil_left_coords,pupil_right_coords)

      nose_end_point2D,image_points,rotation_vector,translation_vector=gaze.get_gaze(params['model_points'],params['cam_matrix'],params['dist_coeffs'])
      # frame=gaze.plot_gaze(frame,image_points,nose_end_point2D)
      pitch,roll,yaw=gaze.get_head_orientation(params['model_points'],rotation_vector,translation_vector,params['cam_matrix'],params['dist_coeffs'])

      ear_ratio,lip_distance=gaze.get_eye_status(frame,frame_num)
      # status=gaze.gaze_ball_detection()
      p1 = [int(image_points[0][0]), int(image_points[0][1])]
      p2 = [int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])]

      if gaze.eye_status=='blinking' and gaze.BLINK_COUNTER_FLAG==False:
        gaze.BLINK_COUNTER_FLAG=True
        gaze.TOTAL_BLINK_COUNT+=1
        gaze.blink_duration=(frame_num/gaze.fps)-(gaze.last_blink_frame_num/gaze.fps)
        gaze.last_blink_frame_num=frame_num

      if gaze.eye_status!='blinking':
        gaze.BLINK_COUNTER_FLAG=False

      if gaze.yawn_status=='yes' and gaze.YAWN_COUNTER_FLAG==False:
        gaze.YAWN_COUNTER_FLAG=True
        gaze.TOTAL_YAWN_COUNT+=1
      if gaze.yawn_status!='yes':
        gaze.YAWN_COUNTER_FLAG=False

    else:
      p1,p2,head_pose,pitch,roll,yaw,ear_ratio,lip_distance=[None for i in range(8)]
      gaze.NO_FACE_COUNTER+=1
      face_status='no'
      if gaze.NO_FACE_COUNTER>=gaze.NO_FACE_THRESH:
        face_status='alert'
        for i in range(gaze.NO_FACE_THRESH):
          gaze.gaze_direction=np.append(gaze.gaze_direction,'no_face')
          gaze.gaze_direction=np.delete(gaze.gaze_direction,0)
          gaze.blink_tracker=np.append(gaze.blink_tracker,'no_face')
          gaze.blink_tracker=np.delete(gaze.blink_tracker,0)

    size=[width,height]
    data_dict={
        'face_status':[face_status],
        'gaze_line_points':[p1,p2],
        'pupil_points':[],
        'blink_duration':[gaze.blink_duration],
        'head_direction':[head_pose],
        'head_orientation':[pitch,roll,yaw],
        'yawn_status':[gaze.yawn_status],
        'eye_status':[gaze.eye_status],
        'blink_counter':[gaze.TOTAL_BLINK_COUNT],
        'yawn_counter':[gaze.TOTAL_YAWN_COUNT],
        'eye_lip_ratio':[ear_ratio,lip_distance],
    }
    frame=adding_bar(frame,size,data_dict,frame_num)
    return frame
