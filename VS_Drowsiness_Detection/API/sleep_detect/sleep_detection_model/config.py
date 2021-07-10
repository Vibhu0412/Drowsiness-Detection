import os
BASE_DIR=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
print('bd: ',BASE_DIR)
face_landmark_model_path=f'{BASE_DIR}/sleep_detect/sleep_detection_model/models_files/shape_predictor_68_face_landmarks.dat'

color_black=(0,0,0)
color_white=(255,255,255)

color_frame_text=color_white
color_frame_background=color_black

color_cream=color_black #(255, 180, 132)#(42, 27, 61) #(255,91,2)
color_light_orange=color_black #bg(0,110,187)

color_gaze_outer_box=(195, 141, 147)
color_gaze_inner_box=color_white
gaze_box_header=color_black

side_panel_text=color_white
side_panel_value=color_white

color_box_heading=color_white
color_box_number=color_white#(226, 125, 96)

color_direction_line=(226, 125, 96)
color_alert=(255,0,0)
# color_frame_text=(245, 230, 204)
# color_frame_background=(195, 141, 147)

# color_cream=(245, 230, 204) #(255, 180, 132)#(42, 27, 61) #(255,91,2)
# color_light_orange=(255, 180, 132) #bg(0,110,187)

# color_black=(0,0,0)
# color_white=(255,255,255)

# color_gaze_outer_box=(195, 141, 147)
# color_gaze_inner_box=(255,255,255)
# gaze_box_header=(245, 230, 204)

# side_panel_text=(154, 23, 80)
# side_panel_value=(226, 125, 96)

# color_box_heading=(154, 23, 80)
# color_box_number= (226, 125, 96)#(226, 125, 96)

# color_direction_line=(226, 125, 96)
# color_alert=(255,0,0)
