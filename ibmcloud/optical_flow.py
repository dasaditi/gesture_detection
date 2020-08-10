import json
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import scipy
np.random.seed(251)

project_path = '/root/w251-data-local/data/'
dir_openpose = 'processed'
dir_video = 'videos'
dir_json = 'json'

exclusion_list = ['ASL_2008_01_11_scene9-camera1_AGAIN_v2_c1_train.mov',
                  'ASL_2008_05_12a_scene8-camera1_AGAIN_v1_c3_test.mov',
                  'ASL_2008_08_04_scene7-camera1_AGAIN_v2_c5_train.mov',
                  'ASL_2011_06_14_Brady_scene2-camera1_AGAIN_v3_c7_train.mov',
                  'ASL_2008_08_04_scene45-camera1_CHAT_v1_c4_train.mov',
                  'ASL_2008_01_11_scene81-camera1_DRESS_v1_c0_train.mov',
                  'ASL_2008_01_18_scene23-camera1_DRESS_v1_c1_train.mov',
                  'ASL_2008_01_18_scene24-camera1_DRESS_v1_c3_test.mov',
                  'ASL_2008_05_12a_scene48-camera1_DRESS_v2_c6_train.mov',
                  'ASL_2008_05_29a_scene4-camera1_DRESS_v2_c9_train.mov',
                  'ASL_2008_08_04_scene50-camera1_DRESS_v1_c11_train.mov',
                  'ASL_2008_08_06_scene24-camera1_DRESS_v1_c12_train.mov',
                  'ASL_2011_06_08_Brady_scene28-camera1_DRESS_v1_c14_train.mov',
                  'ASL_2011_06_14_Brady_scene7-camera1_DRESS_v1_c15_train.mov']

class_list = ['AGAIN', 'ALL', 'AWKWARD', 'BASEBALL', 'BEHAVIOR', 'CAN', 'CHAT', 'CHEAP',
              'CHEAT', 'CHURCH', 'COAT', 'CONFLICT', 'COURT', 'DEPOSIT', 'DEPRESS',
              'DOCTOR', 'DRESS', 'ENOUGH', 'NEG']

body_feature_set = [3, 4, 6, 7]
hand_feature_set = list(range(21))
hand_feature_set1 = [0, 17, 13, 9, 5, 1]
hand_feature_set2 = [0, 18, 14, 10, 6, 2]
hand_feature_set3 = [0, 19, 15, 11, 7, 3]
hand_feature_set4 = [0, 20, 16, 12, 8, 4]

# Create some random colors
color_arr = np.random.randint(0,255,(300,3))
color_red = [255, 0, 0]
color_green = [0, 255, 0]
color_blue = [0, 0, 255]
color_yellow = [255, 255, 0]
color_pink = [255, 0, 255]
color_teal = [0, 255, 255]
color_white = [255, 255, 255]


from scipy.stats import multivariate_normal
# The selection is var_arr is judgemental
#   The goal is to get a good spread of prob_center_lst
var_arr = np.concatenate([np.arange(0.01, 0.1, 0.01), np.round(np.exp(np.arange(0.1, 3, 0.2))-1, 1)])
prob_center_lst = []
for var in var_arr:
    prob_corner = multivariate_normal.cdf([0.5,-0.5], mean=[0,0], cov=[[var,0],[0,var]])
    prob_center = 1 - prob_corner*4
    prob_center_lst.append(prob_center)

# print(prob_center_lst[:10])
# print(prob_center_lst[-10:])

prob_center_arr = np.array(prob_center_lst)

def draw_optical_flow(outdir_opt_flow, outdir_transfer, in_selected_f_dict, sim_num=0, overlay_order_num=0):
    # create directories
    if not os.path.exists(os.path.join(project_path, dir_openpose, outdir_opt_flow)):
        os.makedirs(os.path.join(project_path, dir_openpose, outdir_opt_flow))
    if not os.path.exists(os.path.join(project_path, dir_openpose, outdir_transfer)):
        os.makedirs(os.path.join(project_path, dir_openpose, outdir_transfer))
        for cl in class_list:
            os.makedirs(os.path.join(project_path, dir_openpose, outdir_transfer, cl))
            
    # Remove all the files in dir_optical_flow
    files = glob.glob(os.path.join(project_path, dir_openpose, outdir_opt_flow, '*'))
    for f in files:
        os.remove(f)
    # Remove all the files in the dir_transfer folder
    for i in range(len(class_list)):
        files = glob.glob(os.path.join(project_path, dir_openpose, outdir_transfer, class_list[i], '*'))
        for f in files:
            os.remove(f)
    
    mov_file_lst = [f for f in os.listdir((os.path.join(project_path, dir_openpose, dir_video)))]
    mov_len = len(mov_file_lst)
    i = 0
    for mov_file in mov_file_lst:
        i = i + 1
        if mov_file in exclusion_list:
            continue
        print("Processing {} of {} => {}".format(i, mov_len, mov_file))
        json_files_lst = [f for f in os.listdir(os.path.join(project_path, dir_openpose, dir_json)) 
                          if os.path.splitext(mov_file)[0] in f]
        if len(json_files_lst) > 0:
            print('Found the json files')
        else:
            break
            
        num_of_frame = len(json_files_lst)
        color_step = (255 - COLOR_MIN) / num_of_frame
        
        video_feature_dict = dict()
        
        for json_f in json_files_lst:
            with open(os.path.join(project_path, dir_openpose, dir_json, json_f)) as ff:
                json_code = json.load(ff)
            
            # This assume there is only one person
            body_raw_lst = json_code['people'][0]['pose_keypoints_2d']
            left_hand_raw_lst = json_code['people'][0]['hand_left_keypoints_2d']
            right_hand_raw_lst = json_code['people'][0]['hand_right_keypoints_2d']
            
            for feat in list(in_selected_f_dict.keys()):
                feat_num = int(feat.split('_')[1])
                feat_value =  video_feature_dict.get(feat, [])
                if 'body' in feat:
                    feat_value.append(body_raw_lst[3*feat_num:3*(feat_num+1)])
                    video_feature_dict[feat] = feat_value
                elif 'lefthand' in feat:
                    feat_value.append(left_hand_raw_lst[3*feat_num:3*(feat_num+1)])
                    video_feature_dict[feat] = feat_value
                elif 'righthand' in feat:
                    feat_value.append(right_hand_raw_lst[3*feat_num:3*(feat_num+1)])
                    video_feature_dict[feat] = feat_value
        
        cap = cv2.VideoCapture(os.path.join(project_path, dir_openpose, dir_video, mov_file))
        ret, first_frame = cap.read()
        
        for sim in range(0, sim_num+1):
            mask = np.zeros_like(first_frame)
            if sim == 0:
                key_reordered_arr = video_feature_dict.keys()
            else:
                key_reordered_arr = np.random.choice(list(video_feature_dict.keys()), len(video_feature_dict.keys()), replace=False)
            for k in key_reordered_arr:
                v = video_feature_dict[k]
                # (color, thickness)
                (c, t) = in_selected_f_dict[k] 
                color_counter = 0
                # No simulation for the initial position
                x_0 = int(v[0][0])
                y_0 = int(v[0][1])
                for points in v[1:]:
                    x_1 = int(points[0])
                    y_1 = int(points[1])
                    conf_1 = points[2]
                    if x_0 == 0 and y_0 == 0:
                        x_0 = x_1
                        y_0 = y_1
                    if x_1 != 0 and y_1 != 0:
                        # set default: no movement
                        sim_move_x = 0
                        sim_move_y = 0
                        est_var = var_arr[np.argmin(abs(prob_center_arr-conf_1))]
                        u = np.random.uniform(0, 1)
                        # print(u)
                        calc_prob_corner = multivariate_normal.cdf([0.5,-0.5], mean=[0,0], cov=[[est_var,0],[0,est_var]])
                        calc_prob_center = 1 - calc_prob_corner*4
                        if u > calc_prob_center:
                            # print('We are moving!')
                            while True:
                                try_move_x = int(np.random.choice(move_x_range, 1))
                                try_move_y = int(np.random.choice(move_y_range, 1))
                                try_move_x_neg = -abs(try_move_x)
                                try_move_y_neg = -abs(try_move_x)
                                
                                est_prob = multivariate_normal.cdf([try_move_x_neg+0.5,try_move_y_neg+0.5], mean=[0,0], cov=[[est_var,0],[0,est_var]]) \
                                           - multivariate_normal.cdf([try_move_x_neg+0.5,try_move_y_neg-0.5], mean=[0,0], cov=[[est_var,0],[0,est_var]]) \
                                           - multivariate_normal.cdf([try_move_x_neg-0.5,try_move_y_neg+0.5], mean=[0,0], cov=[[est_var,0],[0,est_var]]) \
                                           + multivariate_normal.cdf([try_move_x_neg-0.5,try_move_y_neg-0.5], mean=[0,0], cov=[[est_var,0],[0,est_var]])
                                u = np.random.uniform(0, 1)
                                if u < est_prob:
                                    sim_move_x = try_move_x
                                    sim_move_y = try_move_y
                                    break
                        x_1 = x_1 + sim_move_x
                        y_1 = y_1 + sim_move_y
                        c = [max(color_i-color_step,0) for color_i in c]
                        mask = cv2.line(mask, (x_0, y_0), (x_1, y_1), c, t)
                        color_counter += 1
                        x_0 = x_1
                        y_0 = y_1
            
            temp_segments = len(mov_file.split('_'))
            word = mov_file.split('_')[temp_segments-4]
            counter = mov_file.split('_')[temp_segments-2]
            train_test = mov_file.split('_')[temp_segments-1].split('.')[0]
            save_file = 'mask_' + word + '_' + counter + '_' + train_test + '_sim' + str(sim) 
            # print('Save as', save_file)
            plt.imsave(os.path.join(project_path, dir_openpose, outdir_opt_flow, save_file + '.png'), mask)
            if 'train' in save_file:
                plt.imsave(os.path.join(project_path, dir_openpose, outdir_transfer, word, save_file + '.png'), mask)
                
        cap.release()
        cv2.destroyAllWindows()


selected_feature_dict = dict()
# add head detection
selected_feature_dict['body_0'] = (color_white, 2)
selected_feature_dict['body_3'] = (color_red, 2)
selected_feature_dict['body_4'] = (color_red, 2)
selected_feature_dict['body_6'] = (color_green, 2)
selected_feature_dict['body_7'] = (color_green, 2)
# different color for thumbs
selected_feature_dict['lefthand_4'] = (color_blue, 1)
selected_feature_dict['lefthand_8'] = (color_yellow, 1)
selected_feature_dict['lefthand_12'] = (color_yellow, 1)
selected_feature_dict['lefthand_16'] = (color_yellow, 1)
selected_feature_dict['lefthand_20'] = (color_yellow, 1)
# different color for thumbs
selected_feature_dict['righthand_4'] = (color_pink, 1)
selected_feature_dict['righthand_8'] = (color_teal, 1)
selected_feature_dict['righthand_12'] = (color_teal, 1)
selected_feature_dict['righthand_16'] = (color_teal, 1)
selected_feature_dict['righthand_20'] = (color_teal, 1)

# increase range of movement
move_x_range = np.arange(-20, 21)
move_y_range = np.arange(-20, 21)

COLOR_MIN = 50

draw_optical_flow(outdir_opt_flow='NEW4_manual_optical_flow_output_trial4', 
                  outdir_transfer='NEW4_image_transfer_trial4', 
                  in_selected_f_dict=selected_feature_dict, 
                  sim_num=20)

print(len([f for f in os.listdir(os.path.join(project_path, dir_openpose, 
                                              'NEW4_manual_optical_flow_output_trial4')) 
           if 'train' in f]))

