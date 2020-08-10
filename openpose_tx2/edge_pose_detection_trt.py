import json
import os
import sys
import time
import math
import cv2
import numpy as np
print(sys.path)
sys.path.append('../root/openpose/build/python')

import json
import tensorflow as tf
#from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
np.random.seed(251)
# Below is needed to load model
from keras_efficientnets import EfficientNetB0 ,preprocess_input
from openpose import pyopenpose as op
from matplotlib import pyplot as plt
import warnings
def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (0, 0, 0)
    thickness = cv2.LINE_AA
    margin = 5

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin
    cv2.putText(img, text, pos, font_face, scale, color, 2, cv2.LINE_AA)


output_names = ['dense_1/Softmax']
input_names = ['model_2_input']



def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def



if __name__ == '__main__':
    
    warnings.filterwarnings("ignore")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    NUM_OF_FRAME = 20
    color_step = 200 / NUM_OF_FRAME
    params = dict()
    params["model_folder"] = "../root/openpose/models/"
    params["hand"] = True
    params["hand_detector"] = 3
    params["hand_scale_number"] = 6
    params["net_resolution"] = "288x288"
    params["hand_render_threshold"] = 0.2
    params["hand_render"] = -1
    #params["write_json"] = "../output/"
    params["model_pose"] = "BODY_25"
    # Paths - should be the folder where Open Pose JSON output was stored
    json_filepath = "../output/"
    save_json=False     
    prediction="GUESSING"
    # Parameters used in the manual optical flow
    color_red = [255, 0, 0]
    color_green = [0, 255, 0]
    color_blue = [0, 0, 255]
    color_yellow = [255, 255, 0]
    color_pink = [255, 0, 255]
    color_teal = [0, 255, 255]
    color_white = [255, 255, 255]

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
    # Parameters needed for model scoring
    model_file = 'trt_graph.pb'
    
    trt_graph = get_frozen_graph(model_file) 
    #Create session and load graph
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(trt_graph, name='')

            # Get graph input size
    for node in trt_graph.node:
        print("node ======> ", node.name) 
        if 'input_' in node.name:
            size = node.attr['shape'].shape
            image_size = [size.dim[i].size for i in range(1, 4)]
            break
            #print("image_size: {}".format(image_size))

            # input and output tensor names.
    input_tensor_name = input_names[0] + ":0"
    output_tensor_name = output_names[0] + ":0"

    print("input_tensor_name: {}\noutput_tensor_name: {}".format(input_tensor_name, output_tensor_name))
    output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)


    #model_path="../code/"
    #model_saved = load_model(os.path.join(model_path, model_file),compile=False)
    #print("saved model :" , model_saved)
    
    #model_saved = tf.keras.models.load_model(os.path.join(model_path, model_file)) #load_model(os.path.join(model_path, model_file))
    #print("saved model :" ,model_saved)
    MODEL_PREDICTION_THRESHOLD = 0.2

    class_list = ['AGAIN', 'ALL', 'AWKWARD', 'BASEBALL', 'BEHAVIOR', 'CAN', 'CHAT', 'CHEAP', 
              'CHEAT', 'CHURCH', 'COAT', 'CONFLICT', 'COURT', 'DEPOSIT', 'DEPRESS', 
              'DOCTOR', 'DRESS', 'ENOUGH', 'NEG']
    def conv_index_to_vocab(ind):
    	temp_dict = dict(enumerate(class_list))
    	return temp_dict[ind]
    
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    # Process Image
    datum = op.Datum()
    keypointdict={}
    outerdict={}
    keypointlist = []
    #Clean the output folder
    for filename in os.listdir(json_filepath):
        if filename.endswith(".json"):
            os.remove(os.path.join(json_filepath,filename))

    #Video capture from webcam
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print("The current fps is " ,fps)
    name= 1 #name of the json file
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            #cv2.imshow('Frame',frame)
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])
            writepath = os.path.join(json_filepath,"keypoint_{:012d}.json".format(name))
            mode = 'w' if os.path.exists(writepath) else 'w+'
            # Display Image
            outerdict["people"]=keypointlist
            keypointdict['pose_keypoints_2d'] = datum.poseKeypoints.flatten().tolist()
            keypointdict['hand_left_keypoints_2d'] = datum.handKeypoints[0].flatten().tolist()
            keypointdict['hand_right_keypoints_2d'] = datum.handKeypoints[1].flatten().tolist()
            keypointlist.append(keypointdict.copy())#must be the copy!!!
            ## Check the size before you save
            if len(keypointdict['pose_keypoints_2d']) ==75 and len(keypointdict['hand_left_keypoints_2d']) ==63 and len(keypointdict['hand_right_keypoints_2d']) ==63 :
            	save_json=True
            
            draw_label(frame, prediction, (30,30), (255,255,255))
            cv2.imshow('Frame',frame)
            # Display the resulting frame
            #cv2.imshow("OpenPose - Gesture Detection", datum.cvOutputData)

            #print("pose_keypoints_2d " ,datum.poseKeypoints.flatten().tolist())
			# Custom Params (refer to include/openpose/flags.hpp for more parameters)
            #print("hand_left_keypoints_2d " ,str(datum.handKeypoints[0]))
            if save_json==True:
            	with open(writepath, mode) as f :
                	json.dump(outerdict, f, indent=0 )
            
            outerdict.clear()
            keypointlist.clear()
            keypointdict.clear()
            #print("keypointdict " ,keypointdict)
            name = name + 1
            cv2.waitKey(10)
            
            #Inference code goes here
            full_file_lst = [f for f in os.listdir(json_filepath) if f.endswith('.json')]
            if len(full_file_lst) < NUM_OF_FRAME:
                print('skip')
                continue
            
            json_files_lst = full_file_lst[-NUM_OF_FRAME:]     
            video_feature_dict = dict()
            for json_f in json_files_lst:
                with open(os.path.join(json_filepath, json_f)) as ff:
                    json_code = json.load(ff)
        
        		# This assume there is only one person
                body_raw_lst = json_code['people'][0]['pose_keypoints_2d']
                left_hand_raw_lst = json_code['people'][0]['hand_left_keypoints_2d']
                right_hand_raw_lst = json_code['people'][0]['hand_right_keypoints_2d']
                
                for feat in list(selected_feature_dict.keys()):
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
                        
            mask = np.zeros_like(frame)
            #imgpath = os.path.join(model_path,"image_{:012d}.png".format(name))
            #plt.imsave(imgpath, mask)
            for (k, v) in video_feature_dict.items():
	        # (color, thickness)
                (c, t) = selected_feature_dict[k]
                color_counter = 0
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
                        c = [max(color_i-color_step,0) for color_i in c]
                        mask = cv2.line(mask, (x_0, y_0), (x_1, y_1), c, t)
                        color_counter += 1
                        x_0 = x_1
                        y_0 = y_1

            model_path = '.'
            x = cv2.resize(mask, (224,224))
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)#tf.keras.applications.resnet50.preprocess_input(x) #preprocess_input(x)



            # Optional image to test model prediction.
            feed_dict = {
                input_tensor_name: x
            }
            y_pred = tf_sess.run(output_tensor, feed_dict)
            #print("y_pred" ,y_pred)
            if np.max(y_pred) < MODEL_PREDICTION_THRESHOLD:
                print('?')
                prediction='Too hard to guess'
            else:
                prediction = conv_index_to_vocab(np.argmax(y_pred))
                print('Prediction: ', conv_index_to_vocab(np.argmax(y_pred)))
            
            feed_dict={}
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
             break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()










