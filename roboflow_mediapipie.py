import argparse
import time
from pathlib import Path
import os
from tkinter import font

import clip
import mediapipe as mp
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import pandas as pd

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import xyxy2xywh, xywh2xyxy, \
    strip_optimizer, set_logging, increment_path, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_sync
from utils.roboflow import predict_image

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_clip_detections as gdet

from utils.yolov5 import Yolov5Engine
from utils.yolov4 import Yolov4Engine



def get_mediapipe_keypoint_numbers():
    #Helper function to get num of points in mediapipe
    numBodyPoints = 33
    numFacePoints = 468
    numHandPoints = 21

    numTrackedPoints = (
        numBodyPoints + numHandPoints * 2 + numFacePoints
    )
    return numBodyPoints,numFacePoints,numHandPoints,numTrackedPoints

def format_mediapipe_for_fmc(results, image_height, image_width):
    # Helper function to reshape mediapipe output to fmc format

    mediapipe_nFrames_XYC = []
    numBodyPoints,numFacePoints,numHandPoints,numTrackedPoints = get_mediapipe_keypoint_numbers()

    numTrackedPoints = numBodyPoints #hardcoding this rn because we're not using face/hands for this testing #NOTE- 12/3 AC

    mediaPipeData_XYC = np.empty((int(numTrackedPoints),3))
    mediaPipeData_XYC[:] = np.NaN

    thisFrame_X_body = np.empty(numBodyPoints)
    thisFrame_X_body[:] = np.nan
    thisFrame_Y_body = thisFrame_X_body.copy()
    thisFrame_C_body = thisFrame_X_body.copy()


    fullFrame = True

    try:
        # pull out ThisFrame's mediapipe data (`mpData.pose_landmarks.landmark` returns something iterable ¯\_(ツ)_/¯)
        thisFrame_poseDataLandMarks = results.landmark  # body ('pose') data
        # stuff body data into pre-allocated nan array
        thisFrame_X_body[:numBodyPoints] = [
            pp.x for pp in thisFrame_poseDataLandMarks
        ]  # PoseX data - Normalized screen coords (in range [0, 1]) - need multiply by image resultion for pixels
        thisFrame_Y_body[:numBodyPoints] = [
            pp.y for pp in thisFrame_poseDataLandMarks
        ]  # PoseY data
        thisFrame_C_body[:numBodyPoints] = [
            pp.visibility for pp in thisFrame_poseDataLandMarks
        ]  #'visibility' is MediaPose's 'confidence' measure in range [0,1]
    except:
        fullFrame = False

    thisFrame_X = thisFrame_X_body
    thisFrame_Y = thisFrame_Y_body
    thisFrame_C = thisFrame_C_body
    # stuff this frame's data into pre-allocated mediaPipeData_.... array
    mediaPipeData_XYC[:,0] = thisFrame_X
    mediaPipeData_XYC[:,1] = thisFrame_Y
    mediaPipeData_XYC[:,2] = thisFrame_C

    mediaPipeData_XYC[:, 0] *= image_width
    mediaPipeData_XYC[:, 1] *= image_height
    
    mediapipe_nFrames_XYC.append(mediaPipeData_XYC)
    #np.append(mediapipe_nFrames_XYC,mediaPipeData_XYC)
    return mediapipe_nFrames_XYC    



def write_bbox_over_image(tracker,save_path,im0, info,detection_engine,thickness,names):
    #Taxes the bounding box coords and plots the box over the image
    if len(tracker.tracks):
        print("[Tracks]", len(tracker.tracks))

    for track in tracker.tracks:
        
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        xyxy = track.to_tlbr()
        class_num = track.class_num
        bbox = xyxy
        class_name = names[int(class_num)] if detection_engine == "yolov5" else class_num
        if info:
            print("Tracker ID: {}, Class: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        label = f'{class_name} #{track.track_id}'
        plot_one_box(xyxy, im0,save_path=save_path, label=label,
                        color=get_color_for(label), line_thickness=thickness)
        

def get_color_for(class_num):
    #Bounding box label color
    colors = [
        "#4892EA",
        "#00EEC3",
        "#FE4EF0",
        "#F4004E",
        "#FA7200",
        "#EEEE17",
        "#90FF00",
        "#78C1D2",
        "#8C29FF"
    ]

    num = hash(class_num) # may actually be a number or a string
    hex = colors[num%len(colors)]
    # adapted from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    rgb = tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    return rgb


def run_roboflow(weights='yolov5s.pt',
           cfg = 'yolov5.cfg0',
           names = 'coco.names',
           input_video = 'data/images',
           img_size = 640,
           confidence=0.4,
           overlap=0.3,
           thickness=3,
           device ='',
           view_img = False,
           save_txt = True,
           save_conf = True,
           classes =0,
           agnostic_nms = True,
           augment = True,
           update = True,
           project = 'runs/detect',
           name = 'exp',
           exist_ok = True,
           nms_max_overlap=1.0, 
           max_cosine_distance = .4,
           nn_budget = None,
           api_key = None,
           info = True,
           url = None,
           detection_engine = 'yolov5',
           save_bbox_vid=True):

    """Function takes an input video, runs through yolo and deep sort algorithms to 
    output a dictionary with the coordinates of a bounding box over each person in the 
    vide"""

    t0 = time_sync()
    # initialize deep sort
    model_filename = "ViT-B/16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = device != "cpu"
    model, transform = clip.load(model_filename, device=device, jit=False)
    model.eval()
    encoder = gdet.create_box_encoder(model, transform, batch_size=1, device=device)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)

    # load yolov5 model here
    if detection_engine == "yolov5":
        yolov5_engine = Yolov5Engine(weights, device, classes, confidence, overlap, agnostic_nms, augment, half)
        names = yolov5_engine.get_names()

    elif detection_engine == "yolov4":
        yolov4_engine = Yolov4Engine(weights, cfg, device, names, classes, confidence, overlap, agnostic_nms, augment, half)

    # initialize tracker
    tracker = Tracker(metric)
    source, weights, view_img, save_txt, imgsz = input_video, weights, view_img, save_txt, img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    frame_count = 0
    dict_of_all_tracks = {}
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    if detection_engine == "yolov5":
        _ = yolov5_engine.infer(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path,im0,im0s,vid_cap,test5 in dataset:
        
        img = torch.from_numpy(im0).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Roboflow Inference
        t1 = time_sync()
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        # choose between prediction engines (yolov5 and roboflow)
        if detection_engine == "roboflow":
            pred, classes = predict_image(im0, api_key, url, confidence, overlap, frame_count)
            pred = [torch.tensor(pred)]
        elif detection_engine == "yolov5":
            print("yolov5 inference")
            pred = yolov5_engine.infer(img)
        else:
            print("yolov4 inference {}".format(im0.shape))
            pred = yolov4_engine.infer(im0)
            pred, classes = yolov4_engine.postprocess(pred, im0.shape)
            pred = [torch.tensor(pred)]

        t2 = time_sync()

        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            #moved up to roboflow inference
            """if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)"""

            p = Path(p)  # to Path

            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Transform bboxes from tlbr to tlwh
                trans_bboxes = det[:, :4].clone()
                trans_bboxes[:, 2:] -= trans_bboxes[:, :2]
                bboxes = trans_bboxes[:, :4].cpu()
                confs = det[:, 4]
                class_nums = det[:, -1]
                classes = class_nums
      
                # encode yolo detections and feed to tracker
                features = encoder(im0, bboxes)
                detections = [Detection(bbox, conf, class_num, feature) for bbox, conf, class_num, feature in zip(
                    bboxes, confs, classes, features)]

                # run non-maxima supression
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                class_nums = np.array([d.class_num.cpu() for d in detections])
                indices = preprocessing.non_max_suppression(
                    boxs, class_nums, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker
                tracker.predict()
                tracker.update(detections)
                
                #Loop through each tracked person and save their coords to a dataframe with their ID
                for track in tracker.tracks:
                    if track.track_id in dict_of_all_tracks.keys():
                        xyxy = np.array(track.to_tlbr())
                        fr_xyxy = np.insert(xyxy,0,frame)
                        this_frame_df = pd.DataFrame(fr_xyxy).T
                        this_frame_df.columns = ['frame','Xmin','Ymin','Xmax','Ymax']
                        current_df_for_this_track =dict_of_all_tracks[track.track_id]
                        
                        appended_df = current_df_for_this_track.append(this_frame_df)
                        dict_of_all_tracks[track.track_id] = appended_df
                    else:
                        xyxy = np.array(track.to_tlbr())
                        fr_xyxy = np.insert(xyxy,0,frame)
                        
                        this_frame_df = pd.DataFrame(fr_xyxy).T
                        this_frame_df.columns = ['frame','Xmin','Ymin','Xmax','Ymax']
                        
                        dict_of_all_tracks[track.track_id] = this_frame_df        

    print(f'Done. ({time.time() - t0:.3f}s)')

    return dict_of_all_tracks

def mediaPipe_multiperson(source_vid, output_vid,tracked_roboflow_data):
    """Function takes in input video and the corresponding dictionary containing 
    the coordinates of the bounding boxes of each person in the video. Function 
    crops the frames of the video to the coordiates of the bounding box and 
    then runs them through mediapipe. Outputs a dictionary of the mediapipe data"""

    #Initilaize medipapipe tools
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    #Create a video capture object
    cap = cv2.VideoCapture(source_vid) 
    
    #Get video properites
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)
    fps = 30
    #Create a buffer to make the bounding box a little bigger(sometimes the bbox cuts off limbs)
    width_bounding_box_buffer = frame_width*.05
    height_bounding_box_buffer = frame_height*.05
    
    #Initialize dictionary for mediapipe data
    fmc_mediapipe_dict = {}
    
    #Get fourcc for video encoding
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #Create output video object
    output = cv2.VideoWriter(output_vid, fourcc, fps, frame_size) #choose a name/path for an output video 
    frame = 0#Initialize frame counter

    #Loop through video
    while (cap.isOpened()):

        frame += 1#increase frame count
        # Capture frame-by-frame
        ret, image = cap.read()

        if ret==True:
            #Copy frame image
            annotated_image = image.copy()
            
            #For each person in the roboflow data 
            for person_key in tracked_roboflow_data.keys():
                #With mediapipe
                with mp_holistic.Holistic(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False) as holistic:
                    #get the df of this person
                    person_df = tracked_roboflow_data[person_key]
                    frames_df = person_df['frame']
                    # Get list of frames this person is in
                    frames_list = frames_df.to_list()
                    #If this frame is in the frames_list (Not all people will be in every frame)
                    if frame in frames_list:
                        #Get idx of this frame in this persons frame list
                        idx = frames_list.index(frame)
                        #Get image properties
                        image_height, image_width, _ = image.shape
                        #convert to numpy
                        person_npy = person_df.to_numpy()
                        #Take data of just this frame
                        frame_yolo_data =person_npy[idx,:]

                        #TODO: Add the following code to make the bounding boxes bigger
                        # if 0 < int(frame_yolo_data[1]  - width_bounding_box_buffer):
                        #     xmin = int(frame_yolo_data[1]  - width_bounding_box_buffer)
                        # else:
                        #     xmin = 0
                        # if frame_width > int((frame_yolo_data[3]) +width_bounding_box_buffer):
                        #     xmax = int((frame_yolo_data[3]) +width_bounding_box_buffer)
                        # else:
                        #     xmax = frame_width
                        # if 0 < int(frame_yolo_data[2] - height_bounding_box_buffer):
                        #     ymin = int(frame_yolo_data[2] - height_bounding_box_buffer)
                        # else:
                        #     ymin = 0
                        # if frame_height > int((frame_yolo_data[4]) + height_bounding_box_buffer):
                        #     ymax = int((frame_yolo_data[4]) + height_bounding_box_buffer)
                        # else:
                        #     ymax = frame_height
                        # if ymin > ymax:
                        #     ymin = ymax
                        #     ymax = ymin
                        # if xmin > xmax:
                        #     xmin = xmax
                        #     xmax = xmin

                        #Get bbox mins and maxs from the yolo data
                        xmin = int(frame_yolo_data[1])
                        xmax  = int(frame_yolo_data[3])
                        ymin  = int(frame_yolo_data[2])
                        ymax  = int(frame_yolo_data[4]) 

                        #Crop image to the bbox coords
                        cropped_image = image[ymin:ymax, xmin:xmax] 
                        #get image shape
                        cropped_image_height, cropped_image_width, _ = cropped_image.shape
                        #Run through mediapipe
                        try:
                            process_results = holistic.process(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)) #process mediapipe on just the cut out rectangle 
                        except:
                            pass

                        #Get the scale for the crop image to original image
                        y_scale = cropped_image_height/image_height 
                        x_scale = cropped_image_width/image_width

                        #Scale each pose, face and hand points to original image coords
                        try:
                            for joint in process_results.pose_landmarks.landmark:
                                joint.y = joint.y*y_scale + ymin/image_height
                                joint.x = joint.x*x_scale + xmin/image_width
                        except: 
                            pass
                        try:
                            for f_joint in process_results.face_landmarks.landmark:
                                f_joint.y = f_joint.y*y_scale + ymin/image_height
                                f_joint.x = f_joint.x*x_scale + xmin/image_width
                        except:
                            pass

                        try:    
                            for r_joint in process_results.right_hand_landmarks.landmark:
                                r_joint.y = r_joint.y*y_scale + ymin/image_height
                                r_joint.x = r_joint.x*x_scale + xmin/image_width
                        except:
                            pass

                        try:    
                            for l_joint in process_results.left_hand_landmarks.landmark:
                                l_joint.y = l_joint.y*y_scale + ymin/image_height
                                l_joint.x = l_joint.x*x_scale + xmin/image_width
                        except:
                            pass
                        #If there is face, hand, or pose landmarks (probably a better way to make this if statement lol)
                        if process_results.face_landmarks or process_results.left_hand_landmarks or process_results.pose_landmarks or process_results.right_hand_landmarks:
                            #Draw face point
                            mp_drawing.draw_landmarks(
                                annotated_image,
                                process_results.face_landmarks,
                                mp_holistic.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_contours_style())
                            #Draw body point
                            mp_drawing.draw_landmarks(
                                annotated_image,
                                process_results.pose_landmarks,
                                mp_holistic.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.
                                get_default_pose_landmarks_style())
                            #Draw right hand point
                            mp_drawing.draw_landmarks(
                                annotated_image,
                                process_results.right_hand_landmarks,
                                mp_holistic.HAND_CONNECTIONS)
                            #Draw left hand point
                            mp_drawing.draw_landmarks(
                                annotated_image,
                                process_results.left_hand_landmarks,
                                mp_holistic.HAND_CONNECTIONS)
                            #TODO: else statement for just mediapipe
                            
                #Get mediapipe data and format it into free mocap form
                mediapipe_data = process_results.pose_landmarks
                Mediapipe_XYC = format_mediapipe_for_fmc(mediapipe_data, frame_height,frame_width)
                
                #Add these mediapipe points to the ID of the person from the roboflow output
                if person_key in fmc_mediapipe_dict.keys():
                    reprocessed_list = fmc_mediapipe_dict[person_key]
                    reprocessed_list.append(Mediapipe_XYC)
                    fmc_mediapipe_dict[person_key] = reprocessed_list

                else:
                    new_list = []
                    new_list.append(Mediapipe_XYC)
                    fmc_mediapipe_dict[person_key] = new_list
                #Add bounding box to image
                annotated_image = cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)),color = (255,0,0), thickness=3, lineType=cv2.LINE_AA)
            output.write(annotated_image)
                
        else:
            break
    cap.release()
    output.release()
    return fmc_mediapipe_dict


def derive_position_array(mediapipe_nFrames_XYC):
    '''Get derivative of mediaipipe data'''
    #Square the data
    mediapipe_nFrames_XYC_squared = mediapipe_nFrames_XYC[:,0]**2
    #Add the x and y axis (a squared + b squared)
    c_squared = mediapipe_nFrames_XYC_squared[:,0,0]+mediapipe_nFrames_XYC_squared[:,0,1]
    #Take square root to get distance 
    mediapipe_nFrames =  np.sqrt(c_squared)
    #Take derivative of the distances
    derived_mediapipe_nFrames_XY = np.gradient(mediapipe_nFrames)
    return np.abs(derived_mediapipe_nFrames_XY)



def multi_person_sorting(list_of_mediapipe_dictionaries,cam_with_most_people):
    '''Function takes in multiperson mediapipe data and assigns a number each person 
    to be consistent across all camera views'''
    #Create empty dictionary for sorted data
    sorted_multi_person_dict = {}

    #Access the dictionary of the video with the most people in it
    max_people_dict = list_of_mediapipe_dictionaries[cam_with_most_people]
    #Create empty dict for the sorting of this video
    max_people_dict_sorted ={}
    k = 1
    for person_key in max_people_dict.keys():
        #Loop through and assign each person in this video a number 1,2,3,....,N
        max_people_dict_sorted[k] = max_people_dict[person_key]
        k +=1

    #Add this to the dictioanry of all videos 
    sorted_multi_person_dict[cam_with_most_people] = max_people_dict_sorted


    #Iterate over each camera view
    for j,mediapipe_dict in enumerate(list_of_mediapipe_dictionaries):
        #If the loop is at the camera view with most people skip that loop
        if j == cam_with_most_people:
            continue
        else:
            this_cam_view_sorted  = {}
            #For each person in the video
            for person_key in mediapipe_dict.keys():
                #Get derived data
                derived_mediapipe_nFrames_XY = derive_position_array(np.array(mediapipe_dict[person_key]))

                #Initilaize high value for comparison
                smallest_difference =1000000
                #For each person in the video with the most people
                for max_person_key in max_people_dict.keys():
                    #get derived data of the person in the max people video
                    derived_mediapipe_nFrames_XY_comparison = derive_position_array(np.array(max_people_dict[max_person_key]))
                    #Find difference
                    difference = np.abs(derived_mediapipe_nFrames_XY - derived_mediapipe_nFrames_XY_comparison)
                    #Sum differences
                    sum_difference = np.sum(difference)
                    #If this is the smallest difference so far
                    if sum_difference < smallest_difference:
                        #Assign it to smallest difference
                        smallest_difference = sum_difference
                        #Assign the number of the person from the max person video 
                        #to the number of the person in this video 
                        this_persons_number = max_person_key   
                #Add this persons data to this camera view
                this_cam_view_sorted[this_persons_number] = mediapipe_dict[person_key]
            #Add this cam view to big dictionary
            sorted_multi_person_dict[j] = this_cam_view_sorted

    return sorted_multi_person_dict

def visualize(mediaPipeData_nCams_nFrames_nPeople_nImgPts_XYC, sourceFolder):
    '''Debugging function to visualize if the people in the video were sorted correctly'''
    vids = os.listdir(sourceFolder)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i,vid in enumerate(vids):
        mediaPipeData_nFrames_nPeople_nImgPts_XYC = mediaPipeData_nCams_nFrames_nPeople_nImgPts_XYC[i,:,:,:,:]
        name, ext = os.path.splitext(vid)  
        frame = 0
        cap = cv2.VideoCapture(sourceFolder+vid)
        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_size = (frame_width,frame_height)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          
        outvid = name+'_sorted.mp4'
        output = cv2.VideoWriter(sourceFolder+outvid, fourcc, fps, frame_size) #choose a name/path for an output video 
        while (frame < frame_count):
            ret, image = cap.read()
            if ret==True:
                annotated_image = image.copy()
                for person_number in range(mediaPipeData_nFrames_nPeople_nImgPts_XYC.shape[1]):
                    xpos = np.mean(mediaPipeData_nFrames_nPeople_nImgPts_XYC[frame,person_number,:,0])
                    ypos = np.mean(mediaPipeData_nFrames_nPeople_nImgPts_XYC[frame,person_number,:,1])
                    if not np.isnan(xpos):
                        cv2.putText(annotated_image, str(person_number), (int(xpos),int(ypos)), font,5, (0, 255, 0), 2, cv2.LINE_AA)
                output.write(annotated_image)
                frame+=1
            else:
                break
        cap.release()

        output.release()
        


if __name__ == '__main__':
    #Set path for where the yolo models are
    yolo_path = 'C:/Users/chris/roboflow_mediapipe/'
    #Set path for the folder containing the input videos
    sourceFolder = "C:/Users/chris/OneDrive/Documents/Spring2022/Humon_Lab/multi_person_detection/Jon_and_Stephen_mediapiped/Multi_test/"
    #Set path for the folder where you want the output videos
    outSourceFolder = "C:/Users/chris/OneDrive/Documents/Spring2022/Humon_Lab/multi_person_detection/Jon_and_Stephen_mediapiped/Multi_test_out/"
    #Get list of all the videos in the input folder
    vids = os.listdir(sourceFolder)
    #Number of camera views equal to amount of input videos
    numCams = len(vids)
    #Create video capture object from first video
    vidcap = cv2.VideoCapture(sourceFolder+vids[0])
    #Get number of frames in the video (Assumes the input videos are synced so all frame lengths will be the same)
    numFrames = int(vidcap.get((cv2.CAP_PROP_FRAME_COUNT)))
    #Get number of points from mediapipe
    num_body_points,num_face_points,num_hand_points,num_points = get_mediapipe_keypoint_numbers()
    
    
    #Create empty list for mediapipe data from each video and number of people in each video
    fmc_mediapipe_dict_dict_from_each_video = []
    num_people_in_each_video = []
    #Loop through each camera view
    for cam in range(numCams):
        #With pytorch
        with torch.no_grad():
            #Run roboflow
            dict_of_all_bounding_boxes =run_roboflow(input_video=sourceFolder + vids[cam],
                        weights = yolo_path+'models/yolov5s.pt', cfg=yolo_path+'models/yolov5s.cfg', names=yolo_path+'coco.names')
        #Make name of output video same as input name with '_mediapipe' at end
        name, ext = os.path.splitext(vids[cam])    
        outvid = name+'_mediapipe.mp4'
        #Run mediapipe on the cropped image and append it the output to list of all camera views
        fmc_mediapipe_dict_dict_from_each_video.append(mediaPipe_multiperson(sourceFolder + vids[cam], outSourceFolder+outvid,dict_of_all_bounding_boxes))
        #Get the number of people in the video and append it to the number of people list
        num_people_in_each_video.append(len(fmc_mediapipe_dict_dict_from_each_video[cam]))
    
    #Get amount of people in the video with the largest number of people
    num_people = np.max(num_people_in_each_video)
    #Get video number with the most people in it
    cam_num_with_max_people = np.argmax(num_people_in_each_video)

    #If single person
    if num_people ==1:
        #Initialize array in format with no people in it 
        mediaPipeData_nCams_nFrames_nImgPts_XYC = np.ndarray((numCams,numFrames,num_points,3))
        k = 0#initialize counter
        #For each camera view
        for cam in range(numCams):
            #Take the mediapipe dictionary from that view
            fmc_mediapipe_dict_dict = fmc_mediapipe_dict_dict_from_each_video[cam]
            #For each person in the dictionary
            for person_key in fmc_mediapipe_dict_dict.keys():
                #Access that persons data 
                person = fmc_mediapipe_dict_dict[person_key]
                #Convert to numpy
                personnpy = np.array(person)
                #Add to the numpy array to the output array 
                mediaPipeData_nCams_nFrames_nImgPts_XYC[cam,:,:,:] = personnpy

    #If multi person
    if num_people >1:
        #Sort through the mediapipe output so the key associated with each person is consistent in all videos
        sorted_multi_person_dict = multi_person_sorting(fmc_mediapipe_dict_dict_from_each_video, cam_with_most_people=int(cam_num_with_max_people))
        #Create an empty array for the output
        mediaPipeData_nCams_nFrames_nPeople_nImgPts_XYC = np.ndarray((numCams,numFrames,int(num_people),num_body_points,3))
        k = 0#Initilaze counter variable
        #For each camera view
        for cam in sorted_multi_person_dict.keys():
            #Get this cameras dictionary
            cam_Mediapipe_dict = sorted_multi_person_dict[cam]
            #For each person in the dictionary
            for person_key in cam_Mediapipe_dict.keys():
                #Convert the data to numpy array
                this_person = np.array(cam_Mediapipe_dict[person_key])
                #Add that data to the output array
                mediaPipeData_nCams_nFrames_nPeople_nImgPts_XYC[cam,:,person_key-1,:,:] = this_person[:,0,:,:]
        
        #Creates a video with the label on each person in each camera view
        visualize(mediaPipeData_nCams_nFrames_nPeople_nImgPts_XYC,outSourceFolder)
        

    #Save out numpy array 
    np.save(sourceFolder+'mediaPipeData_2d.npy',mediaPipeData_nCams_nFrames_nPeople_nImgPts_XYC)
            