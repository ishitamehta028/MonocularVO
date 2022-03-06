import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os



focal = 718.8560
principal_pt = (607.1928, 185.2157)


file_path = "C:\\Users\\Ishita\\Desktop\\CS\\openCV\\MVO\\dataset\\sequences\\00\\image_0\\"
pose_path = "C:\\Users\\Ishita\\Desktop\\CS\\openCV\\MVO\\dataset\\dataset\\poses\\00.txt"





def find_scale(id):

    file = open(pose_path, "r")
    data = file.readlines()
    
    if (id > 0):
        pose = data[id-1].strip().split()
        x_prev = float(pose[3])
        y_prev = float(pose[7])
        z_prev = float(pose[11])
        prev_coord = np.array([x_prev, y_prev, z_prev])

    pose = data[id].strip().split()
    x = float(pose[3])
    y = float(pose[7])
    z = float(pose[11])

    true_coord = np.array([x, y, z])
    print("true co : ", true_coord)

    if (id > 0):
        scale = np.linalg.norm(true_coord - prev_coord)
        return true_coord, scale

    return true_coord,0







def FAST(frame) : 
    #   FAST algorithm to track features

    fast = cv.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True)
    keypoints = fast.detect(frame ,None)

    frame = cv.drawKeypoints(frame, keypoints,frame, color=(255,0,0))
    print("keypoints: ",len(keypoints))
    return keypoints




def visual_odometry(img, img2, id):

    # R, t are Rotation matrix, translation vector
    global R_first, t_co
    R = np.zeros(shape=(3, 3))
    t = np.empty(shape=(3, 1))
    

    print("Frame ", id)

    keypoints = FAST(img)



    #   KLT Tracker

    kp = np.array([x.pt for x in keypoints], dtype=np.float32).reshape(-1, 1, 2)
    mask = np.zeros_like(img)
    lk_params = dict( winSize = (15, 15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                              10, 0.03))

    [flow, st, err] = cv.calcOpticalFlowPyrLK(img, img2, kp, None,  **lk_params)

    # Select good points 
    good_new = flow[st == 1]
    good_old = kp[st == 1]


    # checks for absolute scale and true co-ordinates
    true_co, scale = find_scale(id)


    #  Find Essential matrix
    if (id < 2) :
        R_first = np.zeros(shape=(3, 3))
        t_co = np.empty(shape=(3, 1))
        
        E, _= cv.findEssentialMat(good_new, good_old,focal, principal_pt  , cv.RANSAC,0.999, 1.0, None)
        _, R_first, t_co, _ = cv.recoverPose(E, good_old, good_new, R, t, focal, principal_pt, None)


    else : 
        E, _= cv.findEssentialMat(good_new, good_old,focal, principal_pt  , cv.RANSAC,0.999, 1.0, None)
        _, R, t, _ = cv.recoverPose(E, good_old, good_new, R, t, focal, principal_pt, None)

    

    if (scale > 0.1 ) :
        t_co += scale * R_first.dot(t)
        R_first  = R.dot(R_first)
        
     


    diag = np.array([[-1, 0, 0],
                     [0, -1, 0],
                     [0, 0, -1]])


    adj_coord = np.matmul(diag, t_co)
    mono_coord = adj_coord.flatten()
    print("estimated co ", mono_coord)
    return mono_coord, true_co

