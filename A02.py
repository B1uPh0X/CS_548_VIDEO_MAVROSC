import sys
import A01
import numpy as np
import torch
import cv2
import pandas
import sklearn
import math as m
from skimage.color import rgb2gray
import numpy as np
from enum import Enum
class OPTICAL_FLOW(Enum):
    HORN_SHUNCK = "horn_shunck"
    LUCAS_KANADE = "lucas_kanade"


def compute_video_derivatives(video_frames, size):
    cur_frame = None
    prev_frame = None
    all_fx = []
    all_fy = []
    all_ft = []
    
    if size==2:
        kfx = np.array([[-1,1],
                       [-1,1]], dtype="float64")
        kfy = np.array([[-1,-1],
                       [ 1,1]], dtype="float64")
        kft1 = np.array([[-1,-1],
                        [-1,-1]], dtype="float64")
        kft2 = np.array([[1,1],
                        [1,1]], dtype="float64")
    elif size==3:
        kfx = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]], dtype="float64")
        kfy = np.array([[-1,-2,-1],
                        [0,0,0],
                        [1,2,1]], dtype="float64")
        kft1 = np.array([[-1,-2,-1],
                        [-2,-4,-2],
                        [-1,-2,-1]], dtype="float64")
        kft2 = np.array([[1,2,1],
                        [2,4,2],
                        [1,2,1]], dtype="float64")
    else:
        return None
    
    for i in video_frames:
        cur_frame = (cv2.cvtColor(i,cv2.COLOR_RGB2GRAY).astype("float64"))/255.0
        if prev_frame is None:
            prev_frame = cur_frame
        fx = (cv2.filter2D(prev_frame, -1, kfx) + cv2.filter2D(cur_frame, -1, kfx))
        fy = (cv2.filter2D(prev_frame, -1, kfy) + cv2.filter2D(cur_frame, -1, kfy))
        ft = (cv2.filter2D(prev_frame, -1, kft1) + cv2.filter2D(cur_frame, -1, kft2))
        if size==2:
            fx = fx/4.0
            fy = fy/4.0
            ft = ft/4.0
        else:
            fx = fx/8.0
            fy = fy/8.0
            ft = ft/16.0
        all_fx.append(fx)
        all_fy.append(fy)
        all_ft.append(ft)
        prev_frame = cur_frame
    
    return all_fx, all_fy, all_ft
        
        

def compute_one_optical_flow_horn_shunck(fx, fy, ft, max_iter, max_error, weight=1.0):
    
    u = np.zeros(fx.shape, dtype="float64")
    v = np.zeros(fx.shape, dtype="float64")
    
    kfx = np.array([-1,1], dtype="float64")
    kfy = np.array([[-1],[1]], dtype="float64")
    iter_cnt = 0
    
    while j == "not done":
        uav = cv2.filter2D(u, cv2.CV_64F, kfx)
        vav = cv2.filter2D(v, cv2.CV_64F, kfy)
        
        berr = (fx*uav + fy*vav + ft) * (fx*uav + fy*vav + ft)
        serr = weight + fx*fx + fy*fy
        
        err = np.mean(berr+serr)
                        
        iter_cnt += 1
        
        if (iter_cnt >= max_iter) or (err <= max_error):
            j = "done"
            
    
    extra = np.zeros_like(u)
    combo = np.stack([u,v,extra], axis=-1)
    
    return combo 

def compute_one_optical_flow_lucas_kanade(fx, fy, ft, win_size):
    return None
def compute_optical_flow(video_frames, method=OPTICAL_FLOW.HORN_SHUNCK, max_iter=10, max_error=1e-4, horn_weight=1.0, kanade_win_size=19):
    return None
def main():      
    # Load video frames 
    video_filepath = "assign02/input/simple/image_%07d.png" 
    #video_filepath = "assign02/input/noice/image_%07d.png" 
    video_frames = A01.load_video_as_frames(video_filepath) 
     
    # Check if data is invalid 
    if video_frames is None: 
        print("ERROR: Could not open or find the video!") 
        exit(1) 
         
    # OPTIONAL: Only grab the first five frames 
    video_frames = video_frames[0:5] 
         
    # Calculate optical flow 
    flow_frames = compute_optical_flow(video_frames, method=OPTICAL_FLOW.HORN_SHUNCK) 
 
    # While not closed... 
    key = -1 
    ESC_KEY = 27 
    SPACE_KEY = 32 
    index = 0 
     
    while key != ESC_KEY: 
        # Get the current image and flow image 
        image = video_frames[index] 
        flow = flow_frames[index] 
         
        flow = np.absolute(flow) 
         
        # Show the images 
        cv2.imshow("ORIGINAL", image) 
        cv2.imshow("FLOW", flow) 
             
        # Wait 30 milliseconds, and grab any key presses 
        key = cv2.waitKey(30) 
         
        # If space, move forward 
        if key == SPACE_KEY: 
            index += 1 
            if index >= len(video_frames): 
 
                index = 0 
 
    # Destroy the windows     
    cv2.destroyAllWindows() 
     
if __name__ == "__main__":  
    main() 