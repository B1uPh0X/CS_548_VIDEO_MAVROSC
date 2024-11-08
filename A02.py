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
        cur_frame = (cv2.cvtColor(i,cv2.COLOR_BGR2GRAY).astype("float64"))/255.0
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
    
    # DEBUG
    error = 0
    
    return combo, error, iter_cnt

def compute_one_optical_flow_lucas_kanade(fx, fy, ft, win_size):
    
    u = np.zeros(fx.shape, dtype="float64")
    v = np.zeros(fx.shape, dtype="float64")

    height, width = fx.shape
    
    for x in range(0, width, win_size):
        for y in range(0, height, win_size):

            Ex = min(x + win_size, width)
            Ey = min(y + win_size, height)

            fxB = fx[y:Ey, x:Ex]
            fyB = fy[y:Ey, x:Ex]
            ftB = ft[y:Ey, x:Ex]

            fxfx = np.sum(fxB * fxB)
            fyfy = np.sum(fyB * fyB)
            fxfy = np.sum(fxB * fyB)
            fxft = np.sum(fxB * ftB)
            fyft = np.sum(fyB * ftB)

            de = ((fxfx * fyfy) - (fxfy * fxfy))
            if de >= 1e-6:
                u[y:Ey, x:Ex] = ((((fyfy)*(fxft))-((fxfy)*(fyft)))/de)
                v[y:Ey, x:Ex] = ((((fxfx)*(fyft))-((fxfy)*(fxft)))/de)
            else:
                u = u
                v = v
    
    flo = np.stack((u, v, np.zeros_like(u)), axis=-1)

    return flo
def compute_optical_flow(video_frames, method=OPTICAL_FLOW.HORN_SHUNCK, max_iter=10, max_error=1e-4, horn_weight=1.0, kanade_win_size=19):
    if method==OPTICAL_FLOW.HORN_SHUNK:
        dsize=2
        der=compute_video_derivatives(video_frames, dsize)
    elif method==OPTICAL_FLOW.LUCAS_KANADE:
        dsize=3
        der=compute_video_derivatives(video_frames, dsize)
        
    all_flow = []
    for i in range(len(der[0])):
        if method==OPTICAL_FLOW.HORN_SHUNK:   
            flow, _, _ = compute_one_optical_flow_horn_shunck(der[0][i], der[1][i], der[2][i], max_iter, max_error, horn_weight)
        elif method==OPTICAL_FLOW.LUCAS_KANADE:
            flow = compute_one_optical_flow_lucas_kanade(der[0][i], der[1][i], der[2][i], kanade_win_size)
            
        all_flow.append(flow)
        
    return all_flow
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