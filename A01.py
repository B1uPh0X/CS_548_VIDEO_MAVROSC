import cv2
import sys
import numpy as np
import torch
import pandas
import sklearn
import os
import shutil

def load_video_as_frames(video_filepath):
    video = cv2.VideoCapture(video_filepath)
    if not video.isOpened():
        print("ERROR: no video detected")
        return None
    
    # wName = "vid"
    #cv2.namedWindow(wName)
    frames = []    
    key = -1
    while key == -1:
        ret, frame = video.read()
        if ret == True:
            #cv2.imshow(wName, frame)
            frames.append(frame)
        else:
            break

    video.release()
    #cv2.destroyAllWindows()
    print("end")
    print(len(frames))
    return frames

def compute_wait(fps):
    #wait = (1000/fps)
    #wait = int(wait)
    #print("wait time:" + str(wait))
    #print("FPS: " + str(fps))
    return int(1000/fps)
    
def display_frames(all_frames, title, fps=30):
    print("display")
    wname = title
    cv2.namedWindow(wname)
    delay = compute_wait(fps)
    x=0
    while x<len(all_frames):
        cv2.imshow(wname,all_frames[x])
        cv2.waitKey(delay)
        x = x + 1
    cv2.destroyAllWindows()
    return 0

def save_frames(all_frames, output_dir, basename, fps=30):
    foldername = basename+ "_" +str(fps)
    path = os.path.join(output_dir,foldername)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    x=0
    while x < len(all_frames):
        out = os.path.join(path,("image_%07d"%x + ".png"))
        curFrame=all_frames[x]
        cv2.imwrite(out, curFrame)
        x = x+1
    return 0

def change_fps(input_frames, old_fps, new_fps):
    
    return 0

def main():
    load_video_as_frames("noice.mp4")
    return 0
