###########################
# Christopher G. Mavros   #
# Fall 24 | CS548         #
# Assignment 1            #
###########################

#imports
import cv2
import sys
import numpy as np
from pathlib import Path
import os
import shutil

#function to pull frames from provided video file
def load_video_as_frames(video_filepath):
    video = cv2.VideoCapture(video_filepath)
    if not video.isOpened():
        print("ERROR: no video detected")
        return None
    #list of frames
    frames = []
    #iterate until end of video
    # put each frame into list    
    key = -1
    while key == -1:
        ret, frame = video.read()
        if ret == True:
            frames.append(frame)
        else:
            break
    video.release()
    return frames #end of load_video_as_frames

#determine the delay between frames for a given fps
def compute_wait(fps):
    #wait = (1000/fps)
    #wait = int(wait)
    #print("wait time:" + str(wait))
    #print("FPS: " + str(fps))
    return int(1000/fps) #end of compute_wait

#display the provided frames at a given fps
def display_frames(all_frames, title, fps=30):
    #create window and get delay between frames
    wname = title
    cv2.namedWindow(wname)
    delay = compute_wait(fps)
    #loop through the provided frames, displaying each one in the window
    x=0
    while x<len(all_frames):
        cv2.imshow(wname,all_frames[x])
        cv2.waitKey(delay)
        x = x + 1
    #destroy the window when done
    cv2.destroyAllWindows()
    return 0 #end of display_frames

#save each frame as a png in a specified directory
def save_frames(all_frames, output_dir, basename, fps=30):
    #determine the file paths
    foldername = basename+ "_" +str(fps)
    path = os.path.join(output_dir,foldername)
    #if the path already exists, delete it and remake it
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    #loop through the frames, saving each one at the file path
    x=0
    while x < len(all_frames):
        out = os.path.join(path,("image_%07d"%x + ".png"))
        curFrame=all_frames[x]
        cv2.imwrite(out, curFrame)
        x = x+1
    return 0 #end of save_frames

#change the fps of the video by removing/adding frames as appropriate
def change_fps(input_frames, old_fps, new_fps):
    #determine the frame counts
    old_frame_cnt = len(input_frames)
    new_frame_cnt = int(int(new_fps*old_frame_cnt)/(old_fps))
    output_frames = []
    #add the correct frames to output_frames
    x=0
    while x<new_frame_cnt:
        y=int((old_fps*x)/new_fps)
        output_frames.append(input_frames[y])
        x=x+1
    return output_frames #end of change_fps

#main function, takes up to 3 args
#1- input filepath
#2- output directory path
#3- (OPTIONAL) new FPS of file
def main():
    #determine number of args provided
    if (len(sys.argv)<3):
        print("ERROR: invalid args")
        exit(1)
    #store the correct filepaths
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    filename = Path(input_path).stem
    #determine if FPS was provided, defualt to 30 if it wasnt
    fps=30
    if (len(sys.argv)>3):
        fps = int(sys.argv[3])
    #convert to video to frames, call functions as outlined
    vFrames = load_video_as_frames(input_path)
    if vFrames == None:
        print("ERROR: no frames")
        exit(1)
    display_frames(vFrames, "Input Video", 30)
    outFrames = change_fps(vFrames,30,fps)
    display_frames(outFrames, "Output Video", 30)
    save_frames(outFrames, output_path, filename, fps)
    print("JOB DONE!")
    return 0 #end of main

if __name__=="__main__": 
    main()
#EOF