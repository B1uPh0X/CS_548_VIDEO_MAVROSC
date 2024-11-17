from General_A03 import *
import cv2
import numpy as np
from skimage.segmentation import *

def compute_optical_flow(video_frames):
    # Compute dense optical flow using Farneback method
    all_flow, prev_frame = [], None
    for frame in video_frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is None:
            prev_frame = gray_frame
            continue
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        all_flow.append(np.stack([mag / 20.0] * 3, axis=-1))
        prev_frame = gray_frame
    return all_flow

def cluster_colors(image, k):
    # Cluster image colors using K-means
    samples = image.reshape((-1, 3)).astype("float32")
    _, labels, centers = cv2.kmeans(
        samples, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 70, 0.1),
        3, cv2.KMEANS_RANDOM_CENTERS)
    recolored = centers[labels.flatten()].reshape(image.shape)
    return cv2.convertScaleAbs(recolored), labels, centers

def scale_box(box, psy, psx, ph, pw):
    # Scale a bounding box
    ymin, xmin, ymax, xmax = box
    height, width = ymax - ymin, xmax - xmin
    ymin, xmin = int(ymin + psy * height), int(xmin + psx * width)
    ymax, xmax = int(ymin + height * ph), int(xmin + width * pw)
    return ymin, xmin, ymax, xmax

def get_histogram(image, channels, ranges, bins):
    # compute a histogram for specified channels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], channels, None, bins, ranges)
    return hist / hist.sum()

def draw_box(frame, box, color=(0, 255, 0), thickness=2):
    #Draw a bounding box on the frame
    ymin, xmin, ymax, xmax = box
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)

def update_box(box, keypoints, box_size=(50, 50)):
    #Update the bounding box to fit keypoints.
    height, width = box_size
    if not keypoints:
        return box
       #height = int((box[2] - box[0]))
       #width = int((box[3] - box[1]))
       ##center_y = (box[0] + box[2])//2
    else:
        x_coords = [kp.pt[0] for kp in keypoints]
        y_coords = [kp.pt[1] for kp in keypoints]
        center_x, center_y = int(np.mean(x_coords)), int(np.mean(y_coords))
  
    ymin = max(0,center_y - height//2)
    xmin = max(0, center_x - width//2)
    ymax = center_y + height//2
    xmax = center_x + width//2
    return ymin, xmin, ymax, xmax

def track_doggo(video_frames, first_box):
    #Track the doggo across video frames
    orb = cv2.ORB_create(nfeatures=1000)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    cur_box = first_box
    bh = int((cur_box[2] - cur_box[0])*.85)
    bw = int((cur_box[3] - cur_box[1])*.85)
    cur_box[0]+=35
    cur_box[2]+=35
    cur_box[3]+=5
    cur_box[1]+=5

    keypoints, descriptors = orb.detectAndCompute(video_frames[0], None)
    bound_boxes = []

    for index, frame in enumerate(video_frames):
        bound_boxes.append(cur_box)
        kp, des = orb.detectAndCompute(frame, None)
        matches = bf_matcher.match(descriptors, des)
        # Filter keypoints inside the current bounding box
        ymin, xmin, ymax, xmax = cur_box
        keypoints_in_box = [kp for kp in kp if xmin <= kp.pt[0] <= xmax and ymin <= kp.pt[1] <= ymax]
        # Update bounding box to follow keypoints
        # different box sizes work better for different videos
        # the starting box size works well for most of them
        # for the ones that it dosent, a smaller box is typically better
        cur_box = update_box(cur_box, keypoints_in_box,box_size=(bh,bw))
        # Draw matches
        match_frame = cv2.drawMatches(video_frames[0], keypoints, frame, kp, matches, None)
        cv2.imshow("Matches", match_frame)
        # Draw the current bounding box
        draw_box(frame, cur_box, color=(255, 0, 0), thickness=2)
        cv2.imshow("Tracking", frame)
        # Process subimage
        ymin, xmin, ymax, xmax = cur_box
        subimage = frame[ymin:ymax, xmin:xmax]
        slic_segments = slic(subimage, start_label=0)
        boundary_image = mark_boundaries(subimage, slic_segments)
        cv2.imshow("Segmented", boundary_image)

        # Update bounding box (Optional: Add transformation logic)

        if cv2.waitKey(0) == 27:  # ESC to quit
            break
    return bound_boxes
def main():
    # Main function
    dog_index, max_images, start_index = 7, 60, 30
    dog_images, dog_boxes, _ = load_dog_video(dog_index, max_images_to_load=max_images, starting_frame_index=start_index)
    track_doggo(dog_images, dog_boxes[0])

if __name__ == "__main__":
    main()
