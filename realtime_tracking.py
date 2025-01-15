import os
import time
import torch
import cv2
import matplotlib
import numpy as np
from math import *
from poseestimation.poseestimator import PoseEstimator


video_src = "Chores/krump.mp4"
inFolder = "Chores/"
outFolder = "Chores_seqs/" #"Choreos_seqs/"


def gcd_resize(input_frame, gcd):
    frame_height, frame_width = input_frame.shape[0], input_frame.shape[1]
    if gcd != 1:
        adapted_width = gcd * (frame_width // gcd)
        adapted_height = gcd * (frame_height // gcd)
        adapted_frame = cv2.resize(input_frame, (adapted_width, adapted_height))
    else:
        adapted_frame = input_frame
    return adapted_frame, (adapted_width, adapted_height), (frame_width, frame_height)

def generate_chore_seq(video_src, pose_estimator):
    with open(video_src + ".txt", "w") as out_file:   
        oldTime = time.time()
        cap = cv2.VideoCapture(video_src)

        if not cap.isOpened():
            print("Error opening video stream")
            return
        
        while True:
            ret, frame = cap.read()

            if not ret:
                break


            resized_frame, (h, w), (H, W) = gcd_resize(frame, 64)
            resized_frame = cv2.resize(frame, (256, 256))

            yol_frame, poses_results, _, _ = pose_estimator.detect(resized_frame)

            src_pts = np.ones((34, 2)) * -1
            for i, poses in enumerate(poses_results):

                if len(poses_results):  # check if no pose
                    for c in poses[:, 5].unique():  # Print poses_results
                        n = (poses[:, 5] == c).sum()  # detections per class

                        all_kpts = poses[:, 6:]

                        for i, pose_kpts in enumerate(all_kpts):

                            kpts_xy = pose_kpts.reshape(-1, 3)[:, :-1]
                            kpts_scores = pose_kpts.reshape(-1, 3)[:, -1]

                            for i, kpt_xy in enumerate(kpts_xy):                    
                                xy = kpt_xy.cpu().numpy()

                                src_pts[i] = np.ones((1, 2)) * 100
                                if (kpts_scores[i] > 0.8):
                                    src_pts[i] = xy
                                src_pts[i] = xy


                                cv2.circle(resized_frame, tuple(xy.astype(int)), 3, (0, 255, 0), -1)
                                cv2.putText(resized_frame, str(i), tuple(xy.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

                        src_pts1 = src_pts[~np.all(src_pts == [-1, -1], axis=1)]

                        # Save choreo file 
                        sep = ";"
                        line = "".join([str(coord) + sep for coord in src_pts1.astype(int)])
                        out_file.write(line + "\n")
                        
            dt = time.time() - oldTime
            oldTime = time.time()

            cv2.imshow("Video Stream", resized_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def get_file(inFolder, file_name):
    # Ignorer les fichiers ou dossiers invisibles
        if file_name.startswith("."):
            return None
        
        file_path = os.path.join(inFolder, file_name)
        
        # Vérifier si le fichier existe avant de le traiter
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return None
        return file_path
        



def main():

    print(f"PyTorch version: {torch.__version__}")

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. PyTorch is using GPU support.")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. PyTorch is using CPU only.")


    ################## Init pose estimator ###################
    pose_estimator = PoseEstimator("0", "poseestimation/weights/yolov7-w6-pose.pt")
    names = pose_estimator.names
    ###########################################################
    

    file_list = os.listdir(inFolder)
    for file_name in file_list:
        file_path = get_file(inFolder, file_name) 
        if file_path is None:
            continue
        filename = outFolder + os.path.splitext(os.path.basename(file_path))[0]
        generate_chore_seq(file_path, pose_estimator)


if __name__ == "__main__":
    main()
    matplotlib.use('TkAgg')


