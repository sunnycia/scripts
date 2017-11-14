import imageio
import os, glob
import cv2
import numpy as np
def blend_video(video_path1, video_path2, output_path):
    ## check if exists


    video_reader1 = imageio.get_reader(video_path1)
    video_reader2 = imageio.get_reader(video_path2)

    fps = video_reader1.get_meta_data()['fps']
    assert fps == video_reader2.get_meta_data()['fps']

    video_writer = imageio.get_writer(output_path, fps=fps)

    index = 0
    for frame1, frame2 in zip(video_reader1, video_reader2):
        frame1 = np.array(frame1)
        frame2 = np.array(frame2)
        # print type(frame1)
        if not frame1.shape == frame2.shape:
            h, w, _ = frame1.shape
            frame2 = cv2.resize(frame2, dsize=(w,h))
        # print frame1.shape, frame2.shape
        print "Processing frame", str(index + 1), "of video", os.path.basename(video_path1)
        blend_frame = cv2.addWeighted(frame1, 0.2, frame2, 0.8, 0)
        video_writer.append_data(blend_frame)
        index += 1

if __name__ == "__main__":
    ori_video_dir = "/data/sunnycia/SaliencyDataset/Video/MSU/videos"
    sal_video_dir = "/data/sunnycia/SaliencyDataset/Video/MSU/saliency_video"
    output_dir = "/data/sunnycia/SaliencyDataset/Video/MSU/blend"

    sal_video_path_list = glob.glob(os.path.join(sal_video_dir, "*.*"))
    for sal_video_path in sal_video_path_list:
        video_name = os.path.basename(sal_video_path)
        video_path1 = os.path.join(ori_video_dir,video_name)
        video_path2 = sal_video_path
        output_path = os.path.join(output_dir, video_name)
        if os.path.isfile(output_path):
            print output_path, "already exists, pass..."
            continue

        blend_video(video_path1, video_path2, output_path)