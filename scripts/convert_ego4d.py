import torch
import numpy as np
import os
import subprocess
import glob as glob
import random

base_dir = '/user/work/tp8961/ego4d_data/v2/video_540ss'
new_dir = '/user/work/tp8961/ego4d_data/v2/video_384_30fps_300s'

def extract_single_video(vid_name):
    # create new folder
    out_folder = os.path.join(new_dir, vid_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    else:
        return

    try:
        call_string = f'ffmpeg -i {os.path.join(base_dir, vid_name)} -vf scale=-2:384 -q:v 5 -r 30 -f segment -segment_time 05:00 -force_key_frames "expr: gte(t, n_forced * 300)" -reset_timestamps 1 {out_folder}/%d.mp4'
        subprocess.run(call_string, shell=True, check=True)

        # remove original video file
        os.remove(os.path.join(base_dir, vid_name))

        print('Done with {}'.format(vid_name))

    except subprocess.CalledProcessError as e:
        print(e.output)
        os.rmdir(out_folder)

# get list of videos
orig_videos = glob.glob(os.path.join(base_dir, '*.mp4'))
# shuffle orig_videos
random.shuffle(orig_videos)

for vid in orig_videos:
    vid_name = vid.split('/')[-1]
    extract_single_video(vid_name)


