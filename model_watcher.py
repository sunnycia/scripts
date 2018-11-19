import time
import os


model_path = '/data/sunnycia/saliency_on_videoset/Train/training_output/salicon/vo-v4-2-trimmed_densenet-deconvBN-l1loss-dropout-base_lr-0.01-snapshot-4000-dense3d-new-batch-2_1529414170/snapshot-_iter_48000.caffemodel'
deploy_path = 'prototxt/vo-v4-2-trimmed_densenet-deconvBN-l1loss-deploy.prototxt'

cmd = "python test_video.py --video_deploy_path='%s' --video_model_path='%s' --infertype='slide' --output_type='image' --test_base='videoset' --model_code='v4-2' --videolength=16" %( deploy_path, model_path)
while True:
    if os.path.isfile(model_path):
        os.system(cmd)
        exit()
    time.sleep(5)
