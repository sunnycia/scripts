from math import ceil
import imageio
import cv2
import glob, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--videodir", type=str, required=True, help="Video directory")
parser.add_argument("--outputdir", type=str, required=True, help="Result output directory")
parser.add_argument("--exttype", type=str, required=True, help="(duplicate) or (blank)")
parser.add_argument("--extlength", type=int, required=True, help="Extend length")

args = parser.parse_args()

def extend_video_imageio(video_path, output_dir, type, ext_length):
    video_name = os.path.basename(video_path)
    print "Processing", video_name

    video_reader = imageio.get_reader(video_path)
    video_data = video_reader.get_meta_data()
    size = video_data["size"]
    fps = int(ceil(video_data["fps"]))
    nframes = video_data["nframes"]
    print "\t", size, fps, nframes;#exit()

    output_path = os.path.join(output_dir, video_name)
    video_writer = imageio.get_writer(output_path, fps=fps)

    im = None
    for im in video_reader:
        video_writer.append_data(im)

    data = None
    if type=='duplicate':
        data = im
    elif type=='blank':
        pass # not implement
    for i in range(ext_length):
        video_writer.append_data(data)

def extend_video_opencv(video_path, output_dir, type, ext_length):
    video_name = os.path.basename(video_path)
    print "Processing", video_name

    video_capture = cv2.VideoCapture(video_path)
    size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = ceil(video_capture.get(cv2.CAP_PROP_FPS))
    nframes = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print "\t", size, fps, nframes;#exit()

    output_path = os.path.join(output_dir, video_name)
    codec = cv2.VideoWriter_fourcc('D','I','V','X')
    video_writer = cv2.VideoWriter(output_path, codec, fps, size)
    
    status, frame = video_capture.read()
    while(status):
        video_writer.write(frame)
        last_frame = frame
        status, frame = video_capture.read()

    data = None
    if type=='duplicate':
        data = last_frame
    elif type=='blank':
        # not implement
        return NotImplemented 
    for i in range(ext_length):
        video_writer.write(data)

    video_capture.release()
video_dir = args.videodir
output_dir = args.outputdir
ext_type = args.exttype
ext_length= args.extlength
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

video_path_list = glob.glob(os.path.join(video_dir, "*.*"))

for video_path in video_path_list:
    extend_video_opencv(video_path, output_dir, ext_type, ext_length)