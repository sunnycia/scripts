import cv2

def SI(cur_frame):
    scale = 1
    delta = 0
    ddepth=cv2.CV_16S
    grad_x = cv2.Sobel(cur_frame, ddepth, dx=1, dy=0, ksize=3, scale=scale, delta=delta);
    grad_y = cv2.Sobel(cur_frame, ddepth, dx=1, dy=0, ksize=3, scale=scale, delta=delta);

    abs_grad_x = cv2.convertScaleAbs(grad_x);
    abs_grad_y = cv2.convertScaleAbs(grad_y);

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    mean, std_dev = cv2.meanStdDev(grad)
    return std_dev[0]

def TI(prev_frame, cur_frame):
    diff_ = cur_frame - prev_frame
    mean, std_dev = cv2.meanStdDev(diff_)

    return std_dev

if __name__== '__main__':
    cur_img = cv2.imread('frame2.jpg', 0)
    prev_img = cv2.imread('frame1.jpg', 0)


    print SI(cur_img)
    print TI(prev_img, cur_img)
