import cv2


def grey_to_rgb(gray_img, color='r'):
    # print gray_img.shape
    # gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    # print gray_img.shape
    if color=='y':
        gray_img[..., 0]=0
    if color=='r':
        gray_img[..., 0]=0
        gray_img[..., 1]=0
    if color=='g':
        gray_img[..., 2]=0  
        gray_img[..., 0]=0  

    return gray_img
    


if __name__ =='__main__':
    gray_img = cv2.imread(img_path)
    rgb_img = grey_to_rgb(gray_img)
