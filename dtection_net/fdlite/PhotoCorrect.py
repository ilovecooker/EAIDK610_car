import cv2 
import numpy as np 

def image_white_balance(img):
    dst = np.zeros(img.shape)
    # 1、计算三通道灰度平均值
    Bavg = img[..., 0].mean()
    Gavg = img[..., 1].mean()
    Ravg = img[..., 2].mean()
    aveGray = (int)(Bavg + Gavg + Ravg) / 3
 
    # 2、计算每个通道的增益系数
    bCoef = aveGray / Bavg
    gCoef = aveGray / Gavg
    rCoef = aveGray / Ravg
 
    # 3使用增益系数
    dst[..., 0] = np.floor((img[..., 0] * bCoef))  # 向下取整
    dst[..., 1] = np.floor((img[..., 1] * gCoef))  # 向下取整
    dst[..., 2] = np.floor((img[..., 2] * rCoef))  # 向下取整
 
    # 4将数组元素后处理
    dst = np.clip(dst, 0, 255)
    dst = np.uint8(dst)
    return dst


def undistort_img():
    # Prepare object points 0,0,0 ... 8,5,0
    obj_pts = np.zeros((6*9,3), np.float32)
    obj_pts[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    # Stores all object points & img points from all images
    objpoints = []
    imgpoints = []
    # Get directory for all calibration images
    images = glob.glob('camera_cal/*.jpg')
    for indx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)
    # Test undistortion on img
    img_size = (img.shape[1], img.shape[0])
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Save camera calibration for later use
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump( dist_pickle, open('camera_cal/cal_pickle.p', 'wb') )
def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    #cv2.imwrite('camera_cal/test_cal.jpg', dst)
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst
