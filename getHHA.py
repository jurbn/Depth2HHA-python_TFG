# --*-- coding:utf-8 --*--
import math
import cv2
import os
import math

from src.depth2hha.utils.rgbd_util import *
from src.depth2hha.utils.getCameraParam import *

'''
must use 'COLOR_BGR2GRAY' here, or you will get a different gray-value with what MATLAB gets.
'''
def getImage(root='demo'):
    D = cv2.imread(os.path.join(root, '0.png'), cv2.COLOR_BGR2GRAY)/10000
    RD = cv2.imread(os.path.join(root, '0_raw.png'), cv2.COLOR_BGR2GRAY)/10000
    return D, RD


'''
C: Camera matrix
D: Depth image, the unit of each element in it is "meter"
RD: Raw depth image, the unit of each element in it is "meter"
'''
def getHHA(C, D):
    depth_meters = D.astype(np.float32) / 1000.0  # Assuming input is in millimeters
    missingMask = (depth_meters == 0)

    # Process depth image to obtain required features
    pc, N, yDir, h, pcRot, NRot = processDepthImage(depth_meters * 100, missingMask, C)

    # Calculate angle using dot product
    tmp = np.multiply(N, yDir)
    acosValue = np.minimum(1, np.maximum(-1, np.sum(tmp, axis=2)))
    angle = np.degrees(np.arccos(acosValue))
    angle[np.isnan(angle)] = 180  # Handle NaN values

    # Prepare HHA components
    pc[:, :, 2] = np.clip(pc[:, :, 2], 0.1, None)  # Prevent division by zero
    I = np.zeros_like(pc, dtype=np.float32)

    # Assign components to channels
    I[:, :, 0] = normalize_channel(angle)  # Red
    I[:, :, 1] = normalize_channel(h)      # Green
    I[:, :, 2] = normalize_channel(31000 / pc[:, :, 2])  # Blue


    # Ensure valid pixel values and convert to uint16
    I = np.clip(I, 0, 65535).astype(np.uint16)

    return I

def normalize_channel(channel, low=2, high=98):
    p_low, p_high = np.percentile(channel, [low, high])
    channel = np.clip(channel, p_low, p_high)
    channel = (channel - p_low) / (p_high - p_low)
    return channel * 65535  # Scale to uint16 range

if __name__ == "__main__":
    D, RD = getImage()
    camera_matrix = getCameraParam('color')
    print('max gray value: ', np.max(D))        # make sure that the image is in 'meter'
    hha = getHHA(camera_matrix, D, RD)
    hha_complete = getHHA(camera_matrix, D, D)
    cv2.imwrite('demo/hha.png', hha)
    cv2.imwrite('demo/hha_complete.png', hha_complete)
    
    
    ''' multi-peocessing example '''
    '''
    from multiprocessing import Pool
    
    def generate_hha(i):
        # generate hha for the i-th image
        return
    
    processNum = 16
    pool = Pool(processNum)

    for i in range(img_num):
        print(i)
        pool.apply_async(generate_hha, args=(i,))
        pool.close()
        pool.join()
    ''' 
