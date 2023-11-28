from PIL import Image
import cv2
import numpy as np
import os
import scipy.io as scio

def getSample(test, gt):
    test_sample = []
    sample = []
    for img in test:
        test_sample.append(os.path.basename(img).replace('jpg', 'png'))
    print(test_sample)
    for img in gt:
        if os.path.basename(img) in test_sample:
            sample.append(os.path.basename(img))
    return sample

def getMask(img, threshold = 128, size = (256, 256)):
    gray_image = cv2.imread(img, 0)
    # print(gray_image.shape)
    # print(gray_image.shape == (256, 256))
    if gray_image.shape != size:
        gray_image = cv2.resize(gray_image, size)
        # print(gray_image.shape)
    mask = np.zeros_like(gray_image)
    mask[gray_image < threshold] = 1
    # print(mask)

    return mask

def CalIoU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    # print(intersection)
    iou = intersection / union

    return iou

def resize(img, size=(256, 256)):
    # 打开原始图像
    image = Image.open(img)

    # 缩放图像
    resized_image = image.resize(size)

    # 显示缩放后的图像
    # resized_image.show()

    # 保存缩放后的图像
    path = '/data/shijianyang/data/wifi/epr_256_gray'
    if not os.path.exists(path):
        os.makedirs(path)
    img_path = os.path.join(path, os.path.basename(img))
    resized_image.save(img_path)

# def getMat(path, key):
#     path = '/data/shijianyang/data/wifi/epr_I/2_rectangle_result1_epr_l.mat'
#     matdata = scio.loadmat(path)[key]
#     return mat_mask

if __name__ == '__main__':
    file_dir = '/data/shijianyang/data/wifi/data4psp/test/rgb_epsono'
    test_dir = '/data/shijianyang/code/psp4wifi/test_all/150000/inference_results'
    mat_dir = '/data/shijianyang/data/wifi/epr_gray'
    test60_dir = '/data/shijianyang/data/wifi/epr_mask'
    img_path_list = []
    test_path_list = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            img_path_list.append(os.path.join(root, file))
            # print(os.path.join(root, file))
    # print(len(img_path_list))
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            test_path_list.append(os.path.join(root, file))
    # sample = getSample(test_path_list, img_path_list)
    # print(len(test_path_list))

    # 计算IoU
    total_iou = 0
    n = 0
    for file in test_path_list:
        # print(file)
        test_mask = getMask(file)
        # print(test_mask.shape)
        gt = os.path.join(file_dir, os.path.basename(file).replace('jpg', 'png'))
        # gt = os.path.join(file_dir, os.path.basename(file))
        gt_mask = getMask(gt)
        # if 'triangle_' in os.path.basename(file):
        iou = CalIoU(test_mask, gt_mask)
        total_iou = total_iou + iou
        n = n + 1
        print('the IoU of {}th sample: {}'.format(n, iou))
    print('Average IoU: {}'.format(total_iou / n))

    # resize
    # for root, dirs, files in os.walk(mat_dir):
    #     for file in files:
    #         resize(os.path.join(root, file))
