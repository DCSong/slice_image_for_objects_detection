import os
import cv2

'''
{  
    "0": "背景",  
    "1": "边异常",  
    "2": "角异常",  
    "3": "白色点瑕疵",  
    "4": "浅色块瑕疵",  
    "5": "深色点块瑕疵",  
    "6": "光圈瑕疵"  
}  
'''

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = [0, 0, 0, 0]
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = [0, 0, 0, 0]
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y


def cropImage(img, xyxy, pad=0):
    """crop sub image from original image by xyxy

    Args:
        img: numpy array with cv2.imread of (h, w, c) 
        xyxy: absolute coordinates with [x, y, x, y]
        pad: both left and right padding with int, default=0

    Returns:
        ndarray: sub image's numpy array with (h, w, c)
    """    
    h, w, _ = img.shape
    if xyxy[0] - pad >= 0:
        xyxy[0] -= pad
    if xyxy[2] + pad <= w:
        xyxy[2] += pad
    new_img = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
    return new_img


def transformLabel(cut_xyxy, labels, ori_shape):
    """transform original labels to cut image labels

    Args:
        cut_xyxy (list[int]): absolute coordinates with [x, y, x, y]
        labels (list): original txt labels with [[cls, x, y, w, h], ...]
        ori_shape (tuple(int)): original shape of image with (original_height, original_width)

    Returns:
        list: new label list [(cls, xywh), ...] which is in cut images
        list: new cut xyxy which has been extended
    """    
    ori_h, ori_w = ori_shape
    cut_h, cut_w = cut_xyxy[3] - cut_xyxy[1], cut_xyxy[2] - cut_xyxy[0]

    # to ensure new_cut_xyxy by traverse labels
    new_cut_xyxy = cut_xyxy
    for label in labels:
        center_x, center_y = label[1] * ori_w, label[2] * ori_h
        ori_xyxy = xywh2xyxy(label[1:])
        ori_xyxy = [int(val * factor) for val, factor in zip(ori_xyxy, [ori_w, ori_h, ori_w, ori_h])]
        if ori_xyxy[0] >= cut_xyxy[0] and ori_xyxy[1] >= cut_xyxy[1] and ori_xyxy[2] <= cut_xyxy[2] and ori_xyxy[3] <= cut_xyxy[3]:
            continue
        elif center_x >= cut_xyxy[0] and center_x <= cut_xyxy[2] and center_y >= cut_xyxy[1] and center_y <= cut_xyxy[3]:
            if ori_xyxy[0] < cut_xyxy[0]:
                new_cut_xyxy[0] = ori_xyxy[0]
            if ori_xyxy[1] < cut_xyxy[1]:
                new_cut_xyxy[1] = ori_xyxy[1]
            if ori_xyxy[2] > cut_xyxy[2]:
                new_cut_xyxy[2] = ori_xyxy[2]
            if ori_xyxy[3] > cut_xyxy[3]:
                new_cut_xyxy[3] = ori_xyxy[3]
    
    new_cut_w = new_cut_xyxy[2] - new_cut_xyxy[0]
    new_cut_h = new_cut_xyxy[3] - new_cut_xyxy[1]
    
    # reset all labels which in the cut zone
    reset_labels = []
    for label in labels:
        # label [cls, x, y, w, h]
        cls = int(label[0])
        center_x, center_y = label[1] * ori_w, label[2] * ori_h
        ori_xyxy = xywh2xyxy(label[1:])
        ori_xyxy = [int(val * factor) for val, factor in zip(ori_xyxy, [ori_w, ori_h, ori_w, ori_h])]
        # maybe just judge it by center point (x, y)
        if ori_xyxy[0] >= new_cut_xyxy[0] and ori_xyxy[1] >= new_cut_xyxy[1] and ori_xyxy[2] <= new_cut_xyxy[2] and ori_xyxy[3] <= cut_xyxy[3]:
            new_bbox_w = label[3] * ori_w / new_cut_w
            new_bbox_h = label[4] * ori_h / new_cut_h
            new_bbox_x = (label[1] * ori_w - new_cut_xyxy[0]) / new_cut_w
            new_bbox_y = (label[2] * ori_h - new_cut_xyxy[1]) / new_cut_h
            reset_labels.append([cls, new_bbox_x, new_bbox_y, new_bbox_w, new_bbox_h])

    return reset_labels, new_cut_xyxy


def split_image(image, label_list, cut_shape, overlap, dropSubBackground=False):
    """split original image into sub images and reset labels with cut shape and overlap

    Args:
        image (ndarray): original image read by cv2.imread() with (h, w, c)
        label_list (list): original txt labels with [[cls, x, y, w, h], ...]
        cut_shape (tuple[int]): cut shape with (cut_height, cut_width)
        overlap (int): overlap between cut images

    Returns:
        list: cut image list with [image, image, ...] 
        list: cut label list corresponding to cut image with [[[cls, xywh], ...], ...]
    """    
    ori_h, ori_w, _ = image.shape
    cut_h, cut_w = cut_shape
    num_h, num_w = 0, 0

    cut_image_list, cut_label_list = [], []

    if cut_h >= ori_h and cut_w >= ori_w:
        # a special situation that may not exists
        # if original shape less than cut shape, pad it into cut shape and return
        top_pad, bottom_pad = 0, 0
        left_pad, right_pad = 0, 0
        top_pad = (cut_h - ori_h) // 2
        bottom_pad = cut_h - ori_h - top_pad
        left_pad = (cut_w - ori_w) // 2
        right_pad = cut_w - ori_w - left_pad

        image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cut_image_list.append(image)

        # it may be generated by 
        # transformLabel([-left_pad, -top_pad, ori_w + right_pad, ori_h + bottom_pad], label_list, (ori_h, ori_w))
        new_label_list = []
        for label in label_list:
            cls = int(label[0])
            new_bbox_w = label[3] * ori_w / cut_w
            new_bbox_h = label[4] * ori_h / cut_h
            new_bbox_x = (label[1] * ori_w + left_pad) / cut_w
            new_bbox_y = (label[2] * ori_h + top_pad) / cut_h
            new_label_list.append([cls, new_bbox_x, new_bbox_y, new_bbox_w, new_bbox_h])
        cut_label_list.append(new_label_list)  
    
    else:
        while num_w * (cut_w - overlap) + overlap < ori_w:
            num_w += 1
        while num_h * (cut_h - overlap) + overlap < ori_h:
            num_h += 1
        
        for i in range(num_h):
            for j in range(num_w):
                if i == num_h - 1:
                    if j == num_w - 1:
                        assert (i * (cut_h - overlap) < ori_h) and (i * (cut_h - overlap) + cut_h > ori_h), 'there are some errors in last row'
                        assert (j * (cut_w - overlap) < ori_w) and (j * (cut_w - overlap) + cut_w > ori_w), 'there are some errors in last column'
                        cut_xyxy = [ori_w - cut_w, ori_h - cut_h, ori_w, ori_h]
                    else:
                        assert (i * (cut_h - overlap) < ori_h) and (i * (cut_h - overlap) + cut_h > ori_h), 'there are some errors in last row'
                        cut_xyxy = [j * (cut_w - overlap), ori_h - cut_h, j * (cut_w - overlap) + cut_w, ori_h]
                
                else:
                    if j == num_w - 1:
                        assert (j * (cut_w - overlap) < ori_w) and (j * (cut_w - overlap) + cut_w > ori_w), 'there are some errors in last column'
                        cut_xyxy = [ori_w - cut_w, i * (cut_h - overlap), ori_w, i * (cut_h - overlap) + cut_h] 
                    else:
                        cut_xyxy = [j * (cut_w - overlap), i * (cut_h - overlap), j * (cut_w - overlap) + cut_w, i * (cut_h - overlap) + cut_h]
                
                cut_labels, new_cut_xyxy = transformLabel(cut_xyxy, label_list, (ori_h, ori_w))
                
                # ! dropout cut images without target
                if dropSubBackground and cut_labels == []:
                    continue
                
                cut_img = cropImage(image, new_cut_xyxy)
                cut_image_list.append(cut_img)
                cut_label_list.append(cut_labels)

    return cut_image_list, cut_label_list          


if __name__ == '__main__':
    # ! just change parameters below

    image_root_path = './data/tile_round1_train_20201231/train_imgs'
    label_root_path = './tile-defect-detect/data/train-txt-labels'

    image_save_path = './tile-defect-detect/yolov5/data/images'
    label_save_path = './tile-defect-detect/yolov5/data/labels'

    # h, w of every cut image 
    cut_h, cut_w = 640, 640
    overlap = 640 // 5
    
    # is or not dropout images without bbox (background totally)
    dropBackground = False

    # is or not dropout cut images without bbox (sub images without target)
    dropSubBackground = True

    # ! just change parameters above


    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)
    if not os.path.exists(label_save_path):
        os.mkdir(label_save_path)

    image_name_list = os.listdir(image_root_path)
    # label_name_list = os.listdir(label_root_path)

    for image_name in image_name_list:
        image = cv2.imread(os.path.join(image_root_path, image_name))

        label_path = os.path.join(label_root_path, image_name.replace('.jpg', '.txt'))
        with open(label_path, 'r') as f:
            # read txt label (cls, xywh)
            labels = f.readlines()
        
        # dropout images without target
        if dropBackground and (labels[0] == '\n' or labels[0] == ['']):
            print(f'have skip image {image_name}\n')
            continue
        
        label_list = []
        for label in labels:
            if label == '\n' or label == '':
                continue
            
            label = label.replace('\n', '').split(' ')
            label = list(map(eval, label))
            label_list.append(label)

        cut_image_list, cut_label_list = split_image(image, label_list, (cut_h, cut_w), overlap=overlap, dropSubBackground=dropSubBackground)
        
        for idx, (image, labels) in enumerate(zip(cut_image_list, cut_label_list)):
            save_image_name = image_name.replace('.jpg', '_') + str(idx) + '.jpg'
            cv2.imwrite(os.path.join(image_save_path, save_image_name), image)
            print(f'save image {save_image_name}')

            new_label = ''
            for label in labels:
                new_label += (' '.join(list(map(str, label))) + '\n')
            save_label_name = image_name.replace('.jpg', '_') + str(idx) + '.txt'
            with open(os.path.join(label_save_path, save_label_name), 'w') as f:
                f.writelines(new_label)
            print(f'save label {save_label_name}\n')


    print('---- Finished! ----')