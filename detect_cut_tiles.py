import argparse
import time
import os
import json
from tqdm import tqdm
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random, ascontiguousarray

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def cropImage(img, xyxy, pad=0):
    """crop sub image from original image by xyxy

    Args:
        img: numpy array with cv2.imread of (h, w, c) 
        xyxy: absolute coordinates with [x, y, x, y]
        pad: both left and right padding with int, default=0

    Returns:
        ndarray: sub image's numpy array with (h, w, c)
        list: offsets of coordinates
    """    
    h, w, _ = img.shape
    if xyxy[0] - pad >= 0:
        xyxy[0] -= pad
    if xyxy[2] + pad <= w:
        xyxy[2] += pad
    new_img = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :]
    return new_img, [xyxy[0], xyxy[1]]


def split_image(image, cut_shape, overlap):
    """split original image into sub images with cut shape and overlap

    Args:
        image (ndarray): original image read by cv2.imread() with (h, w, c)
        cut_shape (tuple[int]): cut shape with (cut_height, cut_width)
        overlap (int): overlap between cut images

    Returns:
        list: cut image list with [image, image, ...] 
    """    
    ori_h, ori_w, _ = image.shape
    cut_h, cut_w = cut_shape
    num_h, num_w = 0, 0

    cut_image_list = []
    offsets_list = []

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
        offsets_list.append([left_pad, top_pad])
    
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
                
                cut_img, offsets = cropImage(image, cut_xyxy)
                cut_image_list.append(cut_img)
                offsets_list.append(offsets)

    return cut_image_list, offsets_list


def cut_transform_image(image, cut_size, overlap):
    cut_image_list, offsets_list = split_image(image, (cut_size, cut_size), overlap)
    image_list = []
    for img in cut_image_list:
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = ascontiguousarray(img)

        img = torch.from_numpy(img)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        image_list.append(img)

    image_batch = torch.stack(image_list, dim=0)
    return image_batch, offsets_list


def detect():
    source, weights, imgsz, cut_size, overlap = opt.source, opt.weights, opt.img_size, opt.cut_size, opt.overlap
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #     ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Second-stage classifier
    # classify = False
    # if classify:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    # vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = True
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz)
    # else:
    #     save_img = True
    #     dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    # t0 = time.time()
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model(img) if device.type != 'cpu' else None  # run once

    final_image_names = []
    final_results = []

    image_name_list = os.listdir(source)
    image_name_list = sorted(image_name_list)
    for image_name in tqdm(image_name_list):
        image_path = os.path.join(source, image_name)
        ori_image = cv2.imread(image_path)
        image_batch, offsets_list = cut_transform_image(ori_image, cut_size, overlap)
        image_batch = image_batch.to(device)

        # Inference
        # t1 = time_synchronized()
        pred = model(image_batch, augment=opt.augment)[0]

        num_classes = int(pred.shape[-1]) - 5

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes)
        # t2 = time_synchronized()

        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        # for i, det in enumerate(pred):  # detections per image
        #     p, s, im0 = path, '', im0s

        #     p = Path(p)  # to Path
        #     save_path = str(save_dir / p.name)  # img.jpg
        #     # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        #     s += '%gx%g ' % img.shape[2:]  # print string
        #     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f'{n} {names[int(c)]}s, '  # add to string

                # # Write results
                # for *xyxy, conf, cls in reversed(det):
                #     if save_img or view_img:  # Add bbox to image
                #         label = f'{names[int(cls)]} {conf:.2f}'
                #         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # # Stream results
            # if view_img:
            #     cv2.imshow(str(p), im0)

            # # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
        
        pred_results = []
        for idx, image_pred in enumerate(pred):
            if image_pred.shape[0] == 0 or image_pred == None:
                continue
            else:
                for det in image_pred:
                    # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    det = det.reshape(1, 6)
                    for *xyxy, conf, cls in det:
                        cls = int(cls)
                        result_part1 = [int(xyxy[0] + offsets_list[idx][0]), int(xyxy[1] + offsets_list[idx][1]), 
                                  int(xyxy[2] + offsets_list[idx][0]), int(xyxy[3] + offsets_list[idx][1]),
                                  conf.item()]
                        result_part2 = [0] * num_classes
                        result_part2[cls] = 1
                        result = result_part1 + result_part2
                        pred_results.append(torch.Tensor(result))

        if pred_results != []:
            pred_results = torch.stack(pred_results, dim=0)
        else:
            continue
        
        if opt.final_nms:
            pred_results = non_max_suppression(pred_results.unsqueeze(0), opt.conf_thres, opt.iou_thres, classes=opt.classes)
            final_results.append(pred_results[0])
            final_image_names.append(image_name)
        else:
            # TODO: need to be completed
            final_results.append(pred_results)


    save_results(final_image_names, final_results)

    # print(f'Done. ({time.time() - t0:.3f}s)')


def save_results(image_name_list, pred_results):
    result_json = []
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    for image_name, pred in zip(image_name_list, pred_results):
        if pred == None or pred.shape[0] == 0:
            continue
        for bbox_pred in pred:
            # bbox_pred = bbox_pred.reshape(1, 6)
            bbox_pred = bbox_pred.cpu().numpy()
            cls = int(bbox_pred[5]) + 1
            # cls = int(bbox_pred[5])
            bbox_xyxy = [int(val) for val in bbox_pred[:4]]
            conf = float(bbox_pred[4])
            result_json.append({'name': image_name, 'category': cls, 'bbox': bbox_xyxy, 'score': conf})
    
    with open(os.path.join(opt.save_path, 'results.json'), 'w') as fp:
        json.dump(result_json, fp, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='../data/tile_round1_testA_20201231/testA_imgs', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--cut-size', type=int, default=640, help='cut image size (pixels)')
    parser.add_argument('--overlap', type=int, default=640 // 5, help='overlap when cutting image')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--final-nms', default='True', help='final nms at last for whole image')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save-path', type=str, default='results', help='path to save json result')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()


    detect()

    print('---- Finished! ----')
