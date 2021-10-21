# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
import sys
from pathlib import Path
from mss import mss
import cv2
import numpy as np
import torch
import pyautogui

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, check_suffix, non_max_suppression, scale_coords, set_logging
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)    训练的权重
        imgsz=[640, 640],  # inference size (pixels) 网络输入图片大小
        conf_thres=0.25,  # confidence threshold 置信度阈值
        iou_thres=0.45,  # NMS IOU threshold nms的iou阈值
        max_det=1000,  # maximum detections per image 分类数
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu 设备
        view_img=True,  # show results 是否展示预测之后的图片/视频
        classes=None,  # filter by class: --class 0, or --class 0 2 3 设置只保留某一部分类别
        agnostic_nms=False,  # class-agnostic NMS 进行nms是否也去除不同类别之间的框
        augment=False,  # augmented inference 图像增强
        visualize=False,  # visualize features 可视化
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model 加载float32模型，确保图片分辨率能整除32
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        #设置Float16
        if half:
            model.half()  # to FP16
        # 设置2次分类
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    # 图片或视频
    tmp = False
    tmp2 = False
    mon = {'top': 0, 'left': 0, 'width': 960, 'height': 960}

    while True:
        im = np.array(mss().grab(mon))
        screen = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        dataset = LoadImages(screen, img_size=imgsz, stride=stride, auto=pt)
        dt, seen = [0.0, 0.0, 0.0], 0
        '''
        path 图片/视频路径
        img 进行resize+pad之后的图片，如(3,640,512) 格式(c,h,w)
        img0s 原size图片，如(1080,810,3)
        cap 当读取图片时为None,读取视频时为视频源
        '''

        for img, im0s, vid_cap in dataset:
            t1 = time_sync()
            if onnx:
                img = img.astype('float32')
            else:
                img = torch.from_numpy(img).to(device)
                # print(img)
                # 图片也设置为Float16或者32
                img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            # 没有batch_size时，在最前面添加一个轴
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            if pt:
                '''
                前向传播，返回pred的shape是(1,num_boxes,5+num_class)
                h,w为传入网络图片的高和宽，注意dataset在检测时使用了矩形推理，所以h不一定等于w
                num_boxes = (h/32*w/32+h/16*w/16+h/8*w/8)*3
                例如：图片大小720，1280 -> 15120个boxes = (20*12 + 40*24 + 80*48 = 5040)*3
                pred[...,0:4]为预测框坐标；预测框坐标为xywh
                pred[...,4]为objectness置信度
                pred[...,5:-1]为分类结果
                '''
                pred = model(img, augment=augment, visualize=visualize)[0]

            else:  # tensorflow model (tflite, pb, saved_model)
                imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
                if pb:
                    pred = frozen_func(x=tf.constant(imn)).numpy()
                elif saved_model:
                    pred = model(imn, training=False).numpy()
                elif tflite:
                    if int8:
                        scale, zero_point = input_details[0]['quantization']
                        imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                    interpreter.set_tensor(input_details[0]['index'], imn)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]['index'])
                    if int8:
                        scale, zero_point = output_details[0]['quantization']
                        pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
                pred[..., 0] *= imgsz[1]  # x
                pred[..., 1] *= imgsz[0]  # y
                pred[..., 2] *= imgsz[1]  # w
                pred[..., 3] *= imgsz[0]  # h
                pred = torch.tensor(pred)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            '''
            pred:前向传播的输出
            conf_thres:置信度阈值
            iou_thres:iou阈值
            classes：是否只保留特定的类别
            agnostic_nmsL进行nms是否也去除不同类别之间的框
            经过nms后预测框格式，xywh->xyxy(左上角右上角)
            pred是一个列表list[torch.tensor],长度为nms后目标框个数
            每一个torch.tensor的shape为(num_boxes,6),内容为box(4个值)+cunf+cls
            '''
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # 添加二级分类，默认false
            # if classify:
            #     pred = apply_classifier(pred, modelc, img, im0s)

            # Process predictions
            # 对每一张图片处理
            for i, det in enumerate(pred):  # per image
                seen += 1
                s, im0 = '', im0s.copy()
                # 设置打印信息（图片宽高），s如'640*512'
                s += '%gx%g ' % img.shape[2:]  # print string
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # 调整预测框坐标，基于resize+pad的图片坐标->基于原size图片坐标
                    # 此时坐标格式为xyxy
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # 打印检测到的类别数量
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    # 保存预测结果
                    for *xyxy, conf, cls in reversed(det):
                        if view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                # Stream results
                im0 = annotator.result()
                cv2.imshow('a crop of the screen', im0)
                cv2.moveWindow('a crop of the screen', 960, 0)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    tmp = True
                    break
            if tmp:
                tmp2 = True
                break
        if tmp2:
            break

if __name__ == "__main__":
    run()
