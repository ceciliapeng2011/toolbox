import sys
import os
import numpy as np
import cv2
from mvnc import mvncapi as mvnc
import caffe
test_path = sys.argv[1]
#False for mvnc while True for caffe
caf = False
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
blob_file='./old.graph'
thresh = 0.001
thresh_iou = 0.5
if not caf:
    mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 0)
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        quit()
    device = mvnc.Device(devices[0])
    device.OpenDevice()
    # Load blob
    with open(blob_file, mode='rb') as f:
        blob = f.read()
    graph = device.AllocateGraph(blob)
if caf:
    caffe_proto = sys.argv[2]
    caffe_weights = sys.argv[3]
    # switch to caffe mode_gpu if you want
    caffe.set_mode_cpu()
    net = caffe.Net(caffe_proto, caffe_weights, caffe.TEST)

# read list of images
with open(test_path, 'r') as f:
    lines = f.readlines()
images = [x.strip() for x in lines]

for image_path in images:
    image_id = os.path.basename(image_path).split('.')[0]
    if not caf:
        img=cv2.imread(image_path)
        img_cv=img
        img = cv2.resize(img, (448, 448), cv2.INTER_LINEAR)
        img = img[:, :, ::-1]
        img=np.divide(img,255.0)
        graph.LoadTensor(img.astype(np.float16), image_id)
        output, userobj = graph.GetResult()
        result = output.astype(np.float32)
        img_width = img_cv.shape[1]
        img_height = img_cv.shape[0]
    if caf:
        img = caffe.io.load_image(image_path)
        img_width = img.shape[1]
        img_height = img.shape[0]
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        inputs = img
        result = net.forward_all(data=np.asarray([transformer.preprocess('data', inputs)]))
        result = result['fc9'][0]
    print(image_id, img_width, img_height)

    probs = np.reshape(result[0:980], (7,7,20))
    scale = np.reshape(result[980:1078], (7,7,2))
    boxes = np.reshape(result[1078:], (7,7,2,4))
    conf = np.zeros((7,7,2,24))
    for i in range(2):
        for j in range(20):
            conf[:,:,i,j] = np.multiply(probs[:,:,j],scale[:,:,i])
    for i in range(0, 7):
        for j in range(0, 7):
            for k in range(0, 2):
                box = boxes[j, i, k]
                box = [(box[0] + i) / 7 * img_width, (box[1] + j) / 7 * img_height, (box[2] * box[2]) * img_width, (box[3] * box[3]) * img_height]
                xmin = box[0] - box[2] / 2
                ymin = box[1] - box[3] / 2
                xmax = box[0] + box[2] / 2
                ymax = box[1] + box[3] / 2
                for c in range(0, 20):
                    if not conf[j, i, k, c] > thresh:
                         conf[j, i, k, c] = 0

                conf[j, i, k, 20:24] = [xmin, ymin, xmax, ymax]

    conf = np.reshape(conf, (7*7*2, 24))
    # for each class
    for cls in range(0, len(classes)):
        testname = "test_" + classes[cls] + ".txt"
        if not os.path.exists(testname):
            file = open(testname,"w+")
        else:
            file = open(testname,"a")

        #sort by class confidence
        conf = conf[conf[:,cls].argsort()[::-1]]
        # nms
        for i in range(0, len(conf)):
            if conf[i, cls] == 0 : continue
            boxa = conf[i, 20:24]
            for j in range(i+1,len(conf)):
                if conf[j, cls] == 0 : continue
                boxb = conf[j, 20:24]
                # intersection / union
                inter_w = (min(boxa[2], boxb[2]) - max(boxa[0], boxb[0]))
                inter_h = (min(boxa[3], boxb[3]) - max(boxa[1], boxb[1]))

                intersection = inter_w * inter_h
                if inter_w < 0 or inter_h < 0:
                    intersection = 0

                union = (boxa[2] - boxa[0]) * (boxa[3] - boxa[1]) + (boxb[2] - boxb[0]) * (boxb[3] - boxb[1]) - intersection
                iou = intersection / union

                if iou > thresh_iou :
                    conf[j, cls] = 0

        # result
        for i in range(0, 7*7*2):
            if conf[i, cls] > 0:
                xmin = max(0, conf[i, 20])
                ymin = max(0, conf[i, 21])
                xmax = min(img_width, conf[i, 22])
                ymax = min(img_height, conf[i, 23])
                file.write("{} {} {} {} {} {}\n".format(image_id, conf[i, cls], xmin, ymin, xmax, ymax))
print("DONE")
