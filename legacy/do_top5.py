#! /usr/bin/env python3

from mvnc import mvncapi as mvnc
import numpy
import cv2
import os
import sys
import numpy as np
import caffe
import lmdb
from caffe.proto import caffe_pb2
lmdb_file = sys.argv[1] #path to your folder containing lmdb for example: "/home/yuan/Downloads/lejun_test_lmdb/"
dim=(224,224)
caf=False #False for mvnc
if caf:
    caffe_proto = sys.argv[2]
    caffe_weights = sys.argv[3]
    caffe.set_mode_cpu() #switch to gpu if you want
    net = caffe.Net(caffe_proto, caffe_weights, caffe.TEST)
if not caf:
    mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        quit()
    device = mvnc.Device(devices[0])
    device.OpenDevice()
    filefolder = os.path.dirname(os.path.realpath(__file__))
    network_blob = filefolder + '/graph'
    with open(network_blob, mode='rb') as f:
        blob = f.read()
    graph = device.AllocateGraph(blob)

if __name__ == "__main__":
    count=0
    top1=0
    top5=0
    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        count=count+1
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        im = np.transpose(data, (1, 2, 0))
        im = np.float32(im)
        mean = [91.9314, 90.1989, 91.0537]#load the mean of your training set
        im[:, :, 0] = (im[:, :, 0] - mean[0])
        im[:, :, 1] = (im[:, :, 1] - mean[1])
        im[:, :, 2] = (im[:, :, 2] - mean[2])
        print("label ", label)
        if not caf:
            graph.LoadTensor(im.astype(np.float16), 'user object')
            output, userobj = graph.GetResult()
            order = output.argsort()[::-1][:5]
            print("mvnc:",order)
            if order[0]==int(label): top1=top1+1
            for i in range (0,5):
                if order[i]==int(label):
                    top5=top5+1
        if caf:
            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
            transformer.set_transpose('data', (2, 0, 1))
            inputs = im
            result = net.forward_all(data=np.asarray([transformer.preprocess('data', inputs)]))
            result = result['prob'][0]
            ordercaffe = result.argsort()[::-1][:5]
            print("caffe:", ordercaffe)
            if ordercaffe[0]==int(label): top1=top1+1
            for i in range (0,5):
                if ordercaffe[i]==int(label):
                    top5=top5+1
    print ("This is count",count)
    print("This is top1 accuracy: ",top1/count)
    print("This is top5 accuracy: ",top5/count)
    print("Done")
    if not caf:
        graph.DeallocateGraph()
        device.CloseDevice()