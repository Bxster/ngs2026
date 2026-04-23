import hailo_platform as hp
import numpy as np
import cv2

hef = hp.HEF('/home/pi/yolov11s.hef')
target = hp.VDevice()
network_group = target.configure(hef)[0]
input_vstreams_params = hp.InputVStreamParams.make(network_group, format_type=hp.FormatType.UINT8)
output_vstreams_params = hp.OutputVStreamParams.make(network_group, format_type=hp.FormatType.FLOAT32)

img = cv2.imread('/home/pi/test_images/1.jpeg')
resized = cv2.resize(img, (640, 640))
frame = np.expand_dims(resized.astype(np.uint8), axis=0)

with network_group.activate():
    with hp.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as pipeline:
        output = pipeline.infer({'yolov11s/input_layer1': frame})
        data = output['yolov11s/yolov8_nms_postprocess']
        print('type:', type(data))
        print('len:', len(data))
        print('first element type:', type(data[0]))
        if hasattr(data[0], 'shape'):
            print('first element shape:', data[0].shape)
        else:
            print('first element:', data[0])