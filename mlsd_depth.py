###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import copy
import argparse
import cv2
import numpy as np
import onnxruntime

class M_LSD:
    def __init__(self, modelpath, conf_thres=0.5, dist_thres=20.0):
        # Initialize model
        self.onnx_session = onnxruntime.InferenceSession(modelpath, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_names = [self.onnx_session.get_outputs()[i].name for i in range(3)]

        self.input_shape = self.onnx_session.get_inputs()[0].shape ### n,h,w,c
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        self.conf_threshold = conf_thres
        self.dist_threshold = dist_thres

    def prepare_input(self, image):
        resized_image = cv2.resize(image, dsize=(self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        input_image = np.concatenate([resized_image, np.ones([self.input_height, self.input_width, 1])], axis=-1)
        input_image = np.expand_dims(input_image, axis=0).astype('float32')
        return input_image

    def detect(self, image, depth_map=None):
        h_ratio, w_ratio = [image.shape[0] / self.input_height, image.shape[1] / self.input_width]
        input_image = self.prepare_input(image)

        # Perform inference on the image
        result = self.onnx_session.run(self.output_names, {self.input_name: input_image})

        pts = result[0][0]
        pts_score = result[1][0]
        vmap = result[2][0]

        start = vmap[:, :, :2]
        end = vmap[:, :, 2:]
        dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

        segments_list = []
        for center, score in zip(pts, pts_score):
            y, x = center
            distance = dist_map[y, x]
            if score > self.conf_threshold and distance > self.dist_threshold:
                disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
                x_start = x + disp_x_start
                y_start = y + disp_y_start
                x_end = x + disp_x_end
                y_end = y + disp_y_end
                segments_list.append([x_start, y_start, x_end, y_end])

        try:
            lines = 2 * np.array(segments_list)  # 256 > 512
            lines[:, 0] = lines[:, 0] * w_ratio
            lines[:, 1] = lines[:, 1] * h_ratio
            lines[:, 2] = lines[:, 2] * w_ratio
            lines[:, 3] = lines[:, 3] * h_ratio

            # Draw Line
            dst_image = copy.deepcopy(image)

            for line in lines:
                x_start, y_start, x_end, y_end = [int(val) for val in line]
                cv2.line(dst_image, (x_start, y_start), (x_end, y_end), [0, 0, 255], 3)
                if depth_map is not None:
                    center = ((x_start + x_end) // 2, (y_start + y_end) // 2)
                    # print(x_start, y_start, x_end, y_end, center)
                    center_depth = depth_map[center[1], center[0]]
                    org = center  # (x1, y1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    dst_image = cv2.putText(dst_image, f'{center_depth}', org, font,
                                      fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        except:
            dst_image = copy.deepcopy(image)

        return dst_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default='.\M-LSD-onnxrun-cpp-py\weights\model_512x512_large.onnx', help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--distThreshold', default=20.0, type=float, help='dist threshold')
    args = parser.parse_args()

    # using M-LSD model
    print('using M-LSD')
    detector = M_LSD(args.modelpath, conf_thres=args.confThreshold, dist_thres=args.distThreshold)

    # Configure depth and color streams
    print('Configuring realsense')
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    print('starting streaming')
    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    count = 0
    try:
        while True:

            count += 1

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Convert images to numpy arrays
            depth_colormap = depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                # print(depth_colormap_dim, color_colormap_dim)
                color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                         interpolation=cv2.INTER_AREA)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # line detection
            color_image = detector.detect(color_image, depth_image)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        pipeline.stop()