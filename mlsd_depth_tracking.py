import pyrealsense2 as rs
import copy
import argparse
import cv2
import numpy as np
import onnxruntime
import math
import json
from scipy.spatial.transform import Rotation as R

# result is wrong...
def xyz2qua(vec):
    '''
    return quaternion angle to xyz=(1, 0, 0)
    :param vec:
    :return: qua
    '''
    x, y, z = vec
    zdegree = math.degrees(math.atan(z / x))
    ydegree = math.degrees(math.atan(y / x))
    # rot = R.from_euler('zyx', [0, -ydegree, zdegree], degrees=True) # to unity coord
    theta = [0, -ydegree, zdegree] # to unity coord
    rot = R.from_euler('xyz', theta, degrees=True)
    qua = rot.as_quat()
    return qua

def save_json(lines):
    info = {"LineFeatures": []}
    for idx,(p1, p2) in enumerate(lines):
        # opencv to Unity3D coordinate
        y, x, z = p1
        yy, xx, zz = p2
        y, yy = -y, -yy

        scale = math.sqrt((x-xx)**2+(y-yy)**2+(z-zz)**2)
        center = (x+xx)/(2*scale), (y+yy)/(2*scale), (z+zz)/(2*scale)
        vec = xx-x, yy-y, zz-z
        qua = xyz2qua(vec)

        info["LineFeatures"].append(
                {
                    "ID": idx,
                    "Type": "LineFeature",
                    "Position": {
                        "x": center[0],
                        "y": center[1],
                        "z": center[2]
                    },
                    "Rotation": {
                        "x": qua[0],
                        "y": qua[1],
                        "z": qua[2],
                        "w": qua[3]
                    },
                    "Scale": {
                        "x": scale,
                        "y": scale,
                        "z": scale
                    },
                    "Positions": [
                        {
                            "x": x/scale,
                            "y": y/scale,
                            "z": z/scale
                        },
                        {
                            "x": xx/scale,
                            "y": yy/scale,
                            "z": zz/scale
                        }
                    ]
                }
                )
    json_object = json.dumps(info, indent=4)

    # Writing to sample.json
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)

    return

def get_depth(point, depth_map):
    def generate_points(point):
        x, y = point
        shift_pix = 10
        return [(x, y),
                (x+shift_pix, y), (x, y+shift_pix),
                (x-shift_pix, y), (x, y-shift_pix),
                (x+shift_pix, y+shift_pix), (x+shift_pix, y-shift_pix),
                (x-shift_pix, y+shift_pix), (x-shift_pix, y-shift_pix)]

    points = generate_points(point)
    depths = []
    for i, j in points:
        depth = depth_map[min(max(i,0), depth_map.shape[0]-1), min(max(0,j), depth_map.shape[1]-1)]
        if depth>1e-10:
            depths.append(depth)

    if depths:
        return min(depths)
    else:
        return 0

def camera2world(lines, depth_map, depth_intrin):
    world_lines = []
    # print(f'depth_scale {depth_scale}')
    for line, _ in lines:
        (x, y, xx, yy) = line.astype(int)

        # print(x,y,xx,yy, line, depth_map.shape)
        if (x<depth_map.shape[1] and
                xx<depth_map.shape[1] and
                y<depth_map.shape[0] and
                yy<depth_map.shape[0] and
        x>0 and xx>0 and y>0 and yy>0):
            pass
        else:
            continue

        # z = depth_map[y, x]*depth_scale
        z = get_depth((y, x), depth_map)*depth_scale
        dx, dy, dz = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, [y, x], z)

        # zz = depth_map[yy, xx] * depth_scale
        zz = get_depth((yy, xx), depth_map) * depth_scale
        dxx, dyy, dzz = rs.rs2_deproject_pixel_to_point(
            depth_intrin, [yy, xx], zz)

        world_lines.append([(dx, dy, dz), (dxx, dyy, dzz)])

    return world_lines

def shift_point(box):
    """
    shitf for better iou calculation
    :param box: xyxy 左上右下
    :return xyxy:
    """
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    thred = 20
    if xmax - xmin < thred:
        xmax += 100 - (xmax - xmin)
    if ymax - ymin < thred:
        ymax += 100 - (ymax - ymin)
    return xmin, ymin, xmax, ymax


def cal_iou(box1, box2):
    """
    :param box1: xyxy 左上右下
    :param box2: xyxy
    :return:
    """

    x1min, y1min, x1max, y1max = shift_point(box1)
    x2min, y2min, x2max, y2max = shift_point(box2)

    # 计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交部分的坐标
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    # 计算iou
    iou = intersection / union
    return iou

class M_LSD:
    def __init__(self, modelpath, conf_thres=0.5, dist_thres=20.0):
        # Initialize model ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.onnx_session = onnxruntime.InferenceSession(modelpath, providers=['CUDAExecutionProvider'])
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

    def detect(self, image, pre_lines=None, start_id=0, depth_map=None, depth_intrin=None):
        """
        :param image: xyxy
        :param lines: [(xyxy, idx)]
        :return:
        """
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

        lines = 2 * np.array(segments_list)  # 256 > 512
        lines_with_id = []

        try:
            lines[:, 0] = lines[:, 0] * w_ratio
            lines[:, 1] = lines[:, 1] * h_ratio
            lines[:, 2] = lines[:, 2] * w_ratio
            lines[:, 3] = lines[:, 3] * h_ratio

            # Draw Line
            dst_image = copy.deepcopy(image)
            for line in lines:
                x_start, y_start, x_end, y_end = [int(val) for val in line]

                max_iou = 0
                thred = 0.8
                nearest = False
                for point, idx in pre_lines:
                    iou = cal_iou(line, point)

                    if iou > max_iou:
                        nearest = point, idx
                        max_iou = iou

                if max_iou > thred:
                    cur_id = nearest[1]
                    # info = f'old {cur_id} iou{max_iou:.2f}'
                    info = f'id {cur_id}'
                    lines_with_id.append((line, cur_id))
                else:
                    cur_id = start_id
                    # info = f'new {cur_id} iou{max_iou:.2f}'
                    info = f'id {cur_id}'
                    lines_with_id.append((line, cur_id))
                    start_id += 1


                cv2.line(dst_image, (x_start, y_start), (x_end, y_end), [0, 0, 255], 3)

                center = ((x_start + x_end) // 2, (y_start + y_end) // 2)
                global depth_scale
                center_depth = depth_map[center[1], center[0]]*depth_scale if depth_map is not None else False

                world_coord = camera2world([(line, cur_id)], depth_map, depth_intrin)
                try:
                    (dx, dy, dz), (dxx, dyy, dzz) = world_coord[0]
                    distance = math.sqrt(((dx-dxx) ** 2) + ((dy-dyy) ** 2) + ((dz-dzz) ** 2))
                    info += f' l:{distance:.2f}'

                    org = center  # (x1, y1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    dst_image = cv2.putText(dst_image, f'{info}', org, font,
                                            fontScale=0.5, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                    lines_with_id.append((line, start_id))
                    start_id += 1
                # info = info + f' no depth' if not center_depth else info + f' depth{center_depth:.2f}'
                except:
                    pass


        except:
            dst_image = copy.deepcopy(image)
            pass
        return dst_image, lines_with_id, start_id


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
    # pipeline.start(config)

    profile = pipeline.start(config)
    global depth_scale
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # set depth parameters
    # 0 for depth sensor, 1 for camera sensor
    sensor = profile.get_device().query_sensors()[0]
    sensor.set_option(rs.option.min_distance, 0)
    # sensor.set_option(rs.option.confidence_threshold, 1)
    # # sensor.set_option(rs.option.max_distance, 190)
    # sensor.set_option(rs.option.laser_power, 95)
    # sensor.set_option(rs.option.noise_filtering, 1)
    # sensor.set_option(rs.option.receiver_gain, 18)
    # sensor.set_option(rs.option.post_processing_sharpening, 1)
    # sensor.set_option(rs.option.pre_processing_sharpening, 0)
    # sensor.set_option(rs.option.global_time_enabled, 1.0)

    # for color sensor
    sensor = profile.get_device().query_sensors()[1]
    sensor.set_option(rs.option.global_time_enabled, 1.0)

    align_to = rs.stream.depth
    align = rs.align(align_to)

    count = 0
    lines = []
    start_id = 0
    try:
        while count< 50 :

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

            depth_intrin = aligned_frames.get_color_frame().profile.as_video_stream_profile().intrinsics
            # color_image = detector.detect(color_image, depth_image)
            color_image, lines, start_id = detector.detect(color_image, lines, start_id, depth_image, depth_intrin)

            if lines:
                world_lines = camera2world(lines, depth_image, depth_intrin)
                save_json(world_lines)
                cv2.imwrite('last_frame.png', color_image)
                # print(world_lines)

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