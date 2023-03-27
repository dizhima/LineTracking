# RGBD Line Tracking
The code is for a line tracking project, the goal is to use the RGB channel of a Realsense Camera to detect and track the lines. The depth channel will be used for transfer the 2D lines to 3D. The 3D line results of last frame will be convert into a custom Unity3D format and saved in a json file.

## 1. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 2. Usage

- Run the inference code on realsense camera

```bash
python mlsd_depth_tracking.py
```

## 3. Reference
* [M-LSD: Towards Light-weight and Real-time Line Segment Detection](https://github.com/navervision/mlsd)
* [M-LSD-onnxrun-cpp-py](https://github.com/hpc203/M-LSD-onnxrun-cpp-py)
* [librealsense](https://github.com/IntelRealSense/librealsense)

## License

[MIT](https://choosealicense.com/licenses/mit/)