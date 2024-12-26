import argparse
import os
from os.path import join as ospjoin

from flask import Flask, render_template
from flask_cors import CORS

from api import recognition_api, detect_api, newclass_api
from detect.detecting_tool import UseOurDetect
from recog.recog_tool import Recognition

app = Flask(__name__)
CORS(app)
CORS(app, resources={r'*': {'origins': '*'}})

Recognition_URL = ospjoin("/", "child", "recognition")
DETECTION_URL = ospjoin("/", "child", "detect")
Newclass_URL = ospjoin("/", "child", "newclass")


@app.route('/rec')
def index1():
    return render_template('recognition.html')


@app.route('/ann')
def index2():
    return render_template('detect.html')


@app.route('/new')
def index3():
    return render_template('newclass.html')


@app.route(Recognition_URL, methods=["POST"])
def service1():
    return recognition_api.rec_predict(args, detect, recog)


@app.route(DETECTION_URL, methods=["POST"])
def service2():
    return detect_api.predict(args, detect)


@app.route(Newclass_URL, methods=["POST"])
def service3():
    return newclass_api.predict(args, detect, recog)


# TODO make same pathname
def parse_args():
    parser = argparse.ArgumentParser(description='child_recognition')
    # base
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--batch-size', default=128, type=int, help='')
    parser.add_argument("--port", default=3333, type=int, help="port number")
    parser.add_argument("--save_img_path", default="", type=str,
                        help="if you wanna save image, put path (folder name have '*test*')")
    parser.add_argument("--test", default="0", type=str, help="test or None | test or 0")
    # recog
    parser.add_argument('--rec_checkpoint_path', default='./recog/bestchild.pth', help='path to load model.')
    parser.add_argument('--network', default='r100', type=str, help='')
    parser.add_argument("--feature_path", default="best", type=str, help="if you wanna save image, put path")
    # detect
    # detection model checkpoint
    parser.add_argument('-m', '--trained_detect_model', default='./detect/models/Resnet50_Final.pth', type=str,
                        help='Trained state_dict file path to open')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--vis_threshold', default=0.6, type=float, help='visualization_threshold')
    parser.add_argument("--who_r_u_threshold", default=38.5, type=float, help="")  # 38.5
    parser.add_argument("--draw_bbox_keypoint", default='True', type=str, help="if you need result in face")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect = UseOurDetect(args)
    recog = Recognition(args, detect.device)
    try:
        os.makedirs(args.save_img_path, exist_ok=True)
        print('cropped image is saved in server')
    except FileNotFoundError:
        print('only post mode')
    app.run(host="0.0.0.0", port=args.port)
