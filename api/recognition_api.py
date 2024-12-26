import argparse
import json
import sys
import time
from os.path import join as ospjoin
from typing import List

from flask import request, send_file, flash, redirect

from detect.detecting_tool import show_result_img, base642cv


def parse_args():
    parser = argparse.ArgumentParser(description='child_recognition')
    # general
    parser.add_argument('--rec_checkpoint_path', default=ospjoin('recog', '16_backbone.pth'),
                        help='path to load model.')
    parser.add_argument('--network', default='r100', type=str, help='')
    parser.add_argument("--feature_path", default="childbackup", type=str, help="if you wanna save image, put path")
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--batch-size', default=128, type=int, help='')
    parser.add_argument("--port", default=3333, type=int, help="port number")
    parser.add_argument("--save_image", default=ospjoin("api", "test"), type=str,
                        help="if you wanna save image, put path")
    parser.add_argument("--test", default="test", type=str, help=" test or None | test or 0")
    parser.add_argument("--who_r_u_threshold", default=38.5, type=float, help="")
    return parser.parse_args()


def rec_predict(args, detect, recog):
    '''

    Args:
        args: args
        detect: detect model(retinaface)
        recog: recognition model(arcface)

    Returns: final_result ==  {"annotation' :  [ {"label": kid integer 값1, "bbox": [x,y,x,y]}, {"label": kid integer 값2, "bbox": [x,y,x,y]} ]}

    '''
    global img
    results, total_name, total_result, all_score, times = dict(), list(), list(), list(), list()

    if not request.method == "POST":
        sys.exit()  # Only for post
    # input : {"image": [base64 encoding 된 이미지1] , "school_id" : 기관id }
    if request.data:
        child = json.loads(request.data)
        class_name = child['school_id']
        image_file = child['image']

        class_name, args = request_check(class_name, args)
        args.feature_path = class_name
        image_file = image_file[0]
        img = base642cv(image_file)
        result, scores, img, times = one_img_det_rec(img, recog, detect, args, times)

        if args.test == 'show_test':
            img_file = show_result_img(img)
            print(results)
            return send_file(img_file, mimetype='image/png')

        result = who_r_u(all_score, [result], args.who_r_u_threshold)
        results['annotation'] = result[0]
        return json.dumps(results)
    else:
        flash('No selected file')
        return redirect(request.url)


def one_img_det_rec(img, recog, detect, args, times):
    '''

    Args:
        img: image
        recog: recog_model
        detect: detect_model
        args: args
        times: time information

    Returns:
    '''
    start_time = time.time()
    # Detect
    file, crop_img, bbox = detect.detect_with_retinaface(img)
    print("detect %d face | --- detect %s m seconds ---" % (len(bbox), round((time.time() - start_time) * 1000, 1)))
    if len(bbox) == 0:
        result, scores = [dict()], [0]
    # recognition
    else:
        class_id, result, scores, crop_img = recog.api_recognition(file, crop_img, bbox, img, args.test)
        print("ID : %s | --- total %s m seconds ---" % (class_id, round((time.time() - start_time) * 1000, 1)))
        times.append((time.time() - start_time) * 1000)

    return result, scores, img, times


def request_check(class_name, args):
    # if you wannna change mode, just write for purpose in class_name
    if 'test_' in class_name:  # if you need to check one image in window page
        class_name = class_name.replace('test', '')
        args.test = 'show_test'
    if 'old' in class_name:  # if you need to test checkpoint which don't train our child image
        class_name = '0104'
        args.rec_checkpoint_path = ospjoin('recog', '16_backbone.pth')
    return class_name, args


def who_r_u(score: List[List[int]], result: List[List[dict]], threshold=38.5) -> List[List[dict]]:
    # if score < threshold | label = 'who are you'
    for z, score_cut in enumerate(score):
        for y, scc in enumerate(score_cut):
            if scc < threshold:
                result[z][y]['label'] = 'who are you?'
    return result
