import json
import os
import sys
import time
from os.path import join as ospjoin

import numpy as np
import pandas as pd
from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
CORS(app, resources={r'*': {'origins': '*'}})


def predict(args, detect, recog):
    if not request.method == "POST":
        sys.exit()
    if request.data:  # key check
        child = json.loads(request.data)
        # class_name = school_id
        # js_file =   "annotation": [{"label": kid integer 값1, "bbox": [ ]}, {"label": kid integer 값2, "bbox": [ ] } ]
        # image_file = "image": [base64 encoding 된 이미지1, base64 encoding 된 이미지2 .. ]
        class_name = child['school_id']
        os.makedirs(ospjoin('api', 'newclass_backup', class_name), exist_ok=True)
        image_files = child['image']
        labels = make_main_list(child['annotation'], 'label')
        start_time = time.time()
        # Detect Landmark + crop face
        if not len(labels) == len(image_files):
            print('label num =/= image num')
        bbox, results, imgs = detect.new_img2bbox_annotation(image_files)
        print("detect %d face | --- detect %s seconds ---" % (len(bbox), (int(time.time() - start_time))))
        if len(results) == 0:
            unique_templates = 'no face or json file have problem'
        # make feature / per class
        else:
            feature, unique_templates, template2id = recog.newclass_feature(results, imgs, labels)
            print("ID : %s | --- total %s seconds ---" % (unique_templates, (int(time.time() - start_time))))
            # save
            np.savez(ospjoin('api', 'newclass_backup', class_name, f'{class_name}_{len(labels)}_backup'),
                     feature=feature, unique_templates=unique_templates, template2id=template2id
                     )

        return json.dumps('Registration Number : ' + str(unique_templates))


def img_path_setting(image_files: list, classname: str) -> str:
    img_dir = ospjoin('api', 'newclass_backup', classname, 'temp')
    os.makedirs(img_dir, exist_ok=True)
    return img_dir


def test_each_feature_similarity(feature, unique_templates):
    class_id, scores = [], dict()
    for i in range(len(feature)):
        score = np.sum(feature * feature[i][0], -1)
        scores[i] = list(score)
        predict = unique_templates[score.argmax()]
        class_id.append(predict)
    print(pd.DataFrame.from_dict(scores))


def make_main_list(file, key):
    main_list = list()
    for v in file:
        main_list.append(v[key])
    return main_list
