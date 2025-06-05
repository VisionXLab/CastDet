#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   merge_ovdg_preds.py
@Version      :   1.0
@Time         :   2024/09/12 16:10:56
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   None
'''

import os
import numpy as np
import torch
from mmrotate.structures.bbox import RotatedBoxes
import json
import argparse
from tqdm import tqdm
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge OVD pseudo labels.")
    parser.add_argument("--ann_path", type=str, required=True, help="Path to the original json path")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the prediction path (.box.json)")
    parser.add_argument("--save_path", type=str, required=True, help="Path where the pseudo labels will be saved")
    parser.add_argument("--score_thr", type=float, default=0.02, help="Scores lower than score_thr will be filtered.")
    parser.add_argument("--topk", type=int, default=None, help="Choose top-k bboxes.")
    
    args = parser.parse_args()

    id2pred = {}
    with open(args.pred_path, 'r') as f:
        data_list = json.loads(f.read())

    for item in data_list:
        id = item["image_id"]
        if id not in id2pred:
            id2pred[id] = {
                'bboxes': [item['bbox']],
                'scores': [item['score']]
            }
        else:
            id2pred[id]['bboxes'].append(item['bbox'])
            id2pred[id]['scores'].append(item['score'])

    with open(args.ann_path, 'r') as fp:
        data = json.load(fp)

    ann_id = 0
    for meta in tqdm(data["images"]):
        cid = meta["category_id"]
        image_id = meta["id"]
        if image_id in id2pred:
            scores = torch.tensor(id2pred[image_id]["scores"])
            if args.topk is None:
                filter = scores > args.score_thr
            else:
                filter = scores.topk(args.topk).indices
            bboxes = RotatedBoxes(torch.tensor(id2pred[image_id]["bboxes"]))[filter]
            areas = bboxes.areas.tolist()
            qboxes = bboxes.convert_to('qbox').tensor.tolist()
            hboxes = bboxes.convert_to('hbox').tensor.tolist()
            for area, qbox, hbox in zip(areas, qboxes, hboxes):
                qbox = list(map(int, qbox))
                hbox = list(map(int, hbox))
                data["annotations"].append({
                    "id": ann_id,
                    "area": area,
                    "category_id": cid,
                    "segmentation": [qbox],
                    "iscrowd": 0,
                    "bbox": hbox,
                    "image_id": image_id
                })
                ann_id += 1
            
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    print(f"Total of {len(data['annotations'])} instances.")

    with open(args.save_path, 'w') as fp:
        json.dump(data, fp)
