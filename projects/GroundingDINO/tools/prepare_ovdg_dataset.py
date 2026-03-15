#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   prepare_ovdg_dataset2.py
@Version      :   1.0
@Time         :   2024/09/13 14:01:26
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   None
'''

import os
import json
import imagesize
import argparse

from data_classes import DOTA2NWPU as Data2NWPU


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare NWPU45 dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the nwpu45 dataset directory")
    parser.add_argument("--save_path", type=str, required=True, help="Path where the json file will be saved")
    args = parser.parse_args()

    data = {"annotations": [],
            "images": [],
            "categories": []}
    cname2cid = {cname: cid for cid, cname in enumerate(Data2NWPU.keys())}
    data["categories"] = [{'id': cid, 'name': cname} for cname, cid in cname2cid.items()]

    img_names = []
    dirname2cname = {}
    for cname, dir in Data2NWPU.items():
        if dir is not None:
            dirname2cname[dir] = cname
            files = os.listdir(os.path.join(args.data_dir, dir))
            files = [os.path.join(dir, f) for f in files]
            img_names.extend(files)

    for id, fname in enumerate(img_names):
        width, height = imagesize.get(os.path.join(args.data_dir, fname))
        texts = f"a photo of a {os.path.dirname(fname)}."
        data["images"].append({"id": id,
                            "file_name": fname,
                            "height": height,
                            "width": width,
                            "caption": texts,
                            "category_id": cname2cid[dirname2cname[os.path.dirname(fname)]]})

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    with open(args.save_path, 'w') as f:
        json.dump(data, f, indent=4)
