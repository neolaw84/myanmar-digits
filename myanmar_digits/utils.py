import os
import pickle
from glob import glob

import cv2
from PIL import Image, ImageOps

import numpy as np
from sklearn import cluster
import pandas as pd

import fire

from tqdm.auto import tqdm

def load_gray_image(path:str, return_type:str="numpy"):
    assert return_type in ["numpy", "pillow", "opencv"]
    with open(path, "rb") as f:
        im_sample = Image.open(f)
        im_sample_gray = ImageOps.grayscale(im_sample)
        if return_type == "pillow":
            return im_sample_gray
        elif return_type == "numpy":
            np_sample_gray = np.array(im_sample_gray)
            return np_sample_gray
        elif return_type == "opencv":
            raise NotImplementedError("Not supported yet.")

def get_temp_image(img:np.array, copy:bool):
    if copy:
        _img = img.copy()
    else:
        _img = img
    return _img

def normalize(img:np.array, copy:bool=True):
    _img = get_temp_image(img, copy)
    _img = 1.0 - (_img - _img.min()) / (_img.max() - _img.min())
    _img = (255 * _img).astype(np.uint8)
    return _img

MAX_COLOR = 255

def enhance(img:np.array, thr:int=64, copy:bool=True):
    _img = get_temp_image(img, copy)
    _img[_img < thr] = 0
    _img[_img >= MAX_COLOR - thr] = MAX_COLOR
    return _img

def get_mser_regions(img:np.array, delta:int=10, copy:bool=True):
    _img = get_temp_image(img, copy)
    mser = cv2.MSER_create(delta)
    points, _ = mser.detectRegions(_img)
    points_X = np.concatenate([np.array(p) for p in points])
    return points_X

def get_center(points_X, ys, cluster_id):
    selector = ys == cluster_id
    min_y, max_y =  points_X[selector, 1].min(), points_X[selector, 1].max()
    min_x, max_x, = points_X[selector, 0].min(), points_X[selector, 0].max()
    w, h = max_x - min_x, max_y - min_y
    cx, cy = min_x + w/2, min_y + h/2
    return cy, cx

def get_digit(np_sample, points_X, ys, cluster_id, width=256, height=256):
    cy, cx = get_center(points_X, ys, cluster_id)
    o_st_x, o_st_y = int(cx - width/2), int(cy - height/2)
    st_x, st_y = max(0, o_st_x), max(0, o_st_y)
    o_en_x, o_en_y = o_st_x + width, o_st_y + height
    en_x, en_y = min(np_sample.shape[1], o_en_x), min(np_sample.shape[0], o_en_y)
    left_fill, right_fill = 0 - o_st_x, o_en_x - en_x
    top_fill, bottom_fill = 0 - o_st_y, o_en_y - en_y
    
    digit = np_sample[st_y:en_y, st_x: en_x]
    
    if left_fill > 0:
        digit = np.hstack((np.full((digit.shape[0], left_fill), 255), digit))
    elif right_fill > 0:
        digit = np.hstack((digit, np.full((digit.shape[0], right_fill), 255)))

    if top_fill > 0:
        digit = np.vstack((np.full((top_fill, digit.shape[1]), 255), digit))
    elif bottom_fill > 0:
        digit = np.vstack((digit, np.full((bottom_fill, digit.shape[1]), 255)))

    assert digit.shape == (height, width)
    return digit, cy/np_sample.shape[1], cx/np_sample.shape[0]

def correct_filename(x:str):
    if os.path.sep != "\\":
        return x.replace("\\", "/")
    return x

IMAGE_DATA_TYPES = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]

class UtilsCli(object):
    def chop_images(self, input_path:str="./raw_data", output_path:str="./data", pseudo_label_path:str="./pseudo_label.csv"):
        """
        Chops up raw_data/* to individual images in ./data
        """
        files = []
        for idt in IMAGE_DATA_TYPES:
            files.extend(glob("{}/*.{}".format(input_path, idt)))
        pbar = tqdm(files)
        offset = 0
        fids, cxs = [], []
        for f in pbar:
            pbar.set_description("working on file : {}; init ...".format(f))
            img = load_gray_image(f)
            img_norm = normalize(img)
            img_enh = enhance(img_norm, copy=False)
            pbar.set_description("working on file : {}; mser ... ".format(f))
            points_X = get_mser_regions(img_enh, delta=5, copy=False)
            points_X = np.unique(points_X, axis=0)
            selector = img_enh[points_X[:, 1], points_X[:, 0]] >= MAX_COLOR - 32
            points_X = points_X[selector]
            pbar.set_description("working on file : {}; points_X.shape : {};".format(f, points_X.shape))
            dbscan = cluster.DBSCAN(eps=2.0, min_samples=5)
            ys = dbscan.fit_predict(points_X)
            pbar.set_description("working on file : {}; points_X.shape : {}; num_clusters : {}".format(f, points_X.shape, ys.shape))
            for id in range(ys.min(), ys.max()):
                try:
                    _img, _cy, _cx = get_digit(img, points_X, ys, id, width=64+32, height=64+32)
                    _img = Image.fromarray(_img, mode="L")
                    fid = "img_{}.png".format(str(id + offset).zfill(6))
                    _img.save("{}/{}".format(output_path, fid))
                    fids.append(fid)
                    cxs.append(_cx)
                except Exception as e:
                    pass
            offset = offset + ys.max() #ys.shape[0]
        df = pd.DataFrame({"fid" : fids, "cx" : cxs})
        df.to_csv(pseudo_label_path)

    def prepare_label_csv(self, input_path:str="./data", output_path="./data/label.csv"):
        files = glob("{}/*.png".format(input_path))
        df = pd.DataFrame({"filename" : files, "labels": [-1] * len(files)})
        df.to_csv(output_path)

    def pickle_data(self, input_path:str="./label.csv", output_path="./data.pkl", pickle_protocol:int=pickle.HIGHEST_PROTOCOL):
        tqdm.pandas()
        df_label = pd.read_csv(input_path, index_col=0)
        df_label = df_label[df_label.labels >= 0]
        df_label["filename"] = df_label.filename.apply(lambda x : correct_filename(x))
        df_label["X"] = df_label.progress_apply(lambda x : load_gray_image(x.filename), axis=1)
        df_label.drop(columns="filename", inplace=True)
        df_label = df_label[["X", "labels"]]
        X = df_label.X.values
        y = df_label.labels.values
        data = {
            "X": X,
            "y": y
        }
        with open(output_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle_protocol)

if __name__ == "__main__":
    utils_cli = UtilsCli()
    fire.Fire(utils_cli)