import os
import pickle
import shutil
from pathlib import Path
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
    return digit, cy/np_sample.shape[0], cx/np_sample.shape[1]

def correct_filename(x:str):
    if os.path.sep != "\\":
        return x.replace("\\", "/")
    return x

IMAGE_DATA_TYPES = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
NUM_CLASSES = 10

def load_data(num_classes:int=NUM_CLASSES, return_X_y:bool=True, as_frame:bool=True, data_path:str="./data.pkl"):
    with open (data_path, "rb") as f:
        data = pickle.load(f)
        X = data["X"]
        y = data["y"]
    selector = y < num_classes
    _X = X[selector, :, :].reshape((-1, 48*48))
    _y = y[selector]
    if as_frame:
        _X = pd.DataFrame(data=_X, columns=["p_{}".format(str(i).zfill(5)) for i in range(0, 48*48)])
        _y = pd.Series(_y)
    if return_X_y:
        return _X, _y
    else:
        return {
            "X" : _X, 
            "y" : _y
        }

class UtilsCli(object):
    def chop_images(self, input_path:str="./raw_data", output_path:str="./chopped_data", pseudo_label_path:str="./chopped_data_info.csv"):
        """
        Chops up raw_data/* to individual images in ./data
        """
        files = []
        for idt in IMAGE_DATA_TYPES:
            files.extend(glob("{}/*.{}".format(input_path, idt)))
        pbar = tqdm(files)
        offset = 0
        fids, cxs, cys, sources, num_clusters = [], [], [], [], []
        opath = Path(output_path)
        opath.mkdir(parents=True, exist_ok=True)
        for f in pbar:
            pbar.set_description("working on file : {}; init ...".format(f))
            img = load_gray_image(f)
            img_norm = normalize(img)
            img_enh = enhance(img_norm, thr=96, copy=False)
            pbar.set_description("working on file : {}; mser ... ".format(f))
            points_X = get_mser_regions(img_enh, delta=10, copy=False)
            points_X = np.unique(points_X, axis=0)
            #selector = img_enh[points_X[:, 1], points_X[:, 0]] >= MAX_COLOR - 32
            #points_X = points_X[selector]
            dbscan = cluster.DBSCAN(eps=2.0, min_samples=10)
            ys = dbscan.fit_predict(points_X)
            for id in range(ys.min(), ys.max()):
                try:
                    _img, _cy, _cx = get_digit(img, points_X, ys, id, width=64+32, height=64+32)
                    _img = Image.fromarray(_img, mode="L")
                    fid = "img_{}.png".format(str(id + offset).zfill(6))
                    _img.save("{}/{}".format(output_path, fid))
                    fids.append(fid)
                    cxs.append(_cx)
                    cys.append(_cy)
                except Exception as e:
                    pass
            sources.extend([f] * (len(fids) - len(sources)))
            num_clusters.extend([ys.max()] * (len(fids) - len(num_clusters)))
            assert len(fids) == len(cxs) == len(cys) == len(sources) == len(cys)
            offset = offset + ys.max() #ys.shape[0]
        assert len(fids) == len(cxs) == len(cys) == len(sources) == len(cys)
        df = pd.DataFrame({
            "fid" : fids, "cx" : cxs, "cy" : cys, "source" : sources, "num_clusters" : num_clusters
        })
        df.to_csv(pseudo_label_path)

    def prepare_pseudo_label_data(self, input_path:str="./chopped_data/", info_path:str="./chopped_data_info.csv", output_path="./pseudo_label_data"):
        df_all = pd.read_csv(info_path, index_col=0)
        for _, df in tqdm(df_all.groupby(by="source")):
            df.sort_values(by="cx", inplace=True)
            df.reset_index(inplace=True)
            num_samples = df.shape[0]
            k = num_samples // NUM_CLASSES
            for l in tqdm(range(0, NUM_CLASSES)):
                fnames = df.fid[l*k: (l+1)*k] if l < NUM_CLASSES - 1 else df.fid[l*k: ]
                dp = Path(output_path).joinpath(str(l))
                dp.mkdir(parents=True, exist_ok=True)
                for f in fnames:
                    sp = Path(input_path).joinpath(f)
                    shutil.copy(str(sp), str(dp))

    def pickle_data(self, input_path:str="./labelled_data", output_path="./data.pkl", pickle_protocol:int=pickle.HIGHEST_PROTOCOL):
        tqdm.pandas()
        files = glob("{}/?/*.png".format(input_path))
        X = np.zeros((len(files), 48, 48))
        y = np.full(len(files), fill_value=-1)
        for idx, f in enumerate(tqdm(files)):
            f_parts = f.split("/")
            y[idx] = int(f_parts[-2])
            _X = load_gray_image(f, return_type="pillow")
            _X = _X.resize((48, 48), resample=Image.BICUBIC)
            _X = np.array(_X)
            _X_norm = normalize(_X)
            _X_enh = enhance(_X_norm, thr=96)
            _X_inv = 255 - _X_enh
            X[idx] = _X_inv

        data = {
            "X": X,
            "y": y
        }
        with open(output_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle_protocol)

if __name__ == "__main__":
    utils_cli = UtilsCli()
    fire.Fire(utils_cli)