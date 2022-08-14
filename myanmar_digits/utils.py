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
    st_x, st_y = max(0, int(cx - width/2)), max(0, int(cy - height/2))
    en_x, en_y = min(np_sample.shape[1], st_x + width), min(np_sample.shape[0], st_y + height)
    return np_sample[st_y:en_y, st_x: en_x]

IMAGE_DATA_TYPES = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]

class UtilsCli(object):
    def chop_images(self, input_path:str="./raw_data", output_path:str="./data"):
        """
        Chops up raw_data/* to individual images in ./data
        """
        files = []
        for idt in IMAGE_DATA_TYPES:
            files.extend(glob("{}/*.{}".format(input_path, idt)))
        pbar = tqdm(files)
        offset = 0
        for f in pbar:
            pbar.set_description("working on file : {}; init ...".format(f))
            img = load_gray_image(f)
            img_norm = normalize(img)
            img_enh = enhance(img_norm, copy=False)
            pbar.set_description("working on file : {}; mser ... ".format(f))
            points_X = get_mser_regions(img_enh, delta=5, copy=False)
            points_X = np.unique(points_X, axis=0)
            selector = img_enh[points_X[:, 1], points_X[:, 0]] >= MAX_COLOR - 64
            points_X = points_X[selector]
            pbar.set_description("working on file : {}; points_X.shape : {};".format(f, points_X.shape))
            dbscan = cluster.DBSCAN(eps=2.0, min_samples=10)
            ys = dbscan.fit_predict(points_X)
            #kmeans = cluster.KMeans(n_clusters=100, max_iter=500, n_init=20, random_state=42)
            #ys = kmeans.fit_predict(points_X)
            pbar.set_description("working on file : {}; points_X.shape : {}; num_clusters : {}".format(f, points_X.shape, ys.shape))
            for id in range(ys.min(), ys.max()):
                try:
                    _img = get_digit(img, points_X, ys, id, width=64+32, height=64+32)
                    _img = Image.fromarray(_img, mode="L")
                    _img.save("{}/img_{}.png".format(output_path, str(id + offset).zfill(6)))
                except:
                    pass
            offset = offset + ys.shape[0]

    def prepare_label_csv(self, input_path:str="./data", output_path="./data/label.csv"):
        files = glob("{}/*.png".format(input_path))
        df = pd.DataFrame({"filename" : files, "labels": [-1] * len(files)})
        df.to_csv(output_path)

if __name__ == "__main__":
    utils_cli = UtilsCli()
    fire.Fire(utils_cli)