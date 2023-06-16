from pathlib import Path
from utils import *
from glob import glob
import numpy as np
import os
import plyfile
import time
import platform
from multiprocessing import Pool, TimeoutError
import sys
from joblib import Parallel, delayed
import multiprocessing

def cpyTxtFiles(src_dir, target_dir):
    #copy json, eye_data, hand_data, combined_eye_hand data
    assert os.path.exists(src_dir)
    w_src_orig_path = Path(src_dir)
    json_annotation_file = searchForAnnotationJson(str(w_src_orig_path))
    if not json_annotation_file:
        print(f"could not find json in {str(w_src_orig_path)}")
        assert False
    annotation_file_name = json_annotation_file.split("\\")[-1]
    assert os.path.exists(json_annotation_file)
    src_file_list = [(json_annotation_file, annotation_file_name),
                     (os.path.join(src_dir, "head_hand_eye.csv"),"head_hand_eye.csv"),
                     (os.path.join(src_dir, "norm_proc_eye_data.csv"), "norm_proc_eye_data.csv"),
                     (os.path.join(src_dir, "norm_proc_hand_data.csv"), "norm_proc_hand_data.csv")
                     ]
    if not os.path.exists(target_dir):
        print("making target dir: ", target_dir)
        os.makedirs(target_dir)

    for src_file, filename in src_file_list:
        target_file = os.path.join(target_dir,filename)
        if not os.path.exists(target_file):
            copyfile(src_file, target_file)


def cpyPvFiles(src_dir, target_dir):

    src_pv_file_list = [pv_file for pv_file in os.listdir(src_dir)
                                if ".png" in pv_file]
    assert os.path.exists(src_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if len(os.listdir(src_dir))==len(os.listdir(target_dir)):
        return
        
    for src_file_name in src_pv_file_list:
        src_file_path = os.path.join(src_dir, src_file_name)
        target_file_path = os.path.join(target_dir, src_file_name)
        assert os.path.exists(src_file_path)
        assert os.path.exists(target_dir)
        copyfile(src_file_path, target_file_path)


def handleSinglePly(arg, num_fps_points=4096):
    start = time.process_time()
    use_fps = False
    src_file, target_dir = arg
    assert os.path.exists(src_file)

    if "Windows" in platform.platform():
        filename = src_file.split("\\")[-1]
    else:
        filename = src_file.split("/")[-1]
    target_file = os.path.join(target_dir, filename)
    if os.path.exists(target_file):
        return 0

    plydata = plyfile.PlyData.read(src_file)

    d = np.asarray(plydata['vertex'].data)
    pc = np.column_stack([d[p.name] for p in plydata['vertex'].properties])
    if use_fps:
        sampled_points = fps_ne(npoint=num_fps_points, points=pc, stochastic_sample=False)
    else:
        sampled_points = stochastic_vec_sample_numeric(vec=pc, num_samples=num_fps_points)

    pts = list(zip(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], sampled_points[:, 3],
                   sampled_points[:, 4], sampled_points[:, 5], sampled_points[:, 6], sampled_points[:, 7], sampled_points[:, 8]))
    vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                  ('red', 'B'), ('green', 'B'), ('blue', 'B')])
    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el], text="").write(target_file)
    return(target_file, time.process_time() - start)

def createFpsRecCpy(src_long_throw_dir, target_dir, num_fps_points = 4096):

    ply_file_list = [os.path.join(src_long_throw_dir, long_throw_data) for long_throw_data in os.listdir(src_long_throw_dir)
                     if ".ply" in long_throw_data]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    start = time.process_time()

    #####
    # multiprocessing:
    #####
    num_cores = multiprocessing.cpu_count()
    # num_cores = 4
    arg_zip = list(zip(ply_file_list, [target_dir for _ in range(len(ply_file_list))]))
    Parallel(n_jobs=num_cores)(
        delayed(handleSinglePly)(arg_zip[i]) for i in range(len(ply_file_list)))

    return


def createSmallDataset(src_dataset, target_dataset, furniture_modalities):

    for furniture_name in furniture_modalities:
        furniture_src_dir = os.path.join(src_dataset, furniture_name)
        furniture_target_dir = os.path.join(target_dataset, furniture_name)
        reg_furniture_rec_list = [os.path.join(furniture_src_dir, _dir)for _dir in os.listdir(furniture_src_dir)
                             if "_recDir" in _dir]
        target_furniture_rec_list = [os.path.join(furniture_target_dir, _dir)for _dir in os.listdir(furniture_src_dir)
                             if "_recDir" in _dir]


        for target_furniture_dir, src_furniture_rec_dir in zip(target_furniture_rec_list, reg_furniture_rec_list):
            if not os.path.exists(src_furniture_rec_dir):
                raise NotADirectoryError
            src_long_throw_dir = os.path.join(src_furniture_rec_dir, "Depth Long Throw")
            target_long_throw_dir = os.path.join(target_furniture_dir, "Depth Long Throw")
            print("starting cpy ply for dir: ", src_long_throw_dir)
            createFpsRecCpy(src_long_throw_dir=src_long_throw_dir, target_dir=target_long_throw_dir)
            src_pv = os.path.join(src_furniture_rec_dir, "pv")
            target_pv = os.path.join(target_furniture_dir, "pv")
            print("starting cpy pv for dir: ", src_pv)
            cpyPvFiles(src_pv, target_pv)
            print("starting cpy txt files from dir: ", src_furniture_rec_dir, "to: ", target_furniture_dir)
            cpyTxtFiles(src_furniture_rec_dir, target_furniture_dir)


if __name__ == '__main__':
    furniture_modalities = ["Stool", "Drawer", "Table", "Coffee_Table"]
    if "Windows" in platform.platform():
        src_dataset = r'C:\SmallDataset'
        target_dataset = r'C:\TinyDataset'
    else:
        #linux:
        target_dataset = r'/mnt/c/SmallDataset'
        src_dataset= r'/mnt/d/HoloLens'
    createSmallDataset(src_dataset, target_dataset, furniture_modalities)
