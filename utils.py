from pathlib import Path
from glob import glob
import numpy as np
import os
import csv
from shutil import copyfile
import cv2
import numpy as np
import random
import json
from PIL import Image, ImageDraw, ImageFont
import struct
import open3d as o3d
# import numexpr as ne

import torchvision.transforms as transforms

from hand_defs import HandJointIndex




def save_ply(output_path, points, rgb=None, cam2world_transform=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.estimate_normals()
    if cam2world_transform is not None:
        # Camera center
        camera_center = (cam2world_transform) @ np.array([0, 0, 0, 1])
        o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, camera_center[:3])

    o3d.io.write_point_cloud(output_path, pcd)



def imread_pgm(pgmdir):
    depth_image = cv2.imread(pgmdir, -1)
    return depth_image

def fps_np(points, npoint):
    # returns farthers point distance sampling
    # shuffle each sequence individually but keep correspondance throughout the sequence
    """
    Input:
        points: pointcloud data, [N, 3]
    Return:
        points: farthest sampled pointcloud, [npoint]
    """
    # xyz = points[0] #use the first frame for sampling
    xyz = points

    N, C = xyz.shape
    centroids = np.zeros(npoint)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    idxs = np.array(farthest)[None]

    for i in range(npoint-1):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.array(np.argmax(distance, -1))[None]
        idxs = np.concatenate([idxs, farthest])


    return points[idxs]


def stochastic_vec_sample_numeric(vec, num_samples = 4096):
    np.random.shuffle(vec)
    return vec[:num_samples]

def stochastic_vec_sample_proportional(vec, stochastic_sample_ratio_inv = 1):
    target_length = int(len(vec)/stochastic_sample_ratio_inv)
    return stochastic_vec_sample_numeric(vec, target_length)


def fps_ne(points, npoint, stochastic_sample:bool, stochastic_sample_ratio_inv = 1):
    # returns farthers point distance sampling
    # shuffle each sequence individually but keep correspondance throughout the sequence
    """
    Input:
        points: pointcloud data, [N, 3]
    Return:
        points: farthest sampled pointcloud, [npoint]
    """
    # xyz = points[0] #use the first frame for sampling
    xyz = points
    if stochastic_sample:
        assert len(points)/stochastic_sample_ratio_inv >= npoint
        xyz = stochastic_vec_sample_proportional(xyz, stochastic_sample_ratio_inv)
    N, C = xyz.shape
    centroids = np.zeros(npoint)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    idxs = np.array(farthest)[None]

    for i in range(npoint-1):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = ne.evaluate('sum((xyz - centroid) ** 2, 1)')
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.array(np.argmax(distance, -1))[None]
        idxs = np.concatenate([idxs, farthest])


    return points[idxs]


def fps(n_points, points):
    # returns farthers point distance sampling
    # shuffle each sequence individually but keep correspondance throughout the sequence
    """
    Input:
        points: pointcloud data, [N, 3]
    Return:
        points: farthest sampled pointcloud, [npoint]
    """
    # xyz = points[0] #use the first frame for sampling
    xyz = points
    print(xyz.shape)
    N, C = xyz.shape
    centroids = np.zeros(n_points)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    idxs = np.array(farthest)[None]

    for i in range(n_points-1):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.array(np.argmax(distance, -1))[None]
        idxs = np.concatenate([idxs, farthest])

    return points[idxs]


def read16BitPGM(pgm_dir):
    """Return a raster of integers from a PGM as a list of lists."""
    with open(pgm_dir, 'rb') as pgmf:
        header = pgmf.readline()
        assert header[:2] == b'P5'
        size = pgmf.readline()
        (width, height) = [int(i) for i in size.split()]
        depth = int(pgmf.readline())

        assert depth <= 65535

        raster = []
        for y in range(height):
           row = []
           for y in range(width):
               cv2.imread('a.pgm', -1)
               ###############
               # low_bits =int.from_bytes(pgmf.read(2), byteorder='big')
               # row.append(low_bits)
               ###############
               # low_bits = ord(pgmf.read(1))
               # row.append(low_bits+255*ord(pgmf.read(1)))
               ###############
           raster.append(row)
        # print(raster)
        return raster


def addTextToImg(img_path, txt="null"):
    # Call draw Method to add 2D graphics in an image
    image = Image.open(img_path)
    I1 = ImageDraw.Draw(image)
    # font = ImageFont.load_default()
    # Add Text to an image
    font = ImageFont.truetype(font="arial.ttf", size = 42)

    I1.text((28, 36), txt, fill=(255, 0, 0), font=font)

    # Define a transform to convert PIL
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)
    return img_tensor


def getNumFrames(_dir_):
    pv_dir = os.path.join(_dir_, "pv")
    if not os.path.exists(pv_dir):
        print(f"dir does not exist: {pv_dir}")
    assert os.path.exists(pv_dir)
    return len(os.listdir(pv_dir))

def searchForAnnotationJson(_dir_, dataset_dir):
    _dir_ = os.path.join(dataset_dir, _dir_)
    for sub_dir in os.listdir(_dir_):
        # sub_dir = os.path.join(dataset_dir, sub_dir)
        # print("lets look at {}".format(sub_dir))
        if "annotations.json" in sub_dir[-16:]:
            # print("found {}".format(sub_dir))
            return os.path.join(_dir_, sub_dir)

    return None


def translateSecToFrame(start_sec, end_sec,  fps_error_correction_mapping):
    last_frame_vid = list(fps_error_correction_mapping.keys())[-1]
    if end_sec > float(last_frame_vid)/15:
        end_sec = float(last_frame_vid)/15
    if  start_sec > float(last_frame_vid)/15:
        print(start_sec, end_sec, float(last_frame_vid) / 15)
        return
    seg_start = fps_error_correction_mapping[str(int(np.round(15 * start_sec)))]
    seg_end = fps_error_correction_mapping[str(int(np.round(15 * end_sec)))]

    return [seg_start,seg_end ]


def applyMappingToAnnotations(action_annotation_list, id_to_name, fps_error_correction_mapping):
    action_labels = []
    if fps_error_correction_mapping:
        print("starting to decode using json!!")
    for encoded_annotation in action_annotation_list:
        if fps_error_correction_mapping:

            segment = translateSecToFrame(encoded_annotation["start"], encoded_annotation["end"], fps_error_correction_mapping)
            if not segment:
                print("=============", encoded_annotation["start"], encoded_annotation["end"], "=============")
                continue
            # print()
            # segment = [int(fps_error_correction_mapping[encoded_annotation["start"]]), int(fps_error_correction_mapping[encoded_annotation["end"]])]
            # print (segment)
        else:
            segment = [int(np.round(15 * encoded_annotation["start"])), int(np.round(15 * encoded_annotation["end"]))]
        label = id_to_name[encoded_annotation["action"]]
        decoded_label = {"segment": segment, "label": label}
        action_labels.append(decoded_label)
    return action_labels

def removeBackslashT(action_labels):
    for annotation_num in range(len(action_labels)):
        action_labels[annotation_num]["label"] = action_labels[annotation_num]["label"].replace("\t", " ")
        print(action_labels[annotation_num]["label"])


def getIdToNameMapping(action_label_data: list):
    id_to_name = {}
    for single_id_map in action_label_data:
        # print(single_id_map)
        _id_ = single_id_map["id"]
        name = single_id_map["name"]
        id_to_name[_id_] = name
    print(id_to_name)
    return id_to_name

def saveVideoClip(clip_name, clip_frames):
    video = cv2.VideoWriter(clip_name, 0, 15, (960, 540))
    for frame in clip_frames:
        transposed_frame = np.transpose(frame, (1, 2, 0))
        video.write(cv2.cvtColor(np.array(transposed_frame), cv2.COLOR_RGB2BGR))
    print("saved video: ",clip_name )
def decodeJsonAnnotations(current_json_annotation: dict, fps_error_correction_mapping):
    print(f"current_json_annotation: {current_json_annotation.keys()}")
    id_to_name = getIdToNameMapping(current_json_annotation["config"]["actionLabelData"])
    action_labels = applyMappingToAnnotations(current_json_annotation["annotation"]["actionAnnotationList"], id_to_name,
                                              fps_error_correction_mapping=fps_error_correction_mapping)
    removeBackslashT(action_labels)
    print(action_labels)
    return action_labels


def getAllJsonsInDirList(dir_list, merged_json, subset, dataset_dir):
    assert subset == "training" or subset == "testing"
    for _dir_ in dir_list:
        json_file = searchForAnnotationJson(_dir_, dataset_dir)
        if json_file:
            print("found json file {}".format(json_file))
            with open(json_file) as json_file_obj:
                current_json = json.load(json_file_obj)
                # add error check if json file already exists in database
                fps_error_correction_mapping = None
                fps_error_correction_json_path = os.path.join(os.path.join(dataset_dir, _dir_), "frame_rate_translation.json")
                if os.path.exists(fps_error_correction_json_path):
                    with open(fps_error_correction_json_path) as fps_error_correction_obj:
                        fps_error_correction_mapping = json.load(fps_error_correction_obj)

                print(f"@@@@  directory {_dir_} has a translation json @@@@")
                merged_json["database"][_dir_] = {"subset": subset,
                                                      "annotation": decodeJsonAnnotations(current_json,
                                                                                          fps_error_correction_mapping)}
        else:
            print(f"path {_dir_} does not have json yet")


def getAllJsonAnnotations(dataset_dir, merged_json=None):
    if merged_json is None or merged_json == {}:
        merged_json = {"version": "2.0.0", "database": {}}
    print(merged_json)

    # use this code for one test directory:
    # getAllJsonsInDirList([dataset_dir], merged_json, "testing")
    # use this code for real database:
    test_dir_list_file = os.path.join(dataset_dir, "indexing_files", "all_test_dir_list.txt")
    train_dir_list_file = os.path.join(dataset_dir, "indexing_files", "all_train_dir_list.txt")
    assert os.path.exists(test_dir_list_file) and os.path.exists(train_dir_list_file)
    test_dir_list = getListFromFile(test_dir_list_file)
    train_dir_list = getListFromFile(train_dir_list_file)
    getAllJsonsInDirList(test_dir_list, merged_json, "testing", dataset_dir)
    getAllJsonsInDirList(train_dir_list, merged_json, "training", dataset_dir)
    print(merged_json)

    with open(os.path.join(dataset_dir, "indexing_files", "db_gt_annotations.json"), "w") as new_json_file_obj:
        print(merged_json)
        merged_json = json.dumps(merged_json)
        new_json_file_obj.write(merged_json)
        return


def writeListToFile(filename, line_list):
    with open(filename, "w") as f:
        for line in line_list:
            # FIXME: maybe remove this \n
            f.writelines(line + "\n")
    return

def makeDirListAbs(abs_parent_dir, rel_dir_list):
    return [os.path.join(abs_parent_dir, rel_dir) for rel_dir in rel_dir_list if abs_parent_dir not in rel_dir]

def getListFromFile(filename):
    """
    retrieve a list of lines from a .txt file
    :param :
    :return: list of atomic actions
    """
    with open(filename) as f:
        line_list = f.read().splitlines()
    # line_list.sort()
    return line_list


def getSepereateFurnitureRecDirLists(dataset_dir):
    furniture_sep_rec_dir_list = [
        (
        furniture_name, os.path.join(dataset_dir, "indexing_files", "{}_recording_dir_list.txt".format(furniture_name)))
        for furniture_name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, furniture_name)) and furniture_name != "indexing_files"]
    return furniture_sep_rec_dir_list


def createTrainTestFiles(dataset_dir, train_ratio=0.7):
    furniture_sep_rec_dir_list = getSepereateFurnitureRecDirLists(dataset_dir)
    print(furniture_sep_rec_dir_list)
    all_train_recordings = []
    all_test_recordings = []

    for furniture_name, idx_file_path in furniture_sep_rec_dir_list:
        # check that all furniture indexing files are created.
        print("rec path:", idx_file_path)
        assert (os.path.exists(idx_file_path))
        recording_dir_list = getListFromFile(idx_file_path)
        random.shuffle(recording_dir_list)
        num_furniture_recordings = len(recording_dir_list)
        num_train_recordings = int(train_ratio * num_furniture_recordings)
        num_test_recordings = num_furniture_recordings - num_train_recordings
        train_rec_list = recording_dir_list[:num_train_recordings]
        test_rec_list = recording_dir_list[num_train_recordings:]
        all_train_recordings += train_rec_list
        all_test_recordings += test_rec_list
        writeListToFile(os.path.join(dataset_dir, "indexing_files", "{}_train_dir_list.txt".format(furniture_name)),
                        train_rec_list)
        writeListToFile(os.path.join(dataset_dir, "indexing_files", "{}_test_dir_list.txt".format(furniture_name)),
                        test_rec_list)

    writeListToFile(os.path.join(dataset_dir, "indexing_files", "all_train_dir_list.txt"),
                    all_train_recordings)
    writeListToFile(os.path.join(dataset_dir, "indexing_files", "all_test_dir_list.txt"),
                    all_test_recordings)


def aux_createAllRecordingDirList(dataset_dir, target_file):
    if "_recDir" in dataset_dir[-8:]:
        with open(target_file, "a") as all_rec_idx_file:
            all_rec_idx_file.write(dataset_dir + "\n")
        return

    for sub_dir in glob(rf"{dataset_dir}\*\\"):
        print(f"calling aux_createAllRecordingDirList for path: {sub_dir}, continuing search for recording dir")
        aux_createAllRecordingDirList(sub_dir, target_file)


def createAllRecordingDirList(dataset_dir, target_file):
    if os.path.exists(target_file):
        os.remove(target_file)

    print(f"calling aux_createAllRecordingDirList for top path: {dataset_dir}, continuing search for recording dir")
    aux_createAllRecordingDirList(dataset_dir, target_file)


def getNumRecordings(w_path):
    """
    recursively iterate over directory to get number of sub dirs containing recordings
    """
    return len([f for f in Path(w_path).iterdir() if f.is_dir() and "_recDir" in str(w_path)[-8:]])


def removeOriginalPvImages(w_path):
    orig_pv_path = w_path / "pv"
    assert (w_path / "norm" / "pv").exists()
    orig_pv_images = [f.path for f in os.scandir(orig_pv_path) if os.path.splitext(f)[-1] == '.png']
    for orig_pv_img in orig_pv_images:
        os.remove(orig_pv_img)


def squeeze_class_names(class_names):
    """
    shortens the class names for visualization

    :param class_names: class name
    :return: class_names: shortened class names
    """

    class_names = [substring.split(" ") for substring in class_names]
    for i, cls in enumerate(class_names):
        if len(cls) <= 4:
            class_names[i] = " ".join(cls)
        else:
            class_names[i] = " ".join(cls[0:4]) + "..."
    return class_names


def accume_per_video_predictions(vid_idx, frame_pad, pred_labels_per_video, logits_per_video, pred_labels,
                                 logits, frames_per_clip):
    """
    This is a helper function to accumulate the predictions of teh different batches into a single list
    containing the predictions for each video separately. It is used in all of the test files except the frame based
    (no sequence)
    Parameters
    ----------
    vid_idx : list of video index corresponding for each element in the batch
    frame_pad : list of number of padded frames per element in the batch
    pred_labels_per_video : predicted labels per video - accumulated from previous batch
    logits_per_video : logits per video - accumulated from previous batch
    pred_labels : the current batch predictions
    logits : the current batch logits
    frames_per_clip : number of frames per clip (int)
    Returns
    -------
        pred_labels_per_video : predicted labels per video - accumulated from previous batch
        logits_per_video : logits per video - accumulated from previous batch
    """

    for i in range(len(vid_idx)):
        batch_vid_idx = vid_idx[i].item()
        batch_frame_pad = frame_pad[i].item()

        pred_labels_per_video[batch_vid_idx].extend(pred_labels[i*frames_per_clip:(i+1)*frames_per_clip])
        if not batch_frame_pad == 0:
            pred_labels_per_video[batch_vid_idx] = pred_labels_per_video[batch_vid_idx][0:-batch_frame_pad]


        logits_per_video[batch_vid_idx].extend(logits[i*frames_per_clip:(i+1)*frames_per_clip])
        if not batch_frame_pad == 0:
            logits_per_video[batch_vid_idx] = logits_per_video[batch_vid_idx][0:-batch_frame_pad]

    return pred_labels_per_video, logits_per_video

def convert_frame_logits_to_segment_json(logits, json_filename, video_name_list, action_list, mode="logits",
                                         details=""):
    """
    convert dataset per frame logits to scored segments and save to .json file in the ActivityNet format
    for action localization evaluation
    http://activity-net.org/challenges/2020/tasks/anet_localization.html
    Parameters
    ----------
    logits : per frame logits or labels (depending on mode)
    json_filename : output .json file name (full path)
    video_name_list : list of video names
    action_list : list of action labels
    mode : logits | labels
     Returns
    -------
    """
    json_dict_to_write = {"version": "VERSION 1.3"}

    results = {}
    for i, vid_logits in enumerate(logits):
        segments, scores, segment_labels = find_label_segments_and_scores(vid_logits, mode=mode)
        list_of_result_dicts = []
        vid_name = video_name_list[i]
        for j, segment in enumerate(segments):
            list_of_result_dicts.append({"label": action_list[segment_labels[j]], "score": float(scores[j]),
                                         "segment": segment.tolist()})
        results[vid_name] = list_of_result_dicts
    json_dict_to_write["results"] = results
    json_dict_to_write["external_data"] = {"details": details}
    with open(json_filename, 'w') as outfile:
        json.dump(json_dict_to_write, outfile)

def find_label_segments_and_scores(input, mode="logits"):
    """
    finds the range of label segments in a given labels array and computes the segment score as max logit average
    Parameters
    ----------
    logits : numpy array of per frame logits
    mode : logits | labels
    Returns
    -------
    ranges : list of segments ranges
    scores : segment score (logits average)
    segment_label : the label of the corresponding segment
    """
    if mode == "logits":
        logits = input
        labels = np.argmax(logits, axis=1)
    else:
        labels = input

    diff = np.diff(labels)
    iszero = np.concatenate(([0], np.equal(diff, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    segments = np.where(absdiff == 1)[0].reshape(-1, 2)
    scores = []
    segment_label = []
    for segment in segments:
        segment_label.append(labels[segment[0]])
        if mode == "logits":
            scores.append(np.mean(logits[segment[0]:segment[1], labels[segment[0]]]))
        else:
            scores.append(1.0)
    return segments, np.array(scores), segment_label
