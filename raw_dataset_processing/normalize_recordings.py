from pathlib import Path
from raw_dataset_processing.raw_dataset_procassing_utils import getPvTimestamps, getDepthTimestamps,\
    getHandEyeTimestamps, matchTimestamp, build_normalized_data_dir, removeOriginalPlyFiles, copyRenamePvImage,\
    copyRenameDepthImage, copyRenameHandEyeImage, processHandEyeData
from glob import glob
import numpy as np
import os
# from raw_dataset_processing.project_hand_eye_to_pv import processHandEyeData


def create_processed_had_eye_csv(w_path):
    assert (w_path / "norm").exists()
    if (not (w_path / "norm" / "norm_proc_hand_data.csv").exists()) or (not (w_path / "norm" / "norm_proc_eye_data.csv").exists() ):
        if (w_path / "PV").exists():
            if processHandEyeData(w_path) == -1:
                return -1
    else:
        print(f"{w_path} already has proc_norm_hand_eye_data.csv file!! moving on ")


def removeOriginalProcessedData(w_path, sensor_name="Depth Long Throw"):

    pv_dir = w_path / "PV"
    depth_dir = w_path / "{}".format(sensor_name)
    norm_depth_dir = w_path / "norm" / "{}".format(sensor_name)
    norm_pv_dir = w_path / "norm" / "PV"

    assert pv_dir.exists() and depth_dir.exists() and (w_path / "head_hand_eye.csv").exists()

    if norm_depth_dir.exists() and norm_pv_dir.exists():
        norm_rgb_images = [f.path for f in os.scandir(norm_pv_dir)]
        norm_depth_ply_files = [f.path for f in os.scandir(norm_depth_dir) if os.path.splitext(f)[-1] == '.ply']
        norm_depth_ply_folder_size = len(norm_depth_ply_files)
        norm_rgb_folder_size = len(norm_rgb_images)
        print(w_path, norm_depth_ply_folder_size, norm_rgb_folder_size)
        assert norm_depth_ply_folder_size > 1000
        assert norm_rgb_folder_size > 1000
        assert norm_rgb_folder_size == norm_depth_ply_folder_size
        print(f"starting to delete ply files from dir {depth_dir}")
        removeOriginalPlyFiles(w_path, sensor_name)
        #removeOriginalPvImages(w_path) #optional


    else:
        print(f" dir {depth_dir} does not yet have normalizes data")



def createNormalizedFiles(rec_dir, pv_to_depth_hand_eye_mapping: dict, sensor_name="Depth Long Throw"):
    w_path = Path(rec_dir)
    pv_dir = w_path / "PV"
    depth_dir = w_path / "{}".format(sensor_name)
    assert pv_dir.exists() and depth_dir.exists() and (w_path / "head_hand_eye.csv").exists()
    build_normalized_data_dir(w_path)
    for frame_number, pv_timestamp in enumerate(pv_to_depth_hand_eye_mapping.keys()):
        depth_ts, hand_eye_ts = pv_to_depth_hand_eye_mapping[pv_timestamp]
        copyRenamePvImage(w_path, pv_timestamp, frame_number)
        copyRenameDepthImage(w_path, depth_ts, frame_number)

    copyRenameHandEyeImage(w_path, pv_to_depth_hand_eye_mapping)
    print("normalized recording data done.")


def createPVtoDepthHandEyeMapping(rec_dir, depth_path_suffix='', sensor_name="Depth Long Throw"):
    # sub_dir_lst = glob(rf"{rec_dir}\*\\")
    w_path = Path(rec_dir)
    pv_dir = w_path / "PV"
    depth_dir = w_path / "{}".format(sensor_name)
    assert pv_dir.exists() and depth_dir.exists() and (w_path / "head_hand_eye.csv").exists()

    pv_timestamps = getPvTimestamps(w_path)
    depth_timestamps = getDepthTimestamps(w_path, sensor_name, depth_path_suffix)
    hand_eye_timestamps = getHandEyeTimestamps(w_path)
    # print(f'found the following {len(pv_timestamps)} PV timestamps: {pv_timestamps}')
    # print(f'found the following {len(depth_timestamps)} depth timestamps: {depth_timestamps}')

    pv_to_depth_hand_eye_mapping = {}
    for frame_number, pv_timestamp in enumerate(pv_timestamps):
        matching_depth_ts = matchTimestamp(target=pv_timestamp, all_timestamps=depth_timestamps)
        matching_hand_eye_ts = matchTimestamp(target=pv_timestamp, all_timestamps=hand_eye_timestamps)
        pv_to_depth_hand_eye_mapping[pv_timestamp] = (matching_depth_ts, matching_hand_eye_ts)
    print("got mapping. creating normalized data dir for recording")
    return pv_to_depth_hand_eye_mapping

def checkNormalized(path, sensor_name="Depth Long Throw"):
    if not os.path.exists(os.path.join(path, "norm")):
        return False
    return True

def normalizeAllRecordingsInPath(path, sensor_name="Depth Long Throw"):
    if "_recDir" in path[-30:]:
        # if checkNormalized(path):
        #     # TODO: remove this check because it does not check if a folder is really normalized properly
        #     print(f"{path} already normalized (or atleast the norm folder exists) ")
        #     return
        w_path = Path(path)
        pv_to_depth_hand_eye_mapping = createPVtoDepthHandEyeMapping(path)
        createNormalizedFiles(rec_dir=path, pv_to_depth_hand_eye_mapping=pv_to_depth_hand_eye_mapping)
        # removeOriginalProcessedData(Path(path), sensor_name)
        create_processed_had_eye_csv(w_path)
        return

    for sub_dir in glob(rf"{path}\*\\"):
        print(f"calling process_all_recordings_in_path for path: {sub_dir}, continuing search for recording dir")
        normalizeAllRecordingsInPath(sub_dir)


if __name__ == '__main__':
    # w_path = Path(r'C:\HoloLens')
    normalizeAllRecordingsInPath(r'C:\HoloLens\Stool')
