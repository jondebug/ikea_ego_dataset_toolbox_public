"""
 Copyright (c) Microsoft. All rights reserved.
 This code is licensed under the MIT License (MIT).
 THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
 ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
 IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
 PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
"""
import cv2
import argparse
import numpy as np
from pathlib import Path
import ast

from raw_dataset_processing.raw_dataset_procassing_utils import load_head_hand_eye_data, load_pv_data, matchTimestamp_arg, get_eye_gaze_point


def process_timestamps(path):
    with open(path) as f:
        lines = f.readlines()
    print('Num timestamps:', len(lines))
    return np.array([int(elem) for elem in lines if len(elem)])


def project_hand_eye_to_pv(folder):
    print("projecting hand eye")
    head_hat_stream_path = list(folder.glob('*_eye.csv'))[0]
    print(f"opening eye data file {head_hat_stream_path}")

    pv_info_path = list(folder.glob('*pv.txt'))[0]
    pv_paths = sorted(list((folder / 'PV').glob('*png')))
    if len(pv_paths) == 0:
        print(f"this is an empty recording: {folder}")
        return -1

    # load head, hand, eye data
    (timestamps, _,
     left_hand_transs, left_hand_transs_available,
     right_hand_transs, right_hand_transs_available,
     gaze_data, gaze_available) = load_head_hand_eye_data(head_hat_stream_path)

    eye_str = " and eye gaze" if np.any(gaze_available) else ""

    # load pv info
    (frame_timestamps, focal_lengths, pv2world_transforms,
     ox, oy, width, height) = load_pv_data(pv_info_path)
    principal_point = np.array([ox, oy])
    n_frames = len(pv_paths)
    output_folder = folder / 'eye_hands'
    output_folder.mkdir(exist_ok=True)
    for pv_id in range(min(n_frames, len(focal_lengths))):
        print(".", end="", flush=True)
        pv_path = pv_paths[pv_id]
        sample_timestamp = int(str(pv_path.name).replace('.png', ''))

        hand_ts = matchTimestamp_arg(sample_timestamp, timestamps)
        # print('Frame-hand delta: {:.3f}ms'.format((sample_timestamp - timestamps[hand_ts]) * 1e-4))

        img = cv2.imread(str(pv_path))
        # pinhole
        K = np.array([[focal_lengths[pv_id][0], 0, principal_point[0]],
                      [0, focal_lengths[pv_id][1], principal_point[1]],
                      [0, 0, 1]])
        try:
            Rt = np.linalg.inv(pv2world_transforms[pv_id])

        except np.linalg.LinAlgError:
            print('No pv2world transform')
            continue

        rvec, _ = cv2.Rodrigues(Rt[:3, :3])
        tvec = Rt[:3, 3]

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        hands = [(left_hand_transs, left_hand_transs_available),
                 (right_hand_transs, right_hand_transs_available)]
        for hand_id, hand in enumerate(hands):
            transs, avail = hand
            if avail[hand_ts]:
                for joint_num, joint in enumerate(transs[hand_ts]):

                    hand_tr = joint.reshape((1, 3))
                    #print(Rt, hand_tr, rvec, tvec, K)
                    xy, _ = cv2.projectPoints(hand_tr, rvec, tvec, K, None)
                    #print(xy[0][0][0], xy[0][0][1], "\nthis was the point")
                    ixy = (int(xy[0][0][0]), int(xy[0][0][1]))
                    ixy = (width - ixy[0], ixy[1])
                    img = cv2.circle(img, ixy, radius=3, color=colors[hand_id])

        if gaze_available[hand_ts]:
            point = get_eye_gaze_point(gaze_data[hand_ts])
            xy, _ = cv2.projectPoints(point.reshape((1, 3)), rvec, tvec, K, None)
            ixy = (int(xy[0][0][0]), int(xy[0][0][1]))
            ixy = (width - ixy[0], ixy[1])
            img = cv2.circle(img, ixy, radius=3, color=colors[2])
        if pv_id % 500 == 0:
            print(f"saving picture number {pv_id}")
        #cv2.imwrite(str(output_folder / 'hands') + 'proj{}.png'.format(str(sample_timestamp).zfill(4)), img)
        cv2.imwrite(f"{output_folder}/{str(sample_timestamp)}.png", img)

if __name__ == "__main__":
    # pass the path to folder being processed
    parser = argparse.ArgumentParser(description='Process recorded data.')
    parser.add_argument("--recording_path", required=True,
                        help="Path to recording folder")

    args = parser.parse_args()
    project_hand_eye_to_pv(Path(args.recording_path))
