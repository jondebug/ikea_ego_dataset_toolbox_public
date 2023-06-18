import argparse

from raw_dataset_processing.process_all import process_all_recordings_in_path
from raw_dataset_processing.visualize import visualize_all_recordings_in_path
from raw_dataset_processing.normalize_recordings import normalizeAllRecordingsInPath

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process recorded data.')
    parser.add_argument("--recording_path", required=True,
                        help="Path to recording folder")
    parser.add_argument("--project_hand_eye",
                        required=False,
                        action='store_true',
                        help="Project hand joints (and eye gaze, if recorded) to rgb images")

    args = parser.parse_args()

    # process_all_recordings_in_path(path=args.recording_path, project_hand_eye=args.project_hand_eye)
    visualize_all_recordings_in_path(args.recording_path)
    normalizeAllRecordingsInPath(args.recording_path)



