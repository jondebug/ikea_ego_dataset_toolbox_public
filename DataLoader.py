import numpy as np
import os
import torchvision
import torch
from i3d.ego_i3d import videotransforms, i3d_utils
from i3d.ego_i3d.i3d_utils import probabilisticShuffleClipFrames
from utils import getNumRecordings, getListFromFile, getNumFrames, addTextToImg, imread_pgm
import json
import plotly.express as px
import pandas as pd
import plyfile
import pickle
from torch.utils.data import Dataset


class IKEAEgoDatasetPickleClips(Dataset):
    """
    IKEA Action Dataset class with pre-saved clips into pickles
    """

    def __init__(self, dataset_path, set='train', train_trans=None, test_trans=None):
        # TODO add support for point cloud downsampling using FPS and random sampling
        self.dataset_path = dataset_path
        self.set = set
        self.files_path = os.path.join(dataset_path, set)
        self.file_list = self.absolute_file_paths(self.files_path)
        self.file_list.sort()
        self.rgb_transform = train_trans if set == 'train' else test_trans
        # backwards compatibility
        with open(os.path.join(self.dataset_path, set + '_aux.pickle'), 'rb') as f:
            aux_data = pickle.load(f)
        self.clip_set = aux_data['clip_set']
        self.clip_label_count = aux_data['clip_label_count']
        self.num_classes = aux_data['num_classes']
        self.video_list = aux_data['video_list']
        self.action_list = aux_data['action_list']
        self.frames_per_clip = aux_data['frames_per_clip']
        self.action_labels = aux_data['action_labels']
        print("{}set contains {} clips".format(set, len(self.file_list)))

    def absolute_file_paths(self, directory):
        path = os.path.abspath(directory)
        return [entry.path for entry in os.scandir(path) if entry.is_file()]

    def transform_rgb_frames(self, data):
        rgb_frames = data['inputs']
        frames = torch.from_numpy(rgb_frames)

        # (t,h,w,c)
        if self.rgb_transform is not None:
            # TODO: apply transform iteratively to frames
            transformed_frames = torch.permute(frames, (0, 2, 3, 1))
            transformed_frames = self.rgb_transform(transformed_frames)
            transformed_frames = torch.permute(transformed_frames, (0, 3, 1, 2))

        data['inputs'] = transformed_frames
        return data

    def get_dataset_statistics(self):
        label_count = np.zeros(len(self.action_list))
        for i in range(len(self.file_list)):
            data = self.__getitem__(i)
            label_count = label_count + data[1].sum(-1)
        return label_count

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        with open(self.file_list[index], 'rb') as f:
            data = pickle.load(f)
        data = self.transform_rgb_frames(data)
        return {"rgb_frames": data['inputs']}, data['labels'], data['vid_idx'], data['frame_pad']


class HololensStreamRecBase():
    """Face Landmarks dataset."""

    def __init__(self, dataset_path, furniture_list: list, action_list_filename='action_list.txt',
                 train_filename='all_train_dir_list.txt', test_filename='all_test_dir_list.txt', rgb_transform=None,
                 gt_annotation_filename='db_gt_annotations.json'):
        """
        Args:
            action_list_filename (string): Path to the csv file with annotations.
            dataset_path (string): Root directory with all the data.
            furniture_list = list of strings containing names of furniture assembled in dataset.
            rgb_transform (callable, optional): Optional rgb_transform to be applied
            on a sample.
        """

        self.dataset_root_path = dataset_path
        self.furniture_dirs = [os.path.join(dataset_path, furniture_name) for furniture_name in furniture_list]

        for furniture_dir in self.furniture_dirs:
            assert os.path.exists(furniture_dir)

        self.furniture_dir_sizes = [getNumRecordings(_dir_) for _dir_ in self.furniture_dirs]
        self.num_recordings = sum(self.furniture_dir_sizes)

        # indexing_files:
        self.gt_annotation_filename = os.path.join(dataset_path, 'indexing_files', gt_annotation_filename)
        self.action_list_filename = os.path.join(dataset_path, 'indexing_files', action_list_filename)
        self.train_filename = os.path.join(dataset_path, 'indexing_files', train_filename)
        self.test_filename = os.path.join(dataset_path, 'indexing_files', test_filename)

        # load lists from files:
        print(f"getting actions from file: {self.action_list_filename}")
        self.action_list = getListFromFile(self.action_list_filename)
        print(f"got action list:{self.action_list}")
        self.action_list.sort()
        if "N/A" in self.action_list:
            self.action_list.remove("N/A")

        self.action_list.insert(0, "N/A")  # 0 label for unlabled frames
        self.num_classes = len(self.action_list)
        self.train_video_list = getListFromFile(self.train_filename)
        self.test_video_list = getListFromFile(self.test_filename)
        self.all_video_list = self.test_video_list + self.train_video_list
        self.action_name_to_id_mapping = {}
        self.id_to_action_name_mapping = {}
        for action_id, action in enumerate(self.action_list):
            self.action_name_to_id_mapping[action] = action_id
            self.id_to_action_name_mapping[action_id] = action

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return

    def __len__(self):
        return

    def get_video_info_table(self, dataset_type):
        """
        fetch the annotated videos table from the database
        :param : device: string ['all', 'dev1', 'dev2', 'dev3']
        :return: annotated videos table
        """
        video_data_table = []
        if dataset_type == "all":
            rec_list = self.all_video_list

        elif dataset_type == "train":
            rec_list = self.train_video_list

        elif dataset_type == "test":
            rec_list = self.test_video_list
        else:
            raise ValueError("Invalid dataset name")

        for _dir_ in rec_list:
            row = {"nframes": getNumFrames(os.path.join(self.dataset_root_path, _dir_)), 'video_path': _dir_}
            video_data_table.append(row)

        return video_data_table

    def get_video_annotations_table(self, video_path):
        with open(self.gt_annotation_filename) as json_file_obj:
            db_gt_annotations = json.load(json_file_obj)

        if video_path in db_gt_annotations["database"].keys():
            return db_gt_annotations["database"][video_path]["annotation"]
        else:
            return None

    def get_video_table(self, video_idx):
        """
        fetch the video information row from the video table in the database
        :param :  video_idx: index of the desired video
        :return: video information table row from the databse
        """
        return self.cursor_annotations.execute('''SELECT * FROM videos WHERE id = ?''', (video_idx,))

    def get_annotation_table(self):
        """
        :return: full annotations table (for all videos)
        """
        return self.cursor_annotations.execute('''SELECT * FROM annotations ''')


class HololensStreamRecClipDataset(HololensStreamRecBase):
    def __init__(self, dataset_path, furniture_list: list = [], action_list_filename='action_list.txt',
                 train_filename='all_train_dir_list.txt', test_filename='all_test_dir_list.txt', rgb_transform=None,
                 gt_annotation_filename='db_gt_annotations.json', modalities=["all"], frame_skip=1, frames_per_clip=32,
                 dataset="train", rgb_label_watermark=False, furniture_mod=["all"], smallDataset=False,
                 eye_crop_transform=False, rgb_reshape_factor=1, offcenter_variance = 20, flip_prob=0, shuffle_prob=0,
                 apply_augmentation = False):

        super().__init__(dataset_path, furniture_list, action_list_filename,
                         train_filename, test_filename, rgb_transform, gt_annotation_filename)


        self.apply_augmentation = apply_augmentation
        self.flip_prob = flip_prob
        self.shuffle_prob = shuffle_prob
        self.offcenter_variance = offcenter_variance
        self.eye_crop_transform = eye_crop_transform
        self.smallDataset = smallDataset
        self.rgb_reshape_factor = rgb_reshape_factor
        self.furniture_mod = furniture_mod
        self.rgb_label_watermark = rgb_label_watermark
        self.modalities = modalities
        self.rgb_transform = rgb_transform
        self.set = dataset
        self.frame_skip = frame_skip
        self.frames_per_clip = frames_per_clip

        if self.set == 'train':
            self.video_list = self.filterFurnitureModalities(self.train_video_list)
        elif self.set == 'test':
            self.video_list = self.filterFurnitureModalities(self.test_video_list)
        else:
            raise ValueError("Invalid set name")
        self.annotated_videos = self.get_video_frame_labels()
        self.clip_set, self.clip_label_count = self.get_clips()
        self.action_labels = [vid_data[1].transpose() for vid_data in self.annotated_videos]

        labels = []
        clip_labels_count = []
        for i, label in enumerate(self.action_list):
            print((label, self.clip_label_count[i]))

            labels.append(label)
            clip_labels_count.append(self.clip_label_count[i])

        # Create a dataframe with the bin names and values
        df = pd.DataFrame({'bin': labels, 'count': clip_labels_count})
        # Create a histogram figure
        fig = px.histogram(df, x='bin', y='count', title='Histogram')
        fig.write_html("D:\loaded_clips\LabeledFramesPerAction.html")

    def filterFurnitureModalities(self, rec_list):
        if self.furniture_mod == ["all"]:
            return rec_list
        filtered_rec_list = []
        for furniture_name in self.furniture_mod:
            for rec in rec_list:
                if f"\HoloLens\\{furniture_name}\\" in rec:
                    filtered_rec_list.append(rec)
        return filtered_rec_list

    def get_video_frame_labels(self):
        # Extract the label data from the database
        # outputs a dataset structure of (video_path, multi-label per-frame, number of frames in the video)
        video_info_table = self.get_video_info_table(self.set)
        vid_list = []

        for row in video_info_table:
            n_frames = int(row["nframes"])
            video_path = row['video_path']

            rec_frame_labels = np.zeros((self.num_classes, n_frames), np.float32)  # allow multi-class representation
            rec_frame_labels[0, :] = np.ones((1, n_frames),
                                             np.float32)  # initialize all frames as NA
            annotation_table = self.get_video_annotations_table(video_path)
            if not annotation_table:
                # reached an unannotated directory
                continue
            for ann_row in annotation_table:
                action = ann_row["label"]  # map the labels
                action_id = self.action_name_to_id_mapping[action]

                start_frame = ann_row['segment'][0]
                end_frame = ann_row['segment'][1]
                end_frame = end_frame if end_frame < n_frames else n_frames
                if action is not None:
                    rec_frame_labels[:, start_frame:end_frame] = 0  # remove the N/A
                    rec_frame_labels[action_id, start_frame:end_frame] = 1
            vid_list.append(
                (video_path, rec_frame_labels, n_frames))  # 0 = duration - irrelevant for initial tests, used for start
        return vid_list

    def get_clips(self):
        # extract equal length video clip segments from the full video dataset
        clip_dataset = []
        label_count = np.zeros(self.num_classes)
        # for i, data in enumerate(self.annotated_videos):
        for i, data in enumerate(self.annotated_videos):
            n_frames = data[2]
            n_clips = int(n_frames / (self.frames_per_clip * self.frame_skip))
            # remaining_frames = n_frames % (self.frames_per_clip * self.frame_skip)
            for j in range(0, n_clips):
                for k in range(0, self.frame_skip):
                    start = j * self.frames_per_clip * self.frame_skip + k
                    end = (j + 1) * self.frames_per_clip * self.frame_skip
                    label = data[1][:, start:end:self.frame_skip]

                    label_count = label_count + np.sum(label, axis=1)
                    frame_ind = np.arange(start, end, self.frame_skip).tolist()
                    clip_dataset.append((data[0], label, frame_ind, self.frames_per_clip, i, 0))

        return clip_dataset, label_count

    def getLabelsInClipIdx(self, np_labels):
        action_strings = [self.id_to_action_name_mapping[np.argmax(np_labels[i])] for i in range(len(np_labels))]
        return action_strings

    def load_point_clouds_2(self, rec_dir, frame_indices):
        frames = []
        for counter in frame_indices:
            target_points = torch.zeros(92160, 9)
            if not self.smallDataset:
                point_cloud_full_path = os.path.join(rec_dir, "norm", "Depth Long Throw", "{}.ply".format(i))
            else:
                point_cloud_full_path = os.path.join(rec_dir, "Depth Long Throw", "{}.ply".format(i))

            plydata = plyfile.PlyData.read(point_cloud_full_path)
            d = np.asarray(plydata['vertex'].data)
            pc = np.column_stack([d[p.name] for p in plydata['vertex'].properties])
            target_points[:pc.shape[0], :] = torch.from_numpy(pc)
            frames.append(target_points)
        return torch.stack(frames, 0)

    def load_point_clouds(self, rec_dir, frame_indices):
        point_clouds = []
        for index in frame_indices:
            if not self.smallDataset:
                point_cloud_full_path = os.path.join(rec_dir, "norm", "Depth Long Throw", "{}.ply".format(index))
            else:
                point_cloud_full_path = os.path.join(rec_dir, "Depth Long Throw", "{}.ply".format(index))

            target_points = torch.zeros(92160, 9)
            ply_data = plyfile.PlyData.read(point_cloud_full_path)
            points = ply_data['vertex'].data
            # [(),(),()]=>[[],[],[]]
            points = [list(point) for point in points]
            target_points[:len(points), :] = torch.tensor(points)
            point_clouds.append(target_points)
        return torch.stack(point_clouds)

    def load_data_frames_from_csv(self, rec_dir, frame_indices, filename, modality):
        if not self.smallDataset:
            full_rec_csv_path = os.path.join(rec_dir, "norm", filename)
        else:
            full_rec_csv_path = os.path.join(rec_dir, filename)

        with open(full_rec_csv_path, "rb") as full_rec_csv_f:
            clip_data = np.loadtxt(full_rec_csv_f, delimiter=",")[frame_indices, :]
            if modality == 'eye_data_frames' and self.rgb_reshape_factor>1:
                clip_data[:, -2:] = (clip_data[:, -2:]/self.rgb_reshape_factor).astype(dtype=int)
            if modality == 'hand_data_frames' and self.rgb_reshape_factor > 1:
                assert False #unhandled case
        return torch.Tensor(clip_data)

    def load_depth_frames(self, rec_dir, frame_indices):
        depth_frames = []
        for index in frame_indices:
            if not self.smallDataset:
                depth_frame_full_path = os.path.join(rec_dir, "norm", "Depth Long Throw", "{}.pgm".format(index))
            else:
                depth_frame_full_path = os.path.join(rec_dir, "Depth Long Throw", "{}.pgm".format(index))

            pgm_data = imread_pgm(depth_frame_full_path)
            depth_frames.append(pgm_data)
        return torch.Tensor(depth_frames)

    def load_rgb_frames(self, rec_dir, frame_indices, labels):
        # load video file and extract the frames
        np_labels = np.array(labels).T
        frames = []
        str_labels = self.getLabelsInClipIdx(np_labels)
        assert (self.rgb_label_watermark and len(labels) > 0) or not self.rgb_label_watermark
        for frame_num in frame_indices:
            if not self.smallDataset:

                rgb_frame_full_path = os.path.join(rec_dir, "norm", "pv", "{}.png".format(frame_num))
            else:
                rgb_frame_full_path = os.path.join(rec_dir, "pv", "{}.png".format(frame_num))
            if not os.path.exists(rgb_frame_full_path):
                print(f"rgb frame path does not exist: {rgb_frame_full_path} ")
            assert os.path.exists(rgb_frame_full_path)
            if self.rgb_label_watermark:
                frame = addTextToImg(rgb_frame_full_path, str_labels[frame_num - frame_indices[0]] + f", {frame_num}")
            else:
                frame = torch.Tensor(torchvision.io.read_image(rgb_frame_full_path))
            frames.append(frame)

        frames = torch.stack(frames)
        # (t,h,w,c)
        if self.eye_crop_transform == True:
            assert 'eye_data_frames' in self.modalities
            eye_focus_points = self.load_data_frames_from_csv(
                rec_dir, frame_indices, filename="norm_proc_eye_data.csv", modality='eye_data_frames')
            frames = i3d_utils.offcenterCrop(frames, eye_focus_points[:, -2:], random_offset=self.set == 'train', offcenter_variance=self.offcenter_variance)
            return frames
        elif self.rgb_transform is not None:
            # TODO: apply transform iteratively to frames
            frames = torch.permute(frames, (0, 2, 3, 1))
            frames = self.rgb_transform(frames)
            frames = torch.permute(frames, (0, 3, 1, 2))
        if self.set == 'train' and self.apply_augmentation:
            frames = probabilisticShuffleClipFrames(frames, shuffle_prob=self.shuffle_prob)
            frames = videotransforms.RandomHorizontalFlip(self.flip_prob)(frames)
        return frames


    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.clip_set)

    def __getitem__(self, index):
        # 'Generate one sample of data'
        recording_full_path, labels, frame_ind, n_frames_per_clip, vid_idx, frame_pad = self.clip_set[index]
        recording_full_path = os.path.join(self.dataset_root_path, recording_full_path)

        clip_modalities_dict = {}
        if self.modalities == ["all"]:
            # returning all modalities
            clip_modalities_dict["rgb_frames"] = self.load_rgb_frames(recording_full_path, frame_ind, labels)
            clip_modalities_dict["depth_frames"] = self.load_depth_frames(recording_full_path, frame_ind)
            clip_modalities_dict["point_clouds"] = self.load_point_clouds_2(recording_full_path, frame_ind)
            clip_modalities_dict["eye_data_frames"] = self.load_data_frames_from_csv(recording_full_path, frame_ind,
                                                                                     filename="norm_proc_eye_data.csv")
            clip_modalities_dict["hand_data_frames"] = self.load_data_frames_from_csv(recording_full_path, frame_ind,
                                                                                      filename="norm_proc_hand_data.csv")

            return clip_modalities_dict, torch.from_numpy(labels), vid_idx, frame_pad

        for mod in self.modalities:
            if mod == "rgb_frames":
                clip_modalities_dict["rgb_frames"] = self.load_rgb_frames(recording_full_path, frame_ind, labels)
            elif mod == "point_clouds":
                clip_modalities_dict["point_clouds"] = self.load_point_clouds_2(recording_full_path, frame_ind)
            elif mod == "depth_frames":
                clip_modalities_dict["depth_frames"] = self.load_depth_frames(recording_full_path, frame_ind)
            elif mod == "eye_data_frames":
                clip_modalities_dict["eye_data_frames"] = self.load_data_frames_from_csv(
                    recording_full_path, frame_ind, filename="norm_proc_eye_data.csv", modality='eye_data_frames')
            elif mod == "hand_data_frames":
                clip_modalities_dict["hand_data_frames"] = self.load_data_frames_from_csv(
                    recording_full_path, frame_ind, filename="norm_proc_hand_data.csv", modality='hand_data_frames')
            else:
                raise NotImplementedError

        return clip_modalities_dict, torch.from_numpy(labels), vid_idx, frame_pad



if __name__ == "__main__":

    dataset_path = r'C:\TinyDataset'
    smallDataset = "SmallDataset" in dataset_path or 'Tiny' in dataset_path
    tinyDataset = 'Tiny' in dataset_path
    print("SmallDataset: ", smallDataset)
    train_transforms = videotransforms.RandomCrop((224, 224))
    test_transforms = videotransforms.CenterCrop((224, 224))
    dataset = HololensStreamRecClipDataset(dataset_path=dataset_path,
                                           dataset='train', eye_crop_transform=True, smallDataset=smallDataset,
                                           rgb_reshape_factor=2, modalities=['rgb_frames', 'eye_data_frames'])

    clip_num = 5
    clip_data_dict, labels, vid_idx, frame_pad = dataset[clip_num]
    print(clip_data_dict, labels)
