# ikea_ego_dataset_toolbox_public

tools to process the raw dataset:
the raw_data_processing directory can be used to process the raw dataset.
run the raw_data_processing/main.py with the directory containing all recordings to be processed as an input arg.
the scripts will recursively search the input directory argument and look for recording directories which will be identified by the suffix: recDir

the directories found will be processed, this can be spliut into three major steps:
1. data processing: create point clouds by projecting rgb on depth, projecting eye focus point and hand joint points on rgb image.
2. visualization: create an mp4 clip of the recording from the rgb images
3. normalization: create more informative and and eye csv files, synchronize eye, hand, rgb, and depth data. dump synchronized data into norm folder.

tools to load clips from the ikea_ego_dataset and train i3d model:

use the Dataloader.py to load clips from the ikea_ego_dataset.

Two main dataloaders are implemented:
1. IKEAEgoDatasetPickleClips - some systems process clips faster if large numbers of clips are agregated into pickles. this dataloader enables you to load pickles pf clips instead of loadin individual batches.
2. HololensStreamRecClipDataset - the main dataloader for the class. loads a clip from the dataset. some data augmentations are available and are implemented in the i3d/i3d_ego/i3d_utils.py file. in addition,rgb label watermarking is supported.
  

* HololensStreamRecBase is a base class for inheritance of future dataloaders

if you have downloaded the full version of the dataset you can use the createSmallDataset to resize rgb files, apply random sampling/FPF_sampling to the point clouds for compression purposes.

i3d train/test and eval pipelines are available in the i3d/ego_i3d directory.