# ikea_ego_dataset_toolbox_public
tools to load clips from the ikea_ego_dataset and train i3d model 


use the Dataloader.py to load clips from the ikea_ego_dataset.

Two main dataloaders are implemented:
1. IKEAEgoDatasetPickleClips - some systems process clips faster if large numbers of clips are agregated into pickles. this dataloader enables you to load pickles pf clips instead of loadin individual batches.
2. HololensStreamRecClipDataset - the main dataloader for the class. loads a clip from the dataset. some data augmentations are available and are implemented in the i3d/i3d_ego/i3d_utils.py file. in addition,rgb label watermarking is supported.
  

* HololensStreamRecBase is a base class for inheritance of future dataloaders

if you have downloaded the full version of the dataset you can use the createSmallDataset to resize rgb files, apply random sampling/FPF_sampling to the point clouds for compression purposes.

i3d train/test and eval pipelines are available in the i3d/ego_i3d directory.