import torchvision
from numpy import random
from scipy.stats import bernoulli

import torch
import numpy as np




def probabilisticShuffleClipFrames(frames, shuffle_prob):

    if bernoulli.rvs(shuffle_prob):
        return frames[torch.randperm(frames.size()[0])]
    return frames


def offcenterCrop(frames, eye_focus_points, target_w=224, target_h=224, random_offset=False, offcenter_variance=10):
    assert len(frames) == len(eye_focus_points)
    num_frames = len(frames)
    channels, base_h, base_w = frames[0].shape
    assert base_h >= target_h and base_w >= target_w

    min_center_w = target_w/2
    max_center_w = base_w - target_w/2

    min_center_h = target_h/2
    max_center_h = base_h - target_h/2

    if random_offset:
        random_offset = torch.from_numpy(random.normal(loc=0.0, scale=offcenter_variance, size=eye_focus_points.shape)) #scale=target_h/10
        target_focus_points = random_offset + eye_focus_points
    else:
        target_focus_points = eye_focus_points

    #apply limitiations:
    target_focus_points[:,0] = torch.minimum(torch.maximum(target_focus_points[:, 0], torch.ones(num_frames)*min_center_w), torch.ones(num_frames)*max_center_w).to(torch.int)
    target_focus_points[:,1] = torch.minimum(torch.maximum(target_focus_points[:, 1], torch.ones(num_frames)*min_center_h), torch.ones(num_frames)*max_center_h).to(torch.int)

    top_limit_vec = target_focus_points[:, 1] - target_h/2
    left_limit_vec = target_focus_points[:, 0] - target_w/2
    # frames = torch.permute(frames, (0, 2, 3, 1))
    cropped_frames = []
    # frame_num=0
    for frame, top_limit, left_limit in zip(frames, top_limit_vec.to(torch.int), left_limit_vec.to(torch.int)):
        cropped_frame = torchvision.transforms.functional.crop(frame, top=top_limit, left=left_limit, height=target_h, width=target_w)
        cropped_frames.append(cropped_frame)
        # torchvision.utils.save_image(cropped_frame.float()/256, r'D:\eye_centered_images\{}_frame.png'.format(frame_num))
        # frame_num+=1
    cropped_frames = torch.stack(tensors=cropped_frames)
    # cropped_frames = torch.permute(cropped_frames, (0, 3, 1, 2))

    return cropped_frames
def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""

    batch_size = target.size(0)
    pred = torch.argmax(output, dim=1)
    pred = pred.squeeze()
    correct = pred.eq(target.expand_as(pred))
    acc = correct.view(-1).float().sum(0) * 100 / (batch_size)
    return acc


def sliding_accuracy(logits, target, slider_length):
    '''
        compute the accuracy while averaging over slider_length frames
        implemented to accumulate at the begining of the sequence and give the average for the last frame in the slider
    '''

    n_examples = target.size(0)
    pred = torch.zeros_like(logits)
    for i in range(logits.size(2)):
        pred[:, :, i] = torch.mean(logits[:, :, np.max([0, i - slider_length]):i + 1], dim=2)

    pred = torch.argmax(pred, dim=1)
    pred = pred.squeeze().view(-1)
    correct = pred.eq(target)
    acc = correct.view(-1).float().sum(0) * 100 / n_examples
    return acc, pred


def accuracy_v2(output, target):
    """Computes the precision@k for the specified values of k"""

    batch_size = target.size(0)
    n_frames = target.size(1)
    correct = output.eq(target.expand_as(output))
    acc = correct.view(-1).float().sum(0) * 100 / (batch_size*n_frames)
    return acc


def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def post_process_logits(per_frame_logits, average=False, num_frames_to_avg=12, threshold = 0.7):
    if average:
        last_frame_logits = torch.mean(per_frame_logits[:, :, -num_frames_to_avg - 1:-1], dim=2)
        label_ind = torch.argmax(last_frame_logits, dim=1).item()
        last_frame_logits = torch.nn.functional.softmax(last_frame_logits, dim=1).squeeze()
    else:
        per_frame_logits = torch.nn.functional.softmax(per_frame_logits, dim=1)
        _, pred = per_frame_logits.topk(1, 1, True, True)
        label_ind = pred.squeeze()[-1].item()
        last_frame_logits = per_frame_logits[0, :, -1].squeeze()

    if last_frame_logits[label_ind] < threshold:
        label_ind = 0

    return label_ind, last_frame_logits


def make_weights_for_balanced_classes(clip_set, label_count):
    """ compute the weight per clip for the weighted random sampler"""
    n_clips = len(clip_set)
    nclasses = len(label_count)
    N = label_count.sum()
    weight_per_class = [0.] * nclasses

    for i in range(nclasses):
        if label_count[i]==0:
            print(f"got a label with 0 examples {i}: nume examples: {label_count[i]} ")
            weight_per_class[i] = 0

        else:
            weight_per_class[i] = N/float(label_count[i])

    weight = [0] * n_clips
    for idx, clip in enumerate(clip_set):
        clip_label_sum = clip[1].sum(axis=1)
        if clip_label_sum.sum() == 0:
            print("zero!!!")
        ratios = clip_label_sum / clip_label_sum.sum()
        weight[idx] = np.dot(weight_per_class, ratios)
    return weight