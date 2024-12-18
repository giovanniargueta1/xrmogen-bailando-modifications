import argparse
import mmcv
import numpy as np
import os
from scipy import linalg
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
from utils.dance_features.kinetic import extract_kinetic_features
from utils.dance_features.manual import extract_manual_features


def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)

    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)


def filter_files(root_dir, file_exts=['.pkl', '.npy']):
    """Returns a list of files in the given root directory matching the given extensions."""
    return [file for file in os.listdir(root_dir) if any(file.endswith(ext) for ext in file_exts)]


def calc_motion_quality(predicted_pkl_root, gt_pkl_root):

    pred_features_k = []
    pred_features_m = []
    gt_freatures_k = []
    gt_freatures_m = []

    # Filter .pkl files in the predicted directory
    pred_pkl_files_k = filter_files(os.path.join(predicted_pkl_root, 'kinetic_features'), file_exts=['.pkl'])
    pred_pkl_files_m = filter_files(os.path.join(predicted_pkl_root, 'manual_features_new'), file_exts=['.pkl'])

    # Filter .npy files in the GT directory
    gt_npy_files_k = filter_files(os.path.join(gt_pkl_root, 'kinetic_features'), file_exts=['.npy'])
    gt_npy_files_m = filter_files(os.path.join(gt_pkl_root, 'manual_features_new'), file_exts=['.npy'])

    # Check if the lists of files are empty and raise an error if they are
    if not pred_pkl_files_k or not pred_pkl_files_m:
        raise ValueError(f"No .pkl files found in predicted kinetic or manual features directory: {predicted_pkl_root}")
    if not gt_npy_files_k or not gt_npy_files_m:
        raise ValueError(f"No .npy files found in GT kinetic or manual features directory: {gt_pkl_root}")

    # Load features from filtered .pkl files in the predicted directory
    pred_features_k = [
        mmcv.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl))
        for pkl in pred_pkl_files_k
    ]
    pred_features_m = [
        mmcv.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl))
        for pkl in pred_pkl_files_m
    ]

    # Load features from filtered .npy files in the GT directory
    gt_freatures_k = [
        np.load(os.path.join(gt_pkl_root, 'kinetic_features', npy))
        for npy in gt_npy_files_k
    ]
    gt_freatures_m = [
        np.load(os.path.join(gt_pkl_root, 'manual_features_new', npy))
        for npy in gt_npy_files_m
    ]

    # If the lists are empty after loading, raise an error
    if not gt_freatures_k:
        raise ValueError(f"Failed to load any GT kinetic features from: {os.path.join(gt_pkl_root, 'kinetic_features')}")
    if not gt_freatures_m:
        raise ValueError(f"Failed to load any GT manual features from: {os.path.join(gt_pkl_root, 'manual_features_new')}")

    # Convert lists to numpy arrays
    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    pred_features_m = np.stack(pred_features_m)  # Nx32
    gt_freatures_k = np.stack(gt_freatures_k)  # N' x 72 N' >> N
    gt_freatures_m = np.stack(gt_freatures_m)  #

    gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
    gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m)

    fid_k = calc_fid(pred_features_k, gt_freatures_k)
    fid_m = calc_fid(pred_features_m, gt_freatures_m)

    div_k = calculate_avg_distance(pred_features_k)
    div_m = calculate_avg_distance(pred_features_m)

    metrics = {
        'FIDk': fid_k.real,
        'FIDg': fid_m.real,
        'DIVk': div_k,
        'DIVg': div_m
    }
    return metrics


def calc_fid(kps_gen, kps_gt):
    """compute FID between features of generated dance and GT."""

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1, mu2, sigma1, sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n - 1)


def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist


def calc_and_save_feats(root, start=0, end=1200):
    """
        compute and save motion features
        Args:
            root: folder of pkl files
            start: start frame
            end: ending frame (default 20 seconds for 60fps)
    """
    if not os.path.exists(os.path.join(root, 'kinetic_features')):
        os.mkdir(os.path.join(root, 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'manual_features_new')):
        os.mkdir(os.path.join(root, 'manual_features_new'))

    # Filter only .pkl files in the root directory
    for pkl in filter_files(root, file_exts=['.pkl']):
        if (os.path.exists(os.path.join(root, 'kinetic_features', pkl)) and
                os.path.exists(os.path.join(root, 'manual_features_new', pkl))) or os.path.isdir(
                                                os.path.join(root, pkl)):
            continue
        joint3d = mmcv.load(os.path.join(root, pkl)).reshape(-1,
                                                             72)[start:end, :]

        roott = joint3d[:1, :3]  # the root Tx72 (Tx(24x3))
        joint3d = joint3d - np.tile(
            roott, (1, 24))  # Calculate relative offset with respect to root

        mmcv.dump(
            extract_kinetic_features(joint3d.reshape(-1, 24, 3)),
            os.path.join(root, 'kinetic_features', pkl))
        mmcv.dump(
            extract_manual_features(joint3d.reshape(-1, 24, 3)),
            os.path.join(root, 'manual_features_new', pkl))


def get_music_beat(music_feature_root, key, length=None):
    """
        Fetch music beats from preprocessed music features,
        represented as bool (True=beats)
        Args:
            music_feature_root: the root folder of
                preprocessed music features
            key: dance name
            length: restriction on sample length
    """
    path = os.path.join(music_feature_root, key)
    sample_dict = mmcv.load(path)
    if length is not None:
        beats = np.array(sample_dict['music_array'])[:, 53][:][:length]
    else:
        beats = np.array(sample_dict['music_array'])[:, 53]
    beats = beats.astype(bool)
    beat_axis = np.arange(len(beats))
    beat_axis = beat_axis[beats]

    return beat_axis


def calc_dance_beat(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(
        np.sqrt(np.sum((keypoints[1:] - keypoints[:-1])**2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def beat_align_score(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba += np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))


def calc_beat_align_score(pkl_root, music_feature_root):
    ba_scores = {}

    for pkl in os.listdir(pkl_root):
        if os.path.isdir(os.path.join(pkl_root, pkl)):
            continue
        joint3d = mmcv.load(os.path.join(pkl_root, pkl))

        dance_beats, length = calc_dance_beat(joint3d)
        music_beats = get_music_beat(music_feature_root,
                                     pkl.split('.')[0] + '.json', length)

        score = beat_align_score(music_beats, dance_beats)
        ba_scores[pkl] = score  # Store the score for each video file

        print(f"BA Score for {pkl}: {score}")

    return ba_scores  # Return the individual scores instead of the mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visulize from recorded pkl')
    parser.add_argument('--pkl_root', type=str)
    parser.add_argument('--gt_root', type=str)
    parser.add_argument('--music_feature_root', type=str)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1200)
    args = parser.parse_args()

    # # FIDk, FIDg, DIVk, DIVg
    print('Calculating and saving features')
    calc_and_save_feats(args.pkl_root, args.start, args.end)
    calc_and_save_feats(args.gt_root, args.start, args.end)
    metrics = calc_motion_quality(args.pkl_root, args.gt_root)

    # music-beat align score
    print('Calculating Music-dance beat alignment score')
    ba_scores = calc_beat_align_score(args.pkl_root, args.music_feature_root)

    # Update metrics dictionary with BeatAlignScore as the dictionary of individual scores
    metrics.update(dict(BeatAlignScores=ba_scores))

    print('Quantitative scores:', metrics)
    print(metrics)

    # Save the individual scores in a JSON file
    mmcv.dump(metrics, args.pkl_root + '_scores.json')
