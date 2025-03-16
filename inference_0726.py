import argparse
import imp
import yaml
import shutil
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from glob import glob
from scipy.ndimage.filters import gaussian_filter
from typing import Tuple
from tqdm import tqdm
from nn_transunet.default_configuration import get_default_configuration
from nn_transunet.configuration import default_plans_identifier
from nn_transunet.networks.nnunet_model import Generic_UNet
from nn_transunet.utils.dist_utils import download_from_hdfs
from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from nn_transunet.networks.neural_network import no_op
from torch.cuda.amp import autocast
from typing import Union, Tuple, List

from nn_transunet.networks.transunet3d_model import InitWeights_He


def get_flops(model, test_data):
    from flop_count.flop_count import flop_count
    batch_size = test_data.shape[0]
    flop_dict, _ = flop_count(model, (test_data, ))
    msg = 'model_flops' + '\t' + str(
        sum(flop_dict.values()) / batch_size) + 'G' + '\t params:' + str(
            sum([m.numel()
                 for m in model.parameters()])) + '\n-----------------'
    return msg


parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    default='',
                    type=str,
                    metavar='FILE',
                    help='YAML config file specifying default arguments')
parser.add_argument("--fold", default=0, help='0, 1, ..., 5 or \'all\'')
parser.add_argument("--raw_data_dir", default='')
parser.add_argument("--raw_data_folder",
                    default='imagesTr',
                    help='can be imagesVal')
parser.add_argument("--save_folder", default=None)
parser.add_argument("--save_npz", default=False, action="store_true")
parser.add_argument("--disable_split",
                    default=False,
                    action="store_true",
                    help='just use raw_data_dir, do not use split!')
parser.add_argument("--model_latest",
                    default=False,
                    action="store_true",
                    help='')
parser.add_argument("--model_final",
                    default=False,
                    action="store_true",
                    help='')
parser.add_argument("--mixed_precision", default=True, type=bool, help='')
parser.add_argument("--measure_param_flops",
                    default=False,
                    action="store_true",
                    help='')

##################### the args from train script #########################################
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--loss_name', default='', type=str)
parser.add_argument('--plan_update', default='', type=str)
parser.add_argument('--crop_size',
                    nargs='+',
                    type=int,
                    default=None,
                    help='input to network')
parser.add_argument("--pretrained",
                    default=False,
                    action="store_true",
                    help="")
parser.add_argument("--disable_decoder",
                    default=False,
                    action="store_true",
                    help="disable decoder of mae network")
parser.add_argument("--model_params", default={})
parser.add_argument('--layer_decay',
                    default=1.0,
                    type=float,
                    help="layer-wise dacay for lr")
parser.add_argument(
    '--drop_path',
    type=float,
    default=0.0,
    metavar='PCT',
    help='Drop path rate (default: 0.1), drop_path=0 for MAE pretrain')
parser.add_argument("--find_zero_weight_decay",
                    default=False,
                    action="store_true",
                    help="")
parser.add_argument('--n_class', default=17, type=int, help="")
parser.add_argument('--deep_supervision_scales',
                    nargs='+',
                    type=int,
                    default=[],
                    help='')
parser.add_argument("--fix_ds_net_numpool",
                    default=False,
                    action="store_true",
                    help="")
parser.add_argument("--num_ds", default=None, type=int, help="")
parser.add_argument("--is_sigmoid",
                    default=False,
                    action="store_true",
                    help="")
parser.add_argument("--num_examples", type=int, help="")

args, remaining = parser.parse_known_args()
model_params = {}
if args.config:
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    if "model_params" in cfg.keys():
        model_params = cfg["model_params"]
else:
    raise NotImplementedError

args = parser.parse_args()

network, task, network_trainer, hdfs_base = cfg['network'], cfg['task'], cfg[
    'network_trainer'], cfg['hdfs_base']

plans_identifier = default_plans_identifier
plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
        trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier, hdfs_base=hdfs_base, plan_update='' if 'plan_update' not in cfg.keys() else cfg['plan_update'])
os.makedirs(output_folder_name, exist_ok=True)

fold_name = args.fold if isinstance(args.fold, str) and args.fold.startswith(
    'all') else 'fold_' + str(args.fold)

output_folder = output_folder_name + '/' + fold_name
plans_path = os.path.join(output_folder_name, 'plans.pkl')
shutil.copy(plans_file, plans_path)

val_keys = None
if not args.disable_split:
    splits_file = os.path.join(dataset_directory, "splits_final.pkl")
    splits = load_pickle(splits_file)
    if not args.fold.startswith('all'):
        assert int(args.fold) < len(splits)
        val_keys = splits[int(args.fold)]['val']
        if isinstance(val_keys, np.ndarray):
            val_keys = val_keys.tolist()

print("output folder for snapshot loading exists: ", output_folder)
prefix = "version5"
planfile = plans_path
if os.path.exists(output_folder + '/' + 'model_best.model'
                  ) and not args.model_latest and not args.model_final:
    print("load model_best.model")
    modelfile = output_folder + '/' + 'model_best.model'
elif os.path.exists(output_folder + '/' +
                    'model_final_checkpoint.model') and not args.model_latest:
    print("load model_final_checkpoint.model")
    modelfile = output_folder + '/' + 'model_final_checkpoint.model'
else:
    print("load model_latest.model")
    modelfile = output_folder + '/' + 'model_latest.model'

info = pickle.load(open(planfile, "rb"))
plan_data = {}
plan_data["plans"] = info

resolution_index = 1
if cfg['task'].find('500') != -1:  # multiphase task e.g, Brats
    resolution_index = 0

num_classes = plan_data['plans']['num_classes']

num_classes += 1  # add background

base_num_features = plan_data['plans']['base_num_features']
resolution_index = 0

patch_size = plan_data['plans']['plans_per_stage'][resolution_index][
    'patch_size']
patch_size = args.crop_size if args.crop_size is not None else patch_size

num_input_channels = plan_data['plans']['num_modalities']
conv_per_stage = plan_data['plans'][
    'conv_per_stage'] if "conv_per_stage" in plan_data['plans'].keys() else 2
use_mask_for_norm = plan_data['plans']['use_mask_for_norm']
normalization_schemes = plan_data['plans']['normalization_schemes']
intensity_properties = plan_data['plans']['dataset_properties'][
    'intensityproperties']
transpose_forward, transpose_backward = plan_data['plans'][
    'transpose_forward'], plan_data['plans']['transpose_backward']

pool_op_kernel_sizes = plan_data['plans']['plans_per_stage'][resolution_index][
    'pool_op_kernel_sizes']
conv_kernel_sizes = plan_data['plans']['plans_per_stage'][resolution_index][
    'conv_kernel_sizes']
current_spacing = plan_data['plans']['plans_per_stage'][resolution_index][
    'current_spacing']

try:
    mean = plan_data['plans']['dataset_properties']['intensityproperties'][0][
        'mean']
    std = plan_data['plans']['dataset_properties']['intensityproperties'][0][
        'sd']
    clip_min = plan_data['plans']['dataset_properties']['intensityproperties'][
        0]['percentile_00_5']
    clip_max = plan_data['plans']['dataset_properties']['intensityproperties'][
        0]['percentile_99_5']
except:
    # if cfg['task'].find('500') != -1 or '005' in task or '001' in task:
    #     mean, std, clip_min, clip_max = 0, 1, -9999, 9999
    # else:
    mean, std, clip_min, clip_max = None, None, -9999, 9999

args.disable_ds = True
if cfg['model'].startswith('Generic'):
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

    if cfg['model'] == 'Generic_UNet_with_PWAM':
        from nn_transunet.networks.nnunet_model import Generic_UNet_with_PWAM
        net = Generic_UNet_with_PWAM(
            num_input_channels, base_num_features, num_classes,
            len(pool_op_kernel_sizes), conv_per_stage, 2, nn.Conv3d,
            nn.InstanceNorm3d, norm_op_kwargs, nn.Dropout3d, dropout_op_kwargs,
            net_nonlin, net_nonlin_kwargs, False, False, lambda x: x,
            InitWeights_He(1e-2), pool_op_kernel_sizes, conv_kernel_sizes,
            False, True, True)
    elif cfg['model'] == 'Generic_UNet':
        net = Generic_UNet(num_input_channels, base_num_features, num_classes,
                           len(pool_op_kernel_sizes), conv_per_stage, 2,
                           nn.Conv3d, nn.InstanceNorm3d, norm_op_kwargs,
                           nn.Dropout3d, dropout_op_kwargs, net_nonlin,
                           net_nonlin_kwargs, False, False, lambda x: x,
                           InitWeights_He(1e-2), pool_op_kernel_sizes,
                           conv_kernel_sizes, False, True, True)
    elif cfg['model'] == 'Generic_SwinUNETR_with_PWAM':
        from nn_transunet.networks.swin_unetr import SwinUNETR as SwinUNETR_with_PWAM
        net = SwinUNETR_with_PWAM(
            img_size=(128, 128, 128),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        )
    elif cfg['model'] == 'Generic_UNet_with_PWAM_bot':
        from nn_transunet.networks.nnunet_model import Generic_UNet_with_PWAM_bot
        net = Generic_UNet_with_PWAM_bot(
            num_input_channels, base_num_features, num_classes,
            len(pool_op_kernel_sizes), conv_per_stage, 2, nn.Conv3d,
            nn.InstanceNorm3d, norm_op_kwargs, nn.Dropout3d, dropout_op_kwargs,
            net_nonlin, net_nonlin_kwargs, False, False, lambda x: x,
            InitWeights_He(1e-2), pool_op_kernel_sizes, conv_kernel_sizes,
            False, True, True)
    else:
        from monai.networks.nets import SwinUNETR
        net = SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        )

else:
    net = None
    print("Note implemented for cfg['model']")
    raise NotImplementedError
total = sum([param.nelement() for param in net.parameters()])
net.cuda()

if args.measure_param_flops:
    with torch.no_grad():
        test_data = torch.zeros(
            (1, 1, patch_size[0], patch_size[1], patch_size[2])).cuda()
        msg = get_flops(
            net, test_data
        )  # same flops results with detectron, but count params as well.
        print(args.config, msg)
    sys.exit(0)

checkpoint = torch.load(modelfile)

print("load epoch", checkpoint['epoch'])
new_state_dict = OrderedDict()
curr_state_dict_keys = list(net.state_dict().keys())
for k, value in checkpoint['state_dict'].items():
    key = k
    if key not in curr_state_dict_keys and key.startswith('module.'):
        key = key[7:]
    new_state_dict[key] = value

net.load_state_dict(new_state_dict, strict=False)
cur_dict = net.state_dict()
print("missing keys of pretrained",
      [k for k in new_state_dict.keys() if k not in cur_dict.keys()])
print("extra keys of pretrained",
      [k for k in cur_dict.keys() if k not in new_state_dict.keys()])
print("weights loaded to network")
net.eval()


def _get_arr(path):
    sitkimg = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(sitkimg)
    return arr, sitkimg


def _write_arr(arr, path, info=None):
    sitkimg = sitk.GetImageFromArray(arr)
    if info is not None:
        sitkimg.CopyInformation(info)
    sitk.WriteImage(sitkimg, path)


def get_do_separate_z(spacing, anisotropy_threshold=2):
    do_separate_z = spacing[-1] > anisotropy_threshold
    return do_separate_z


def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...],
                                      image_size: Tuple[int, ...],
                                      step_size: float) -> List[List[int]]:
    assert [i >= j for i, j in zip(image_size, patch_size)
            ], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [
        int(np.ceil((i - k) / j)) + 1
        for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)
    ]

    steps = []
    for dim in range(len(patch_size)):
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [
            int(np.round(actual_step_size * i)) for i in range(num_steps[dim])
        ]

        steps.append(steps_here)

    return steps


def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp,
                                              sigmas,
                                              0,
                                              mode='constant',
                                              cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(
        gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])
    return gaussian_importance_map


gaussian_mask = torch.from_numpy(
    _get_gaussian(patch_size)[np.newaxis,
                              np.newaxis]).cuda().half().clamp_min_(1e-4)


def predict(arr, l=None, l_mask=None):
    if args.num_ds is not None:
        prob_map = torch.zeros((
            args.num_ds,
            1,
            num_classes,
        ) + arr.shape[-3:]).half().cuda()
    else:
        prob_map = torch.zeros((
            1,
            num_classes,
        ) + arr.shape[-3:]).half().cuda()

    cnt_map = torch.zeros_like(prob_map)

    arr_clip = np.clip(arr, clip_min, clip_max)

    if mean is None and std is None:
        raw_norm = (arr_clip - arr_clip.mean()) / (arr_clip.std() + 1e-8)
    else:
        raw_norm = (arr_clip - mean) / std

    step_size = 0.5
    steps = _compute_steps_for_sliding_window(
        patch_size, raw_norm.shape[-3:], step_size)  # step_size=0.5 for Brats

    num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

    print("data shape:", raw_norm.shape)
    print("patch size:", patch_size)
    print("steps (x, y, and z):", steps)
    print("number of tiles:", num_tiles)

    for x in steps[0]:
        lb_x = x
        ub_x = x + patch_size[0]
        for y in steps[1]:
            lb_y = y
            ub_y = y + patch_size[1]
            for z in steps[2]:
                lb_z = z
                ub_z = z + patch_size[2]
                with torch.no_grad():
                    numpy_arr = raw_norm[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z][
                        np.newaxis] if len(raw_norm.shape) == 4 else raw_norm[
                            lb_x:ub_x, lb_y:ub_y, lb_z:ub_z][np.newaxis,
                                                             np.newaxis]
                    tensor_arr = torch.from_numpy(numpy_arr).cuda().half()
                    seg_pro = net(tensor_arr)  # (1, c, d, h, w)
                    if args.num_ds is not None and isinstance(
                            seg_pro, dict) and (
                                'is_max_hungarian' in model_params.keys()
                                and model_params['is_max_hungarian']):
                        aux_cls_out = [
                            p['pred_logits'] for p in seg_pro['aux_outputs']
                        ]
                        aux_mask_out = [
                            p['pred_masks'] for p in seg_pro['aux_outputs']
                        ]
                        all_cls_out, all_mask_out = [seg_pro["pred_logits"]
                                                     ] + aux_cls_out[::-1], [
                                                         seg_pro["pred_masks"]
                                                     ] + aux_mask_out[::-1]
                        for i, (mask_cls, mask_pred) in enumerate(
                                zip(all_cls_out,
                                    all_mask_out)):  # desceding order
                            mask_cls = F.softmax(mask_cls, dim=-1)[
                                ..., :-1]  # filter out non-object class
                            mask_pred = mask_pred.sigmoid()
                            _seg_pro = torch.einsum("bqc,bqdhw->bcdhw",
                                                    mask_cls, mask_pred)
                            _pred = _seg_pro * gaussian_mask
                            prob_map[i, :, :, lb_x:ub_x, lb_y:ub_y,
                                     lb_z:ub_z] += _pred

                    elif isinstance(seg_pro, dict) and (
                            'is_max_hungarian' in model_params.keys()
                            and model_params['is_max_hungarian']):
                        mask_cls, mask_pred = seg_pro["pred_logits"], seg_pro[
                            "pred_masks"]
                        mask_cls = F.softmax(
                            mask_cls,
                            dim=-1)[..., :-1]  # filter out non-object class
                        mask_pred = mask_pred.sigmoid()
                        seg_pro = torch.einsum("bqc,bqdhw->bcdhw", mask_cls,
                                               mask_pred)
                        _pred = seg_pro

                        # if args.config.find('500Region') != -1 or  task.find('005') != -1  or task.find('001')  != -1:
                        #     _pred = seg_pro
                        # else:
                        _pred = seg_pro * gaussian_mask

                        prob_map[:, :, lb_x:ub_x, lb_y:ub_y,
                                 lb_z:ub_z] += _pred
                        # NOTE: should also smooth cnt_map if apply gaussian_mask before |  neural_network.py -> network.predict_3D -> _internal_predict_3D_3Dconv_tiled
                        cnt_map[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += 1

                    elif args.num_ds is not None and not isinstance(
                            seg_pro, dict
                    ):  # (isinstance(seg_pro, list) or isinstance(seg_pro, tuple))
                        assert len(seg_pro) == args.num_ds, (len(seg_pro),
                                                             args.num_ds)
                        for i, _seg_pro in enumerate(seg_pro):
                            if torch.sum(_seg_pro[0, :, 0, 0, 0]) != 1:
                                _seg_pro = torch.softmax(_seg_pro, dim=1)
                            _pred = _seg_pro * gaussian_mask
                            prob_map[i, :, :, lb_x:ub_x, lb_y:ub_y,
                                     lb_z:ub_z] += _pred

                    elif isinstance(seg_pro, list) or isinstance(
                            seg_pro, tuple):
                        seg_pro = seg_pro[0]
                        if torch.sum(seg_pro[0, :, 0, 0, 0]) != 1:
                            seg_pro = torch.softmax(seg_pro, dim=1)
                        _pred = seg_pro * gaussian_mask
                        prob_map[:, :, lb_x:ub_x, lb_y:ub_y,
                                 lb_z:ub_z] += _pred
                    else:

                        if args.is_sigmoid:
                            _pred = seg_pro.sigmoid()
                        elif torch.sum(seg_pro[0, :, 0, 0, 0]) != 1:
                            seg_pro = torch.softmax(seg_pro, dim=1)
                            _pred = seg_pro * gaussian_mask

                        prob_map[:, :, lb_x:ub_x, lb_y:ub_y,
                                 lb_z:ub_z] += _pred
                        cnt_map[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += 1

    print("before devision", prob_map.max(), prob_map.min(), cnt_map.min())
    # if args.config.find('500Region') != -1 or  task.find('005') != -1  or task.find('001')  != -1:
    #     prob_map /= cnt_map
    print("after devision", prob_map.max(), prob_map.min())
    # viz_data(torch.from_numpy(raw_norm[np.newaxis, np.newaxis]).cuda().half(), prob_map)
    torch.cuda.empty_cache()
    return prob_map.detach().cpu()


def itk_change_spacing(src_itk, output_spacing, interpolate_method='Linear'):
    assert interpolate_method in ['Linear', 'NearestNeighbor']
    src_size = src_itk.GetSize()
    src_spacing = src_itk.GetSpacing()

    re_sample_scale = tuple(
        np.array(src_spacing) / np.array(output_spacing).astype(float))
    re_sample_size = tuple(
        np.array(src_size).astype(float) * np.array(re_sample_scale))

    re_sample_size = [int(round(x)) for x in re_sample_size]
    output_spacing = tuple((np.array(src_size) / np.array(re_sample_size)) *
                           np.array(src_spacing))

    re_sampler = sitk.ResampleImageFilter()
    re_sampler.SetOutputPixelType(src_itk.GetPixelID())
    re_sampler.SetReferenceImage(src_itk)
    re_sampler.SetSize(re_sample_size)
    re_sampler.SetOutputSpacing(output_spacing)
    re_sampler.SetInterpolator(eval('sitk.sitk' + interpolate_method))
    return re_sampler.Execute(src_itk)


def resample_image_to_ref(image,
                          ref,
                          interp=sitk.sitkNearestNeighbor,
                          pad_value=0):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref)
    resample.SetDefaultPixelValue(pad_value)
    resample.SetInterpolator(interp)
    return resample.Execute(image)


def Inference3D(rawf, save_path=None, l=None, l_mask=None):
    arr_raw, sitk_raw = _get_arr(rawf)
    origin_spacing = sitk_raw.GetSpacing()
    rai_size = sitk_raw.GetSize()
    print("origin_spacing: ", origin_spacing)
    print("raw_size", rai_size)
    if (get_do_separate_z(origin_spacing)
            or get_do_separate_z(current_spacing[::-1])):
        print('preprocessing: do seperate z.....')
        img_arr = []
        for i in range(rai_size[-1]):
            img_arr.append(
                sitk.GetArrayFromImage(
                    itk_change_spacing(sitk_raw[:, :, i],
                                       current_spacing[::-1][:-1])))
        img_arr = np.array(img_arr)
        img_sitk = sitk.GetImageFromArray(img_arr)
        img_sitk.SetOrigin(sitk_raw.GetOrigin())
        img_sitk.SetDirection(sitk_raw.GetDirection())
        img_sitk.SetSpacing(
            tuple(current_spacing[::-1][:-1]) + (origin_spacing[-1], ))
        img_arr = sitk.GetArrayFromImage(
            itk_change_spacing(img_sitk,
                               current_spacing[::-1],
                               interpolate_method='NearestNeighbor'))
    else:
        img_arr = sitk.GetArrayFromImage(
            itk_change_spacing(sitk_raw, current_spacing[::-1]))
    pad_flag = 0
    padzyx = np.clip(np.array(patch_size) - np.array(img_arr.shape), 0,
                     1000)  # clip the shape..
    if np.any(padzyx > 0):
        pad_flag = 1
        pad_left = padzyx // 2
        pad_right = padzyx - padzyx // 2
        img_arr = np.pad(img_arr, ((pad_left[0], pad_right[0]),
                                   (pad_left[1], pad_right[1]),
                                   (pad_left[2], pad_right[2])))

    # PREDICT!
    if args.mixed_precision:
        context = autocast
    else:
        context = no_op

    with context():
        with torch.no_grad():
            prob_map = predict(img_arr, l, l_mask)

    if args.num_ds is not None:
        img_arr_shape = img_arr.shape
        del img_arr
        for idx in range(args.num_ds):
            _prob_map = prob_map[idx]
            if pad_flag:
                _prob_map = _prob_map[:, :, pad_left[0]:img_arr_shape[0] -
                                      pad_right[0],
                                      pad_left[1]:img_arr_shape[1] -
                                      pad_right[1],
                                      pad_left[2]:img_arr_shape[2] -
                                      pad_right[2]]

            if (get_do_separate_z(origin_spacing)
                    or get_do_separate_z(current_spacing[::-1])):
                print('postpreprocessing: do seperate z......')
                prob_map_interp_xy = torch.zeros(list(_prob_map.size()[:2]) + [
                    _prob_map.size()[2],
                ] + list(sitk_raw.GetSize()[::-1][1:]),
                                                 dtype=torch.half)

                for i in range(_prob_map.size(2)):
                    prob_map_interp_xy[:, :, i] = F.interpolate(
                        _prob_map[:, :, i].cuda().float(),
                        size=sitk_raw.GetSize()[::-1][1:],
                        mode="bilinear").detach().half().cpu()
                del _prob_map

                prob_map_interp = np.zeros(
                    list(prob_map_interp_xy.size()[:2]) +
                    list(sitk_raw.GetSize()[::-1]),
                    dtype=np.float16)

                for i in range(prob_map_interp.shape[1]):
                    prob_map_interp[:, i] = F.interpolate(
                        prob_map_interp_xy[:, i:i + 1].cuda().float(),
                        size=sitk_raw.GetSize()[::-1],
                        mode="nearest").detach().half().cpu().numpy()
                del prob_map_interp_xy

            else:
                prob_map_interp = np.zeros(list(_prob_map.size()[:2]) +
                                           list(sitk_raw.GetSize()[::-1]),
                                           dtype=np.float16)
                for i in range(_prob_map.size(1)):
                    prob_map_interp[:, i] = F.interpolate(
                        _prob_map[:, i:i + 1].cuda().float(),
                        size=sitk_raw.GetSize()[::-1],
                        mode="trilinear").detach().half().cpu().numpy()
                del _prob_map

            if save_path is None:
                save_dir = rawf.replace(".nii.gz", "_pred.nii.gz")
                assert 1 == 2
            else:
                uid = rawf.split("/")[-1].replace(".nii.gz", "_pred.nii.gz")
                _save_path = os.path.join(save_path, 'ds_' + str(idx))
                os.makedirs(_save_path, exist_ok=True)
                save_dir = os.path.join(_save_path, uid)
            if args.save_npz:  # added
                save_npz_dir = save_path.replace("nnUNet_inference",
                                                 "nnUNet_inference_npz")
                save_npz_dir = os.path.join(save_npz_dir, 'ds_' + str(idx))
                os.makedirs(save_npz_dir, exist_ok=True)
                npz_name = rawf.split("/")[-1].replace(".nii.gz", ".npz")
                np.savez_compressed(os.path.join(save_npz_dir, npz_name),
                                    softmax=prob_map_interp.squeeze(0))

            segmentation = np.argmax(prob_map_interp.squeeze(0), axis=0)
            del prob_map_interp
            pred_sitk = sitk.GetImageFromArray(segmentation.astype(np.uint8))
            pred_sitk.CopyInformation(sitk_raw)
            pred_sitk = resample_image_to_ref(pred_sitk, sitk_raw)
            sitk.WriteImage(pred_sitk, save_dir)
        return

    if pad_flag:
        prob_map = prob_map[:, :, pad_left[0]:img_arr.shape[0] - pad_right[0],
                            pad_left[1]:img_arr.shape[1] - pad_right[1],
                            pad_left[2]:img_arr.shape[2] - pad_right[2]]
    del img_arr

    if (get_do_separate_z(origin_spacing)
            or get_do_separate_z(current_spacing[::-1])):
        print('postpreprocessing: do seperate z......')
        prob_map_interp_xy = torch.zeros(list(prob_map.size()[:2]) + [
            prob_map.size()[2],
        ] + list(sitk_raw.GetSize()[::-1][1:]),
                                         dtype=torch.half)

        for i in range(prob_map.size(2)):
            prob_map_interp_xy[:, :, i] = F.interpolate(
                prob_map[:, :, i].cuda().float(),
                size=sitk_raw.GetSize()[::-1][1:],
                mode="bilinear").detach().half().cpu()
        del prob_map

        prob_map_interp = np.zeros(list(prob_map_interp_xy.size()[:2]) +
                                   list(sitk_raw.GetSize()[::-1]),
                                   dtype=np.float16)

        for i in range(prob_map_interp.shape[1]):
            prob_map_interp[:, i] = F.interpolate(
                prob_map_interp_xy[:, i:i + 1].cuda().float(),
                size=sitk_raw.GetSize()[::-1],
                mode="nearest").detach().half().cpu().numpy()
        del prob_map_interp_xy

    else:
        prob_map_interp = np.zeros(list(prob_map.size()[:2]) +
                                   list(sitk_raw.GetSize()[::-1]),
                                   dtype=np.float16)
        for i in range(prob_map.size(1)):
            prob_map_interp[:, i] = F.interpolate(
                prob_map[:, i:i + 1].cuda().float(),
                size=sitk_raw.GetSize()[::-1],
                mode="trilinear").detach().half().cpu().numpy()
        del prob_map

    if save_path is None:
        raise ValueError("save_path is None")
        save_dir = rawf.replace(".nii.gz", "_pred.nii.gz")
    else:
        #change name to msd format
        uid = rawf.split("/")[-1].replace('_0000', '')
        save_dir = os.path.join(save_path, uid)

    if args.save_npz:
        save_npz_dir = save_path.replace("nnUNet_inference",
                                         "nnUNet_inference_npz")
        os.makedirs(save_npz_dir, exist_ok=True)
        npz_name = uid.replace(".nii.gz", ".npz")
        np.savez_compressed(os.path.join(save_npz_dir, npz_name),
                            softmax=prob_map_interp.squeeze(0))
    segmentation = np.argmax(prob_map_interp.squeeze(0), axis=0)

    del prob_map_interp
    pred_sitk = sitk.GetImageFromArray(segmentation.astype(np.uint8))
    pred_sitk.CopyInformation(sitk_raw)
    pred_sitk = resample_image_to_ref(pred_sitk, sitk_raw)

    sitk.WriteImage(pred_sitk, save_dir)


if not args.raw_data_dir:
    raw_data_dir = os.getenv(
        'nnUNet_raw_data_base'
    ) + "/nnUNet_raw_data/" + task + '/' + args.raw_data_folder
else:
    raw_data_dir = args.raw_data_dir

if not args.save_folder:
    task_name = task.replace('Task0', 'Task')
    args.save_folder = os.getenv(
        'nnUNet_raw_data_base'
    ) + "/nnUNet_inference/" + fold_name + '/' + task_name + '/' + os.path.basename(
        args.raw_data_dir)
print(args.save_folder)
os.makedirs(args.save_folder, exist_ok=True)

rawf = sorted(glob(raw_data_dir + "/*.nii.gz"))
# print('==========================================================',raw_data_dir)
if val_keys is not None:
    valid_rawf = [
        i for i in rawf if os.path.basename(i).replace('.nii.gz', '').replace(
            '_0000', '') in val_keys
    ]
else:
    valid_rawf = rawf

import json

json_file = "/public/home/lixueping/backups/nnUNet-1.7.1/DATASET/nnUNet_preprocessed/Task002_ATLAS/meta_data_atlas.json"
meta_data = json.load(open(json_file, 'r'))
sentences = []
if val_keys is not None:
    for k in val_keys:
        for meta in meta_data:
            name = meta['image']
            sentence = meta['sentence']
            if name == k:
                sentences.append(sentence)
else:
    for k in valid_rawf:
        for meta in meta_data:
            name = meta['image']
            sentence = meta['sentence']
            if name == k[-25:-12]:
                sentences.append(sentence)
                break

# valid_rawf是val set的列表，表位置
from nn_transunet.bert.tokenization_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_tokens = 50
from nn_transunet.bert.modeling_bert import BertModel

bert_model = BertModel.from_pretrained(
    '/public/home/lixueping/backups/3D-TransUNet/nn_transunet/bert/bert-base-uncased'
)
bert_model.cuda()
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch

# print('==================================================',len(valid_rawf))

for i, raw in enumerate(tqdm(zip(valid_rawf, sentences))):

    # print(raw)
    input_idss = [0] * max_tokens
    attn_mask = [0] * max_tokens

    input_ids = tokenizer.encode(text=raw[1], add_special_tokens=True)
    input_ids = input_ids[:max_tokens]

    input_idss[:len(input_ids)] = input_ids
    attn_mask[:len(input_ids)] = [1] * len(input_ids)

    input_idss = np.array(input_idss)
    attn_mask = np.array(attn_mask)

    input_idss = maybe_to_torch(input_idss).unsqueeze(dim=0)
    attn_mask = maybe_to_torch(attn_mask).unsqueeze(dim=0)

    input_idss = input_idss.int()
    attn_mask = attn_mask.int()
    input_idss = to_cuda(input_idss)
    attn_mask = to_cuda(attn_mask)

    last_hidden_states = bert_model(input_idss, attention_mask=attn_mask)[0]
    embedding = last_hidden_states.permute(0, 2, 1)
    attentions = attn_mask.unsqueeze(-1)

    Inference3D(raw[0], args.save_folder, l=embedding, l_mask=attentions)
    if args.num_examples is not None and i >= args.num_examples:
        break

print('inference done!')
