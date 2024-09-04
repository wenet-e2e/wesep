# Author: Rongzhi Gu, Yi Luo
# Copyright: Tencent AI Lab
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torchaudio.functional import highpass_biquad
from torchaudio.transforms import Resample

# set random seed
seed = 20231
np.random.seed(seed)
torch.manual_seed(seed)


def calc_cos(orientation_rad):
    """
    cos_theta: tensor, [azimuth, elevation] with shape [..., 2]
    return: [..., 3]
    """
    return torch.stack(
        [
            torch.cos(
                orientation_rad[..., 0] * torch.sin(orientation_rad[..., 1])
            ),
            torch.sin(
                orientation_rad[..., 0] * torch.sin(orientation_rad[..., 1])
            ),
            torch.cos(orientation_rad[..., 1]),
        ],
        -1,
    )


def freq_invariant_decay_func(cos_theta, pattern="cardioid"):
    """
    cos_theta: tensor
    Return:
    amplitude: tensor with same shape as cos_theta
    """

    if pattern == "cardioid":
        return 0.5 + 0.5 * cos_theta

    elif pattern == "omni":
        return torch.ones_like(cos_theta)

    elif pattern == "bidirectional":
        return cos_theta

    elif pattern == "hyper_cardioid":
        return 0.25 + 0.75 * cos_theta

    elif pattern == "sub_cardioid":
        return 0.75 + 0.25 * cos_theta

    elif pattern == "half_omni":
        c = torch.clamp(cos_theta, 0)
        c[c > 0] = 1.0
        return c
    else:
        raise NotImplementedError


def freq_invariant_src_decay_func(
    mic_pos, src_pos, src_orientation_rad, pattern="cardioid"
):
    """
    mic_pos: [n_mic, 3] (tensor)
    src_pos: [n_src, 3] (tensor)
    src_orientation_rad: [n_src, 2] (tensor), elevation, azimuth

    Return:
    amplitude: [n_mic, n_src, n_image]
    """
    # Steering vector of source(s)
    orV_src = calc_cos(src_orientation_rad).unsqueeze(0)  # [nsrc, 3]

    # receiver to src vector
    rcv_to_src_vec = mic_pos.unsqueeze(1) - src_pos.unsqueeze(
        0
    )  # [n_mic, n_src, 3]

    cos_theta = (rcv_to_src_vec * orV_src).sum(-1)  # [n_mic, n_src]
    cos_theta /= torch.sqrt(rcv_to_src_vec.pow(2).sum(-1))
    cos_theta /= torch.sqrt(orV_src.pow(2).sum(-1))

    return freq_invariant_decay_func(cos_theta, pattern)


def freq_invariant_mic_decay_func(
    mic_pos, img_pos, mic_orientation_rad, pattern="cardioid"
):
    """
    mic_pos: [n_mic, 3] (tensor)
    img_pos: [n_src, n_image, 3] (tensor)
    mic_orientation_rad: [n_mic, 2] (tensor), azimuth, elevation

    Return:
    amplitude: [n_mic, n_src, n_image]
    """
    # Steering vector of source(s)
    orV_src = calc_cos(mic_orientation_rad)  # [nmic, 3]
    orV_src = orV_src.view(-1, 1, 1, 3)  # [n_mic, 1, 1, 3]

    # image to receiver vector
    # [1, n_src, n_image, 3] - [n_mic, 1, 1, 3] => [n_mic, n_src, n_image, 3]
    img_to_rcv_vec = img_pos.unsqueeze(0) - mic_pos.unsqueeze(1).unsqueeze(1)

    cos_theta = (img_to_rcv_vec * orV_src).sum(-1)  # [n_mic, n_src, n_image]
    cos_theta /= torch.sqrt(img_to_rcv_vec.pow(2).sum(-1))
    cos_theta /= torch.sqrt(orV_src.pow(2).sum(-1))

    return freq_invariant_decay_func(cos_theta, pattern)


def FRAM_RIR(
    mic_pos,
    sr,
    T60,
    room_dim,
    src_pos,
    num_src=1,
    direct_range=(-6, 50),
    n_image=(1024, 4097),
    a=-2.0,
    b=2.0,
    tau=0.25,
    src_pattern="omni",
    src_orientation_rad=None,
    mic_pattern="omni",
    mic_orientation_rad=None,
):
    """Fast Random Appoximation of Multi-channel Room Impulse Response (FRAM-RIR)

    Args:
        mic_pos: The microphone(s) position with respect to the room coordinates,
                 with shape [num_mic, 3] (in meters). Room coordinate system must be defined in advance,
                 with the constraint that the origin of the coordinate is on the floor(so positive z axis points up).
        sr: RIR sampling rate (Hz).
        T60: RT60 (second).
        room_dim: Room size with shape [3] (meters).
        src_pos: The source(s) position with respect to the room coordinate system, with shape [num_src, 3] (meters).
        num_src: Number of sources. Defaults to 1.
        direct_range: 2-element tuple, range of early reflection time (milliseconds,
                                        defined as the context around the direct path signal) of RIRs.
                                        Defaults to (-6, 50).
        n_image: 2-element tuple, minimum and maximum number of images to sample from.
                                   Defaults to (1024, 4097).
        a: controlling the random perturbation added to each virtual sound source.  Defaults to -2.0.
        b: controlling the random perturbation added to each virtual sound source. Defaults to 2.0.
        tau: controlling the relationship between the distance and the number of reflections of each
                               virtual sound source. Defaults to 0.25.
        src_pattern: Polar pattern for all of the sources. Defaults to "omni".
        src_orientation_rad: Array-like with shape [num_src, 2]. Orientation (rad) of all
                                                the sources, where the first column indicate azimuth and the
                                                second column indicate elevation. Defaults to None.
        mic_pattern: Polar pattern for all of the receivers. Defaults to "omni".
        mic_orientation_rad: Array-like with shape [num_mic, 2]. Orientation (rad) of all
                                                the microphones, where the first column indicate azimuth and
                                                the second column indicate elevation. Defaults to None.

    Returns:
        rir: RIR filters for all mic-source pairs, with shape [num_mic, num_src, rir_length].
        early_rir: Early reflection (direct path) RIR filters for all mic-source pairs,
                   with shape [num_mic, num_src, rir_length].
    """

    # sample image
    image = np.random.choice(range(n_image[0], n_image[1]))

    R = torch.tensor(
        1.0 / (2 * (1.0 / room_dim[0] + 1.0 / room_dim[1] + 1.0 / room_dim[2]))
    )

    eps = np.finfo(np.float16).eps
    mic_position = torch.from_numpy(mic_pos)
    src_position = torch.from_numpy(src_pos)  # [nsource, 3]
    n_mic = mic_position.shape[0]
    num_src = src_position.shape[0]

    # [nmic, nsource]
    direct_dist = (
        (mic_position.unsqueeze(1) - src_position.unsqueeze(0)).pow(2).sum(-1)
        + 1e-3
    ).sqrt()
    # [nsource]
    nearest_dist, nearest_mic_idx = direct_dist.min(0)
    # [nsource, 3]
    nearest_mic_position = mic_position[nearest_mic_idx]

    ns = n_mic * num_src
    ratio = 64
    sample_sr = sr * ratio
    velocity = 340.0
    T60 = torch.tensor(T60)

    direct_idx = (
        torch.ceil(direct_dist * sample_sr / velocity)
        .long()
        .view(
            ns,
        )
    )
    rir_length = int(np.ceil(sample_sr * T60))

    resample1 = Resample(sample_sr, sample_sr // int(np.sqrt(ratio)))
    resample2 = Resample(sample_sr // int(np.sqrt(ratio)), sr)

    reflect_coef = (1 - (1 - torch.exp(-0.16 * R / T60)).pow(2)).sqrt()
    dist_range = [
        torch.linspace(1.0, velocity * T60 / nearest_dist[i] - 1, rir_length)
        for i in range(num_src)
    ]

    dist_prob = torch.linspace(0.0, 1.0, rir_length)
    dist_prob /= dist_prob.sum()
    dist_select_idx = dist_prob.multinomial(
        num_samples=int(image * num_src), replacement=True
    ).view(num_src, image)

    dist_nearest_ratio = torch.stack(
        [dist_range[i][dist_select_idx[i]] for i in range(num_src)], 0
    )

    # apply different dist ratios to mirophones
    azm = torch.FloatTensor(num_src, image).uniform_(-np.pi, np.pi)
    ele = torch.FloatTensor(num_src, image).uniform_(-np.pi / 2, np.pi / 2)
    # [nsource, nimage, 3]
    unit_3d = torch.stack(
        [
            torch.sin(ele) * torch.cos(azm),
            torch.sin(ele) * torch.sin(azm),
            torch.cos(ele),
        ],
        -1,
    )
    # [nsource] x [nsource, T] x [nsource, nimage, 3] => [nsource, nimage, 3]
    image2nearest_dist = nearest_dist.view(
        -1, 1, 1
    ) * dist_nearest_ratio.unsqueeze(-1)
    image_position = (
        nearest_mic_position.unsqueeze(1) + image2nearest_dist * unit_3d
    )
    # [nmic, nsource, nimage]
    dist = (
        (mic_position.view(-1, 1, 1, 3) - image_position.unsqueeze(0))
        .pow(2)
        .sum(-1)
        + 1e-3
    ).sqrt()

    # reflection perturbation
    reflect_max = (torch.log10(velocity * T60) - 3) / torch.log10(reflect_coef)
    reflect_ratio = (dist / (velocity * T60)) * (
        reflect_max.view(1, -1, 1) - 1
    ) + 1
    reflect_pertub = torch.FloatTensor(num_src, image).uniform_(
        a, b
    ) * dist_nearest_ratio.pow(tau)
    reflect_ratio = torch.maximum(
        reflect_ratio + reflect_pertub.unsqueeze(0), torch.ones(1)
    )

    # [nmic, nsource, 1 + nimage]
    dist = torch.cat([direct_dist.unsqueeze(2), dist], 2)
    reflect_ratio = torch.cat(
        [torch.zeros(n_mic, num_src, 1), reflect_ratio], 2
    )

    delta_idx = (
        torch.minimum(
            torch.ceil(dist * sample_sr / velocity),
            torch.ones(1) * rir_length - 1,
        )
        .long()
        .view(ns, -1)
    )
    delta_decay = reflect_coef.pow(reflect_ratio) / dist

    #################################
    # source orientation simulation #
    #################################
    if src_pattern != "omni":
        # randomly sample each image's relative orientation with respect to the original source
        # equivalent to a random decay corresponds to the source's orientation pattern decay
        img_orientation_rad = torch.FloatTensor(num_src, image, 2).uniform_(
            -np.pi, np.pi
        )
        img_cos_theta = torch.cos(img_orientation_rad[..., 0]) * torch.cos(
            img_orientation_rad[..., 1]
        )  # [nsource, nimage]
        img_orientation_decay = freq_invariant_decay_func(
            img_cos_theta, pattern=src_pattern
        )  # [nsource, nimage]

        # direct path orientation should use the provided parameter
        if src_orientation_rad is None:
            # assume random orientation if not given
            src_orientation_azi = torch.FloatTensor(num_src).uniform_(
                -np.pi, np.pi
            )
            src_orientation_ele = torch.FloatTensor(num_src).uniform_(
                -np.pi, np.pi
            )
            src_orientation_rad = torch.stack(
                [src_orientation_azi, src_orientation_ele], -1
            )
        else:
            src_orientation_rad = torch.from_numpy(
                src_orientation_rad
            )  # [nsource, 2]

        src_orientation_decay = freq_invariant_src_decay_func(
            mic_position,
            src_position,
            src_orientation_rad,
            pattern=src_pattern,
        )  # [nmic, nsource]
        # apply decay
        delta_decay[:, :, 0] *= src_orientation_decay
        delta_decay[:, :, 1:] *= img_orientation_decay.unsqueeze(0)

    if mic_pattern != "omni":
        # mic orientation simulation #
        # when not given, assume that all mics facing up (positive z axis)
        if mic_orientation_rad is None:
            mic_orientation_rad = torch.stack(
                [torch.zeros(n_mic), torch.zeros(n_mic)], -1
            )  # [nmic, 2]
        else:
            mic_orientation_rad = torch.from_numpy(mic_orientation_rad)
        all_src_img_pos = torch.cat(
            (src_position.unsqueeze(1), image_position), 1
        )  # [nsource, nimage+1, 3]
        mic_orientation_decay = freq_invariant_mic_decay_func(
            mic_position,
            all_src_img_pos,
            mic_orientation_rad,
            pattern=mic_pattern,
        )  # [nmic, nsource, nimage+1]
        # apply decay
        delta_decay *= mic_orientation_decay

    rir = torch.zeros(ns, rir_length)
    delta_decay = delta_decay.view(ns, -1)
    for i in range(ns):
        remainder_idx = delta_idx[i]
        valid_mask = np.ones(len(remainder_idx))
        while np.sum(valid_mask) > 0:
            valid_remainder_idx, unique_remainder_idx = np.unique(
                remainder_idx, return_index=True
            )
            rir[i][valid_remainder_idx] += (
                delta_decay[i][unique_remainder_idx]
                * valid_mask[unique_remainder_idx]
            )
            valid_mask[unique_remainder_idx] = 0
            remainder_idx[unique_remainder_idx] = 0

    direct_mask = torch.zeros(ns, rir_length).float()

    for i in range(ns):
        direct_mask[
            i,
            max(direct_idx[i] + sample_sr * direct_range[0] // 1000, 0) : min(
                direct_idx[i] + sample_sr * direct_range[1] // 1000, rir_length
            ),
        ] = 1.0

    rir_direct = rir * direct_mask

    all_rir = torch.stack([rir, rir_direct], 1).view(ns * 2, -1)
    rir_downsample = resample1(all_rir)
    rir_hp = highpass_biquad(
        rir_downsample, sample_sr // int(np.sqrt(ratio)), 80.0
    )
    rir = resample2(rir_hp).float().view(n_mic, num_src, 2, -1)

    return rir[:, :, 0].data.numpy(), rir[:, :, 1].data.numpy()


def sample_mic_arch(n_mic, mic_spacing=None, bounding_box=None):
    if mic_spacing is None:
        mic_spacing = [0.02, 0.10]
    if bounding_box is None:
        bounding_box = [0.08, 0.12, 0]

    sample_n_mic = np.random.randint(n_mic[0], n_mic[1] + 1)
    if sample_n_mic == 1:
        mic_arch = np.array([[0, 0, 0]])
    else:
        mic_arch = []
        while len(mic_arch) < sample_n_mic:
            this_mic_pos = np.random.uniform(
                np.array([0, 0, 0]), np.array(bounding_box)
            )

            if len(mic_arch) != 0:
                ok = True
                for other_mic_pos in mic_arch:
                    this_mic_spacing = np.linalg.norm(
                        this_mic_pos - other_mic_pos
                    )
                    if (
                        this_mic_spacing < mic_spacing[0]
                        or this_mic_spacing > mic_spacing[1]
                    ):
                        ok = False
                        break
                if ok:
                    mic_arch.append(this_mic_pos)
            else:
                mic_arch.append(this_mic_pos)
        mic_arch = np.stack(mic_arch, 0)  # [nmic, 3]
    return mic_arch


def sample_src_pos(
    room_dim,
    num_src,
    array_pos,
    min_mic_dis=0.5,
    max_mic_dis=5,
    min_dis_wall=None,
):
    if min_dis_wall is None:
        min_dis_wall = [0.5, 0.5, 0.5]

    # random sample the source positon
    src_pos = []
    while len(src_pos) < num_src:
        pos = np.random.uniform(
            np.array(min_dis_wall), np.array(room_dim) - np.array(min_dis_wall)
        )
        dis = np.linalg.norm(pos - np.array(array_pos))

        if dis >= min_mic_dis and dis <= max_mic_dis:
            src_pos.append(pos)

    return np.stack(src_pos, 0)


def sample_mic_array_pos(mic_arch, room_dim, min_dis_wall=None):
    """
    Generate the microphone array position according to the given microphone architecture (geometry)
    :param mic_arch: np.array with shape [n_mic, 3]
                    the relative 3D coordinate to the array_pos in (m)
                    e.g., 2-mic linear array [[-0.1, 0, 0], [0.1, 0, 0]];
                    e.g., 4-mic circular array [[0, 0.035, 0], [0.035, 0, 0], [0, -0.035, 0], [-0.035, 0, 0]]
    :param min_dis_wall: minimum distance from the wall in (m)
    :return
        mic_pos: microphone array position in (m) with shape [n_mic, 3]
        array_pos: array CENTER / REFERENCE position in (m) with shape [1, 3]
    """

    def rotate(angle, valuex, valuey):
        rotate_x = valuex * np.cos(angle) + valuey * np.sin(angle)  # [nmic]
        rotate_y = valuey * np.cos(angle) - valuex * np.sin(angle)
        return np.stack(
            [rotate_x, rotate_y, np.zeros_like(rotate_x)], -1
        )  # [nmic, 3]

    if min_dis_wall is None:
        min_dis_wall = [0.5, 0.5, 0.5]

    if isinstance(mic_arch, dict):  # ADHOC ARRAY
        n_mic, mic_spacing, bounding_box = (
            mic_arch["n_mic"],
            mic_arch["spacing"],
            mic_arch["bounding_box"],
        )
        sample_n_mic = np.random.randint(n_mic[0], n_mic[1] + 1)

        if sample_n_mic == 1:
            mic_arch = np.array([[0, 0, 0]])
        else:
            mic_arch = [
                np.random.uniform(np.array([0, 0, 0]), np.array(bounding_box))
            ]
            while len(mic_arch) < sample_n_mic:
                this_mic_pos = np.random.uniform(
                    np.array([0, 0, 0]), np.array(bounding_box)
                )
                ok = True
                for other_mic_pos in mic_arch:
                    this_mic_spacing = np.linalg.norm(
                        this_mic_pos - other_mic_pos
                    )
                    if (
                        this_mic_spacing < mic_spacing[0]
                        or this_mic_spacing > mic_spacing[1]
                    ):
                        ok = False
                        break
                if ok:
                    mic_arch.append(this_mic_pos)
            mic_arch = np.stack(mic_arch, 0)  # [nmic, 3]
    else:
        mic_arch = np.array(mic_arch)

    mic_array_center = np.mean(mic_arch, 0, keepdims=True)  # [1, 3]
    max_radius = max(np.linalg.norm(mic_arch - mic_array_center, axis=-1))
    array_pos = np.random.uniform(
        np.array(min_dis_wall) + max_radius,
        np.array(room_dim) - np.array(min_dis_wall) - max_radius,
    ).reshape(1, 3)
    mic_pos = array_pos + mic_arch
    # assume the array is always horizontal
    rotate_azm = np.random.uniform(-np.pi, np.pi)
    mic_pos = array_pos + rotate(
        rotate_azm, mic_arch[:, 0], mic_arch[:, 1]
    )  # [n_mic, 3]

    return mic_pos, array_pos


def sample_a_config(simu_config):
    room_config = simu_config["min_max_room"]
    rt60_config = simu_config["rt60"]
    mic_dist_config = simu_config["mic_dist"]
    num_src = simu_config["num_src"]
    room_dim = np.random.uniform(
        np.array(room_config[0]), np.array(room_config[1])
    )
    rt60 = np.random.uniform(rt60_config[0], rt60_config[1])
    sr = simu_config["sr"]

    if (
        "array_pos" not in simu_config.keys()
    ):  # mic_arch must be given in this case
        mic_arch = simu_config["mic_arch"]
        mic_pos, array_pos = sample_mic_array_pos(mic_arch, room_dim)
    else:
        array_pos = simu_config["array_pos"]

    if "src_pos" not in simu_config.keys():
        src_pos = sample_src_pos(
            room_dim,
            num_src,
            array_pos,
            min_mic_dis=mic_dist_config[0],
            max_mic_dis=mic_dist_config[1],
        )
    else:
        src_pos = np.array(simu_config["src_pos"])

    return mic_pos, sr, rt60, room_dim, src_pos, array_pos


# === single-channel FRA-RIR ===
def single_channel(simu_config):
    mic_arch = {"n_mic": [1, 1], "spacing": None, "bounding_box": None}
    simu_config["mic_arch"] = mic_arch
    mic_pos, sr, rt60, room_dim, src_pos, array_pos = sample_a_config(
        simu_config
    )

    rir, rir_direct = FRAM_RIR(mic_pos, sr, rt60, room_dim, src_pos, array_pos)
    # with shape [1, n_src, rir_len]
    return rir, rir_direct


# === multi-channel (fixed) ===
def multi_channel_array(simu_config):
    mic_arch = [[-0.05, 0, 0], [0.05, 0, 0]]

    simu_config["mic_arch"] = mic_arch
    mic_pos, sr, rt60, room_dim, src_pos, array_pos = sample_a_config(
        simu_config
    )

    rir, rir_direct = FRAM_RIR(mic_pos, sr, rt60, room_dim, src_pos)
    # with shape [n_mic, n_src, rir_len]
    return rir, rir_direct


# === multi-channel (adhoc) ===
def multi_channel_adhoc(simu_config):
    mic_arch = {
        "n_mic": [1, 3],
        "spacing": [0.02, 0.05],
        "bounding_box": [0.5, 1.0, 0],  # x, y, z
    }
    simu_config["mic_arch"] = mic_arch
    mic_pos, sr, rt60, room_dim, src_pos, array_pos = sample_a_config(
        simu_config
    )

    rir, rir_direct = FRAM_RIR(mic_pos, sr, rt60, room_dim, src_pos)
    # with shape [sample_n_mic, n_src, rir_len]
    return rir, rir_direct


def multi_channel_src_orientation():
    """
    ========================= → y axis
    |                       |
    |    *1          *2     |
    |                       |
    |          ↑            |
    |                       |
    |    *3          *4     |
    |                       |
    =========================
    ↓
    x axis
    """
    sr = 16000
    rt60 = 0.6
    room_dim = [8, 8, 3]
    src_pos = np.array([[4, 4, 1.5]])  # middle of the room
    mic_pos = np.array(
        [[2, 2, 1.5], [2, 6, 1.5], [6, 2, 1.5], [6, 6, 1.5]]  # mic 1, 2
    )  # mic 3, 4
    src_pattern = "sub_cardioid"
    src_orientation_rad = (
        np.array([180, 90]) / 180.0 * np.pi
    )  # facing *front* (negative x axis)

    rir, rir_direct = FRAM_RIR(
        mic_pos,
        sr,
        rt60,
        room_dim=room_dim,
        src_pos=src_pos,
        src_pattern=src_pattern,
        src_orientation_rad=src_orientation_rad,
    )

    return rir, rir_direct


def multi_channel_mic_orientation():
    """
    ========================= → y axis
    |                       |
    |    ↑1          ↓2     |
    |                       |
    |          o            |
    |                       |
    |    ↑3          ↓4     |
    |                       |
    =========================
    ↓
    x axis
    """

    sr = 16000
    rt60 = 0.6
    room_dim = [8, 8, 3]
    src_pos = np.array([[4, 4, 1.5]])  # middle of the room
    mic_pos = np.array(
        [[2, 2, 1.5], [2, 6, 1.5], [6, 2, 1.5], [6, 6, 1.5]]  # mic 1, 2
    )  # mic 3, 4
    mic_pattern = "sub_cardioid"
    mic_orientation_rad = (
        np.array(
            [
                [180, 90],
                [0, 90],  # mic 1 (negative x axis), 2 (positive x axis)
                [180, 90],
                [0, 90],
            ]
        )
        / 180.0
        * np.pi
    )  # mic 3 (negative x axis), 4 (positive x axis)

    rir, rir_direct = FRAM_RIR(
        mic_pos,
        sr,
        rt60,
        room_dim=room_dim,
        src_pos=src_pos,
        mic_pattern=mic_pattern,
        mic_orientation_rad=mic_orientation_rad,
    )
    return rir, rir_direct
