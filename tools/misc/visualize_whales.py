"""Visualize WHALES data in the WARM-V2X composite format.

Each call produces a single composite PNG per agent-frame: a seamless 2x2
camera grid (Front / Left / Back / Right) followed by a matplotlib BEV with
plasma height-coloured LiDAR scatter, a vertical height colourbar, and a
class legend in the lower right.

Sources
-------
frame_info   raw CARLA frame_info.json (rebuilds ego-frame boxes on the fly)
pkl          single (scene, frame, agent) info PKL entry
pkl_grid     batch-render a sample of info PKL entries
coco         a single image from a mono3D COCO JSON (per-camera output)
coco_batch   batch-render from a mono3D COCO JSON

Usage examples
--------------
python tools/misc/visualize_whales.py frame_info \
    --path data/whales/2024-02-25-14-38-30/5/frame_info.json --agent vehicle0

python tools/misc/visualize_whales.py pkl \
    --path data/whales/whales_infos_val.pkl \
    --token 2024-02-25-14-38-30_5_0

python tools/misc/visualize_whales.py pkl_grid \
    --pkls data/whales/whales_infos_{train,val}.pkl \
    --num-per-pkl 20 --one-per-scene --out whales_vis/

python tools/misc/visualize_whales.py coco \
    --path data/whales/whales_infos_val_mono3d.coco.json \
    --image-id 2024-02-25-14-38-30_5_0_camera

python tools/misc/visualize_whales.py coco_batch \
    --path data/whales/whales_infos_val_mono3d.coco.json \
    --num-tokens 20 --one-per-scene --out whales_vis_coco/
"""

import argparse
import json
import math
import os
import os.path as osp
import pickle
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Shared constants (mirror WARM-V2X vis_full_whales.py + whales_converter)
# ---------------------------------------------------------------------------

CLASSES = ('Vehicle', 'Pedestrian', 'Cyclist')
CLS_COLOUR_RGB = {
    'Vehicle':    (255,  80,   0),
    'Pedestrian': ( 50, 200,   0),
    'Cyclist':    (  0, 140, 220),
}
CLS_COLOUR_MPL = {k: tuple(c / 255.0 for c in v)
                  for k, v in CLS_COLOUR_RGB.items()}

CAMERAS = ('camera', 'camera_l', 'camera_b', 'camera_r')
CAM_LABELS = {'camera':   'Front',
              'camera_l': 'Left',
              'camera_b': 'Back',
              'camera_r': 'Right'}
# (top_left, top_right), (bottom_left, bottom_right)
GRID_LAYOUT = (('camera',   'camera_l'),
               ('camera_b', 'camera_r'))

_LIDAR_HEIGHT_VEH = 1.8
_LIDAR_HEIGHT_RSU = 0.0

_IMG_W, _IMG_H = 1920, 1080
_FX = _FY = _IMG_W / (2.0 * math.tan(90.0 * math.pi / 360.0))
_CX, _CY = _IMG_W / 2.0, _IMG_H / 2.0
DEFAULT_K = np.array([[_FX, 0.0, _CX],
                      [0.0, _FY, _CY],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

_VEH_CAMERA_POSES = {
    'camera':   (0.0, 0.0, _LIDAR_HEIGHT_VEH,   0.0),
    'camera_l': (0.0, 0.0, _LIDAR_HEIGHT_VEH, -90.0),
    'camera_r': (0.0, 0.0, _LIDAR_HEIGHT_VEH, +90.0),
    'camera_b': (0.0, 0.0, _LIDAR_HEIGHT_VEH, 180.0),
}
_RSU_CAMERA_POSES = {
    'camera':   (0.0, 0.0, _LIDAR_HEIGHT_RSU,   0.0),
    'camera_l': (0.0, 0.0, _LIDAR_HEIGHT_RSU, -90.0),
    'camera_r': (0.0, 0.0, _LIDAR_HEIGHT_RSU, +90.0),
    'camera_b': (0.0, 0.0, _LIDAR_HEIGHT_RSU, 180.0),
}

_CARLA_CAM_TO_OPENCV = np.array([[0.0,  1.0,  0.0],
                                 [0.0,  0.0, -1.0],
                                 [1.0,  0.0,  0.0]], dtype=np.float64)

# Faces (4 indices each) for near-plane clipping; ordering matches
# `_lidar_box_corners` (bottom 0..3, top 4..7, same x/y assignment).
_BOX_FACES = ((0, 1, 2, 3),
              (4, 5, 6, 7),
              (0, 1, 5, 4),
              (3, 2, 6, 7),
              (0, 3, 7, 4),
              (1, 2, 6, 5))
_NEAR = 0.1

_DEFAULT_HALF_EXTENTS = {
    'Cyclist':    (0.885, 0.40, 0.825),
    'Motorcycle': (0.885, 0.40, 0.825),
    'Vehicle':    (2.25,  1.00, 0.775),
    'Pedestrian': (0.30,  0.30, 0.875),
}

BEV_BG = '#05070f'
BEV_RANGE = (-40.0, 40.0, -40.0, 70.0)    # xmin_right, xmax_right, zmin_fwd, zmax_fwd
GRID_SCALE = 0.4
_DW = int(_IMG_W * GRID_SCALE)
_DH = int(_IMG_H * GRID_SCALE)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _lidar_box_corners(boxes: np.ndarray) -> np.ndarray:
    """Return (N, 8, 3) corners for (N, 7) (cx,cy,cz,L,W,H,yaw) LiDAR boxes."""
    boxes = np.asarray(boxes, dtype=np.float32)
    N = len(boxes)
    if N == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    cx, cy, cz = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    L, W, H = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    yaw = boxes[:, 6]
    # Canonical order: bottom ring (0..3) then top ring (4..7).
    off = np.array([[+1, +1, -1], [+1, -1, -1], [-1, -1, -1], [-1, +1, -1],
                    [+1, +1, +1], [+1, -1, +1], [-1, -1, +1], [-1, +1, +1]],
                   dtype=np.float32) * 0.5
    half = np.stack([L, W, H], axis=1)[:, None, :]
    local = off[None, :, :] * half
    c, s = np.cos(yaw), np.sin(yaw)
    zeros, ones = np.zeros_like(c), np.ones_like(c)
    R = np.stack([np.stack([c, -s, zeros], axis=1),
                  np.stack([s,  c, zeros], axis=1),
                  np.stack([zeros, zeros, ones], axis=1)], axis=1)
    rotated = np.einsum('nij,nkj->nki', R, local)
    centre = np.stack([cx, cy, cz], axis=1)[:, None, :]
    return rotated + centre


def _lidar_to_camera_opencv(points_lidar: np.ndarray,
                            sensor2lidar_rotation: np.ndarray,
                            sensor2lidar_translation: np.ndarray) -> np.ndarray:
    """Transform WHALES LiDAR points (y=left) to OpenCV camera (z=forward)."""
    p = np.asarray(points_lidar, dtype=np.float64).copy()
    p[..., 1] = -p[..., 1]
    R = np.asarray(sensor2lidar_rotation, dtype=np.float64)
    t = np.asarray(sensor2lidar_translation, dtype=np.float64)
    p_carla = (p - t) @ R
    return p_carla @ _CARLA_CAM_TO_OPENCV.T


def _camera_sensor2lidar_from_pose(cam_pose: Tuple[float, float, float, float],
                                   lidar_height: float
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Default camera extrinsics when only (x, y, z, yaw_deg) is available."""
    cam_x, cam_y, cam_z, cam_yaw = cam_pose
    yaw = math.radians(cam_yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    t = np.array([cam_x, cam_y, cam_z - lidar_height], dtype=np.float64)
    return R, t


# ---------------------------------------------------------------------------
# 3D box drawing (Sutherland–Hodgman near-plane clipped faces)
# ---------------------------------------------------------------------------

def _clip_poly_near(poly: Sequence[np.ndarray], near: float):
    """Clip a 3D polygon (list of (3,) points) against z >= near."""
    if not poly:
        return []
    out = []
    n = len(poly)
    for i in range(n):
        curr = poly[i]
        prev = poly[(i - 1) % n]
        curr_in = curr[2] >= near
        prev_in = prev[2] >= near
        if curr_in:
            if not prev_in:
                t = (near - prev[2]) / (curr[2] - prev[2])
                out.append(prev + t * (curr - prev))
            out.append(curr)
        elif prev_in:
            t = (near - prev[2]) / (curr[2] - prev[2])
            out.append(prev + t * (curr - prev))
    return out


def _draw_box3d(img: np.ndarray,
                pts_c: np.ndarray,
                K: np.ndarray,
                colour: Tuple[int, int, int],
                thickness: int = 3) -> None:
    """Draw a box by clipping each of its 6 faces, projecting and outlining.

    ``pts_c`` is a (3, 8) array of corners in the OpenCV camera frame.
    """
    if np.all(pts_c[2, :] < _NEAR):
        return
    H, W = img.shape[:2]
    LIM = 30000
    fx, fy = K[0, 0], K[1, 1]
    cx0, cy0 = K[0, 2], K[1, 2]
    for face in _BOX_FACES:
        poly3d = [pts_c[:, i] for i in face]
        clipped = _clip_poly_near(poly3d, _NEAR)
        if len(clipped) < 2:
            continue
        pts2d = []
        for p in clipped:
            z = p[2]
            u = int(max(-LIM, min(LIM, fx * p[0] / z + cx0)))
            v = int(max(-LIM, min(LIM, fy * p[1] / z + cy0)))
            pts2d.append((u, v))
        m = len(pts2d)
        for k in range(m):
            p1, p2 = pts2d[k], pts2d[(k + 1) % m]
            ok, cp1, cp2 = cv2.clipLine((0, 0, W, H), p1, p2)
            if ok:
                cv2.line(img, cp1, cp2, colour, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Rendering: per-camera and BEV
# ---------------------------------------------------------------------------

def _overlay_camera_label(img_rgb: np.ndarray, cam_key: str) -> None:
    """Blend a dark panel in the top-left and write the camera label on top."""
    label = CAM_LABELS[cam_key]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    pad = 12
    x0, y0 = 20, 20
    overlay = img_rgb.copy()
    cv2.rectangle(overlay, (x0, y0),
                  (x0 + tw + 2 * pad, y0 + th + 2 * pad), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img_rgb, 0.45, 0, img_rgb)
    cv2.putText(img_rgb, label, (x0 + pad, y0 + th + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)


def _render_camera_tile(img_bgr: Optional[np.ndarray],
                        corners_lidar: np.ndarray,
                        names: Sequence[str],
                        cam_info: dict,
                        cam_key: str) -> np.ndarray:
    """Draw 3D wireframes + label onto one camera image, return a resized RGB tile."""
    if img_bgr is None:
        img_rgb = np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)
        cv2.putText(img_rgb, f'no {cam_key}', (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (128, 128, 128), 3,
                    cv2.LINE_AA)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    K = np.asarray(cam_info['cam_intrinsic'])
    R = cam_info['sensor2lidar_rotation']
    t = cam_info['sensor2lidar_translation']
    if len(corners_lidar):
        corners_cam = _lidar_to_camera_opencv(
            corners_lidar.reshape(-1, 3), R, t).reshape(-1, 8, 3)
        for cs, name in zip(corners_cam, names):
            colour = CLS_COLOUR_RGB.get(name, (200, 200, 200))
            _draw_box3d(img_rgb, cs.T, K, colour, thickness=3)
    _overlay_camera_label(img_rgb, cam_key)
    return cv2.resize(img_rgb, (_DW, _DH), interpolation=cv2.INTER_AREA)


def _render_bev(points: Optional[np.ndarray],
                boxes: np.ndarray,
                names: Sequence[str],
                ego_yaw_deg: Optional[float],
                is_rsu: bool,
                pc_range: Tuple[float, float, float, float] = BEV_RANGE
                ) -> np.ndarray:
    """Render the matplotlib BEV panel and return it as an RGB ndarray."""
    xmin, xmax, zmin, zmax = pc_range
    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    ax.set_facecolor(BEV_BG)
    fig.patch.set_facecolor(BEV_BG)

    h_vmin, h_vmax = (-3.75, -1.75) if is_rsu else (-1.75, 0.25)
    sc = None
    if points is not None and len(points):
        z_bev = points[:, 0]                # x = forward
        x_bev = -points[:, 1]               # flip y=left into right-positive
        h_bev = points[:, 2]
        mask = ((z_bev > zmin) & (z_bev < zmax)
                & (x_bev > xmin) & (x_bev < xmax))
        xm, zm, hm = x_bev[mask], z_bev[mask], h_bev[mask]
        ax.scatter(xm, zm, s=4.0, c=hm, cmap='plasma',
                   vmin=h_vmin, vmax=h_vmax, alpha=0.12, linewidths=0)
        sc = ax.scatter(xm, zm, s=1.2, c=hm, cmap='plasma',
                        vmin=h_vmin, vmax=h_vmax, alpha=0.9, linewidths=0)

    corners = _lidar_box_corners(boxes)
    for cs, name in zip(corners, names):
        bottom_idx = np.argsort(cs[:, 2])[:4]
        x_bev_box = -cs[bottom_idx, 1]
        z_bev_box = cs[bottom_idx, 0]
        cxb = x_bev_box.mean()
        czb = z_bev_box.mean()
        order = np.argsort(np.arctan2(z_bev_box - czb, x_bev_box - cxb))
        xb = np.append(x_bev_box[order], x_bev_box[order[0]])
        zb = np.append(z_bev_box[order], z_bev_box[order[0]])
        ax.plot(xb, zb, '-', color=CLS_COLOUR_MPL.get(name, (0.7, 0.7, 0.7)),
                linewidth=1.8)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_xlabel('x (m, rightward)', color='white', fontsize=10)
    ax.set_ylabel('z (m, forward)', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=9)
    title = 'BEV' if ego_yaw_deg is None else f'BEV  ego_yaw={ego_yaw_deg:.1f}'
    ax.set_title(title, color='white', fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label('height (m)', color='white', fontsize=9)
        cbar.ax.tick_params(colors='white', labelsize=8)
        cbar.outline.set_edgecolor('white')

    ax.legend(
        handles=[mpatches.Patch(color=CLS_COLOUR_MPL[c], label=c)
                 for c in CLASSES],
        loc='lower right', fontsize=9, framealpha=0.6,
        facecolor=BEV_BG, edgecolor='white', labelcolor='white')

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    bev_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    bev_arr = bev_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return bev_arr


def _compose(drawn: dict, bev_arr: np.ndarray) -> np.ndarray:
    """Seamless 2x2 camera grid + BEV resized to matching height."""
    rows = [np.concatenate([drawn[c] for c in row], axis=1)
            for row in GRID_LAYOUT]
    cam_grid = np.concatenate(rows, axis=0)
    target_h = cam_grid.shape[0]
    bev_w = int(bev_arr.shape[1] * target_h / bev_arr.shape[0])
    bev_arr = cv2.resize(bev_arr, (bev_w, target_h))
    return np.concatenate([cam_grid, bev_arr], axis=1)


def _write_composite(composite_rgb: np.ndarray, out_path: str) -> None:
    """Save an RGB composite as a BGR PNG on disk."""
    os.makedirs(osp.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Camera image / path resolution
# ---------------------------------------------------------------------------

def _load_lidar_bin(path: str) -> Optional[np.ndarray]:
    """Load a WHALES point_cloud.bin as (N, 4) float32."""
    if not osp.exists(path):
        return None
    raw = np.fromfile(path, dtype=np.float32)
    return raw.reshape(-1, 4) if raw.size else np.zeros((0, 4),
                                                        dtype=np.float32)


def _resolve_pkl_path(data_root: Optional[str], stored: str) -> str:
    """Locate a path that may be absolute or rooted under ``data_root``."""
    if osp.exists(stored):
        return stored
    if data_root:
        rel = stored.lstrip('./')
        for base in (data_root, osp.dirname(data_root.rstrip('/'))):
            cand = osp.join(base, rel)
            if osp.exists(cand):
                return cand
            cand = osp.join(base,
                            osp.basename(osp.dirname(osp.dirname(stored))),
                            osp.basename(osp.dirname(stored)),
                            osp.basename(stored))
            if osp.exists(cand):
                return cand
    raise FileNotFoundError(f'Could not locate {stored!r}')


def _resolve_image_path(data_root: str, file_name: str) -> str:
    """Find a COCO-referenced image on disk."""
    if osp.isabs(file_name) and osp.exists(file_name):
        return file_name
    for base in (data_root, osp.dirname(data_root.rstrip('/'))):
        cand = osp.join(base, file_name)
        if osp.exists(cand):
            return cand
    raise FileNotFoundError(f'Could not locate image {file_name!r}')


def _resolve_camera_image(cam_key: str,
                          agent_dir: Optional[str],
                          cam_info: Optional[dict],
                          data_root: Optional[str]) -> Optional[np.ndarray]:
    """Load a camera image from either a PKL data_path or the agent folder."""
    if cam_info is not None and cam_info.get('data_path') and data_root:
        try:
            return cv2.imread(_resolve_pkl_path(data_root, cam_info['data_path']))
        except FileNotFoundError:
            pass
    if agent_dir:
        p = osp.join(agent_dir, f'{cam_key}.png')
        if osp.exists(p):
            return cv2.imread(p)
    return None


def _cams_from_default_poses(is_rsu: bool) -> dict:
    """Synthetic sensor2lidar for the four fixed CARLA-spawned cameras."""
    lidar_height = _LIDAR_HEIGHT_RSU if is_rsu else _LIDAR_HEIGHT_VEH
    poses = _RSU_CAMERA_POSES if is_rsu else _VEH_CAMERA_POSES
    out = {}
    for cam, pose in poses.items():
        R, t = _camera_sensor2lidar_from_pose(pose, lidar_height)
        out[cam] = dict(cam_intrinsic=DEFAULT_K,
                        sensor2lidar_rotation=R,
                        sensor2lidar_translation=t)
    return out


# ---------------------------------------------------------------------------
# Composite builder (shared by frame_info, pkl, pkl_grid)
# ---------------------------------------------------------------------------

def _build_composite(boxes: np.ndarray,
                     names: Sequence[str],
                     points: Optional[np.ndarray],
                     cams_info: dict,
                     ego_yaw_deg: Optional[float],
                     is_rsu: bool,
                     agent_dir: Optional[str] = None,
                     data_root: Optional[str] = None) -> np.ndarray:
    """Produce the full 2x2 + BEV composite from resolved inputs."""
    corners_lidar = _lidar_box_corners(boxes)
    drawn = {}
    for cam in CAMERAS:
        cam_info = cams_info.get(cam)
        if cam_info is None:
            cam_info = dict(cam_intrinsic=DEFAULT_K,
                            sensor2lidar_rotation=np.eye(3),
                            sensor2lidar_translation=np.zeros(3))
        img_bgr = _resolve_camera_image(cam, agent_dir, cam_info, data_root)
        drawn[cam] = _render_camera_tile(img_bgr, corners_lidar, names,
                                         cam_info, cam)
    bev = _render_bev(points, boxes, names, ego_yaw_deg, is_rsu)
    return _compose(drawn, bev)


# ---------------------------------------------------------------------------
# Source: frame_info.json
# ---------------------------------------------------------------------------

def _annotation_to_ego_box(ann: dict,
                           ego_loc: Sequence[float],
                           ego_yaw_deg: float,
                           is_rsu: bool) -> Sequence[float]:
    """Convert a CARLA-world annotation to a WHALES LiDAR-frame (y=left) box.

    Mirrors the converter/loader so the frame_info path is consistent with
    what the info PKLs contain.
    """
    hx, hy, hz = ann['size']
    cls = ann.get('type', 'Vehicle')
    dflt = _DEFAULT_HALF_EXTENTS.get(cls, (0.885, 0.40, 0.825))
    if hx == 0:
        hx = dflt[0]
    if hy == 0:
        hy = dflt[1]
    if hz == 0:
        hz = dflt[2]
    L, W, H = 2.0 * hx, 2.0 * hy, 2.0 * hz

    cx_w, cy_w, cz_w = ann['location']
    sh = 0.0 if is_rsu else _LIDAR_HEIGHT_VEH
    rx = cx_w - ego_loc[0]
    ry = cy_w - ego_loc[1]
    rz = cz_w - (ego_loc[2] + sh)

    yaw = math.radians(ego_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    cx_rot =  c * rx + s * ry
    cy_rot = -s * rx + c * ry
    cx = cx_rot
    cy = -cy_rot
    cz = rz

    w_q, qx, qy, qz = ann['rotation']
    fwd_wx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_wy = 2.0 * (qx * qy + qz * w_q)
    fwd_cx =  c * fwd_wx + s * fwd_wy
    fwd_cy = -(-s * fwd_wx + c * fwd_wy)
    yaw_ego = math.atan2(fwd_cy, fwd_cx)
    return [cx, cy, cz, L, W, H, yaw_ego]


def _resolve_agent(fi: dict, agent_tag: str) -> Tuple[int, bool]:
    """Map ``vehicleN`` / ``rsu`` to ``(index, is_rsu)``."""
    vnum = int(fi.get('vehicle_num', 0))
    if agent_tag == 'rsu':
        return vnum, True
    if agent_tag.startswith('vehicle'):
        return int(agent_tag[len('vehicle'):]), False
    raise ValueError(f'Unknown agent tag: {agent_tag!r}')


def visualize_frame_info(fi_path: str, agent_tag: str, out_dir: str,
                         scene_dir: Optional[str] = None) -> None:
    with open(fi_path) as f:
        fi = json.load(f)
    if scene_dir is None:
        scene_dir = osp.dirname(osp.dirname(fi_path))
    frame_dir = osp.dirname(fi_path)

    idx, is_rsu = _resolve_agent(fi, agent_tag)
    agent_str = 'rsu' if is_rsu else f'vehicle{idx}'
    agent_dir = osp.join(frame_dir, agent_str)
    if not osp.isdir(agent_dir):
        raise FileNotFoundError(f'No agent folder at {agent_dir}')

    ego_loc = fi['veh_locations'][idx]
    ego_yaw = fi['veh_rotations'][idx][1]

    boxes, names = [], []
    skip_idx = None if is_rsu else idx
    for ann_i, ann in enumerate(fi['sample_annotation']):
        if ann_i == skip_idx:
            continue
        names.append(ann.get('type', 'Vehicle'))
        boxes.append(_annotation_to_ego_box(ann, ego_loc, ego_yaw, is_rsu))
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    names = np.asarray(names)

    points = _load_lidar_bin(osp.join(agent_dir, 'point_cloud.bin'))
    cams_info = _cams_from_default_poses(is_rsu)

    scene_name = osp.basename(osp.normpath(scene_dir))
    frame_name = osp.basename(osp.normpath(frame_dir))
    tag = f'{scene_name}_{frame_name}_{agent_str}'

    composite = _build_composite(boxes, names, points, cams_info,
                                 ego_yaw_deg=ego_yaw, is_rsu=is_rsu,
                                 agent_dir=agent_dir)
    out_path = osp.join(out_dir, f'{tag}.png')
    _write_composite(composite, out_path)
    print(f'wrote {out_path}')


# ---------------------------------------------------------------------------
# Source: info PKL (single entry + batch grid)
# ---------------------------------------------------------------------------

def _info_for_token(pkl_data: dict, token: str) -> dict:
    for info in pkl_data['infos']:
        if info['token'] == token:
            return info
    preview = [i['token'] for i in pkl_data['infos'][:5]]
    raise KeyError(f'Token {token!r} not found. First 5 tokens: {preview}')


def _info_ego_yaw_deg(info: dict) -> Optional[float]:
    """Recover ego yaw (degrees) from info keys if any carry it."""
    for key in ('ego_yaw_deg', 'ego_yaw'):
        if key in info:
            v = info[key]
            return float(v) if v is not None else None
    pose = info.get('ego_pose') or {}
    for key in ('yaw', 'yaw_deg'):
        if key in pose:
            return float(pose[key])
    return None


def _info_is_rsu(info: dict) -> bool:
    """Heuristic: RSU tokens carry ``rsu`` in the agent position."""
    agent = info.get('agent')
    if agent:
        return str(agent).lower() == 'rsu'
    tok = str(info.get('token', ''))
    return tok.rsplit('_', 1)[-1].lower() == 'rsu'


def visualize_pkl(pkl_path: str, token: str, out_dir: str,
                  data_root: Optional[str] = None) -> None:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    info = _info_for_token(data, token)

    gt_boxes = np.asarray(info['gt_boxes']).reshape(-1, 7)
    names = np.asarray(info['gt_names'])
    try:
        lidar_path = _resolve_pkl_path(data_root, info['lidar_path'])
        points = _load_lidar_bin(lidar_path)
    except FileNotFoundError:
        points = None

    composite = _build_composite(
        gt_boxes, names, points, info['cams'],
        ego_yaw_deg=_info_ego_yaw_deg(info),
        is_rsu=_info_is_rsu(info),
        data_root=data_root)
    _write_composite(composite, osp.join(out_dir, f'{info["token"]}.png'))
    print(f'wrote {info["token"]}.png in {out_dir}')


def _pick_tokens(infos: Sequence[dict], num: int,
                 one_per_scene: bool) -> Sequence[str]:
    if one_per_scene:
        seen, out = set(), []
        for info in infos:
            scene = info['token'].rsplit('_', 2)[0]
            if scene in seen:
                continue
            seen.add(scene)
            out.append(info['token'])
            if len(out) >= num:
                break
        return out
    if num >= len(infos):
        return [i['token'] for i in infos]
    step = max(1, len(infos) // num)
    return [infos[i]['token'] for i in range(0, len(infos), step)][:num]


def visualize_pkl_grid(pkl_paths: Sequence[str],
                       out_dir: str,
                       num_per_pkl: int,
                       data_root: Optional[str],
                       one_per_scene: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        infos = data['infos']
        split = osp.splitext(osp.basename(pkl_path))[0]
        tokens = _pick_tokens(infos, num_per_pkl, one_per_scene)
        print(f'{split}: {len(tokens)} frames')
        token_to_info = {i['token']: i for i in infos}
        for tok in tokens:
            info = token_to_info[tok]
            gt_boxes = np.asarray(info['gt_boxes']).reshape(-1, 7)
            names = np.asarray(info['gt_names'])
            try:
                lp = _resolve_pkl_path(data_root, info['lidar_path'])
                points = _load_lidar_bin(lp)
            except FileNotFoundError:
                points = None
            try:
                composite = _build_composite(
                    gt_boxes, names, points, info['cams'],
                    ego_yaw_deg=_info_ego_yaw_deg(info),
                    is_rsu=_info_is_rsu(info),
                    data_root=data_root)
            except FileNotFoundError as e:
                print(f'  skip {tok}: {e}')
                continue
            _write_composite(composite,
                             osp.join(out_dir, f'{split}__{tok}.png'))
            total += 1
    print(f'wrote {total} composite(s) to {out_dir}')


# ---------------------------------------------------------------------------
# Source: mono3D COCO JSON
# ---------------------------------------------------------------------------

def _cam3d_box_corners(cx: float, cy: float, cz: float,
                       L: float, H: float, W: float, ry: float) -> np.ndarray:
    """Return (8, 3) corners for an mmdet3d (x, y, z, l, h, w, ry) cam box.

    OpenCV camera frame: +x right, +y down, +z forward.  ``l`` runs along
    ``+z``, ``w`` along ``+x``, ``h`` along ``+y``; ``ry`` rotates around ``+y``.
    """
    x_off = np.array([+1, +1, -1, -1, +1, +1, -1, -1]) * 0.5 * W
    y_off = np.array([-1, -1, -1, -1, +1, +1, +1, +1]) * 0.5 * H
    z_off = np.array([+1, -1, -1, +1, +1, -1, -1, +1]) * 0.5 * L
    c, s = math.cos(ry), math.sin(ry)
    x = c * x_off + s * z_off + cx
    y = y_off + cy
    z = -s * x_off + c * z_off + cz
    return np.stack([x, y, z], axis=1)


def _draw_coco_annotations(image_bgr: np.ndarray, K: np.ndarray,
                           anns: Sequence[dict]) -> None:
    """Draw mono3D COCO annotations (3D wireframe or fallback 2D) in place.

    The image is expected to be BGR; we draw in BGR so cv2.imwrite rounds
    trip correctly without a redundant colour conversion.
    """
    for ann in anns:
        colour_rgb = CLS_COLOUR_RGB.get(ann['category_name'], (0, 255, 0))
        colour_bgr = (colour_rgb[2], colour_rgb[1], colour_rgb[0])
        x, y, w, h = ann['bbox']
        if 'bbox_cam3d' in ann:
            cx, cy, cz, L, H_cam, W_cam, ry = ann['bbox_cam3d']
            corners = _cam3d_box_corners(cx, cy, cz, L, H_cam, W_cam, ry)
            _draw_box3d(image_bgr, corners.T, K, colour_bgr, thickness=2)
        else:
            p1 = (int(round(x)), int(round(y)))
            p2 = (int(round(x + w)), int(round(y + h)))
            cv2.rectangle(image_bgr, p1, p2, colour_bgr, 2, cv2.LINE_AA)
        cv2.putText(image_bgr, ann['category_name'],
                    (int(round(x)), max(int(round(y)) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_bgr, 1, cv2.LINE_AA)


def visualize_coco(coco_path: str, image_id: str, data_root: str,
                   out_dir: str) -> None:
    with open(coco_path) as f:
        coco = json.load(f)

    id_to_img = {im['id']: im for im in coco['images']}
    if image_id not in id_to_img:
        sample_ids = list(id_to_img)[:5]
        raise KeyError(f'image_id {image_id!r} not found. '
                       f'First 5 ids: {sample_ids}')
    image = id_to_img[image_id]
    anns = [a for a in coco['annotations'] if a['image_id'] == image_id]

    img_path = _resolve_image_path(data_root, image['file_name'])
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'cv2 could not read {img_path}')
    K = np.asarray(image['cam_intrinsic'])
    _draw_coco_annotations(img, K, anns)

    os.makedirs(out_dir, exist_ok=True)
    out = osp.join(out_dir, f'{image_id}.png')
    cv2.imwrite(out, img)
    print(f'wrote {out} ({len(anns)} annotations)')


def visualize_coco_batch(coco_path: str, data_root: str, out_dir: str,
                         num_tokens: int, one_per_scene: bool) -> None:
    """Render a sampled subset of images from a mono3D COCO JSON."""
    with open(coco_path) as f:
        coco = json.load(f)

    ann_by_img: dict = {}
    for a in coco['annotations']:
        ann_by_img.setdefault(a['image_id'], []).append(a)

    by_token: dict = {}
    for im in coco['images']:
        token = im.get('token') or im['id'].rsplit('_', 2)[0]
        by_token.setdefault(token, []).append(im)
    tokens = list(by_token)
    if one_per_scene:
        seen, picked = set(), []
        for tok in tokens:
            scene = tok.rsplit('_', 2)[0]
            if scene in seen:
                continue
            seen.add(scene)
            picked.append(tok)
            if len(picked) >= num_tokens:
                break
        tokens = picked
    else:
        step = max(1, len(tokens) // num_tokens)
        tokens = tokens[::step][:num_tokens]

    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for tok in tokens:
        for im in by_token[tok]:
            img_path = _resolve_image_path(data_root, im['file_name'])
            image = cv2.imread(img_path)
            if image is None:
                print(f'  skip {im["id"]}: unreadable {img_path}')
                continue
            K = np.asarray(im['cam_intrinsic'])
            _draw_coco_annotations(image, K, ann_by_img.get(im['id'], []))
            cv2.imwrite(osp.join(out_dir, f'{im["id"]}.png'), image)
            total += 1
    print(f'wrote {total} image(s) to {out_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Visualize WHALES data in the WARM-V2X composite format.')
    sub = p.add_subparsers(dest='source', required=True)

    p_fi = sub.add_parser('frame_info', help='raw CARLA frame_info.json')
    p_fi.add_argument('--path', required=True, help='Path to frame_info.json')
    p_fi.add_argument('--agent', default='vehicle0',
                      help='Agent tag (vehicle0..vehicleN or rsu)')
    p_fi.add_argument('--scene-dir', default=None,
                      help='Override scene directory (defaults to parent of '
                           'the frame_info.json folder)')
    p_fi.add_argument('--out', default='whales_vis/')

    p_pkl = sub.add_parser('pkl', help='whales_infos_*.pkl')
    p_pkl.add_argument('--path', required=True)
    p_pkl.add_argument('--token', required=True,
                       help='scene_frame_agent token, e.g. '
                            '2024-02-25-14-38-30_5_0')
    p_pkl.add_argument('--data-root', default='data/whales/',
                       help='Root used to resolve lidar/image paths stored '
                            'in the PKL')
    p_pkl.add_argument('--out', default='whales_vis/')

    p_grid = sub.add_parser(
        'pkl_grid',
        help='Batch-render the composite format for a sample of PKL entries.')
    p_grid.add_argument('--pkls', nargs='+', required=True,
                        help='One or more whales_infos_*.pkl paths.')
    p_grid.add_argument('--out', default='whales_vis/')
    p_grid.add_argument('--num-per-pkl', type=int, default=20,
                        help='How many frames to render per PKL (default 20).')
    p_grid.add_argument('--data-root', default='data/whales/',
                        help='Root used to resolve lidar/image paths.')
    p_grid.add_argument('--one-per-scene', action='store_true',
                        help='Pick the first frame of each scene (capped at '
                             '--num-per-pkl).')

    p_coco = sub.add_parser('coco', help='whales_infos_*_mono3d.coco.json')
    p_coco.add_argument('--path', required=True)
    p_coco.add_argument('--image-id', required=True,
                        help='Value of image.id / annotation.image_id')
    p_coco.add_argument('--data-root', default='data/whales/')
    p_coco.add_argument('--out', default='whales_vis_coco/')

    p_coco_batch = sub.add_parser(
        'coco_batch',
        help='Batch-render a sample of images from a mono3D COCO JSON.')
    p_coco_batch.add_argument('--path', required=True)
    p_coco_batch.add_argument('--data-root', default='data/whales/')
    p_coco_batch.add_argument('--out', default='whales_vis_coco/')
    p_coco_batch.add_argument('--num-tokens', type=int, default=20,
                              help='How many frame tokens to render (each '
                                   'produces 4 camera images).')
    p_coco_batch.add_argument('--one-per-scene', action='store_true',
                              help='Pick one frame token per scene.')

    return p


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.source == 'frame_info':
        visualize_frame_info(args.path, args.agent, args.out,
                             scene_dir=args.scene_dir)
    elif args.source == 'pkl':
        visualize_pkl(args.path, args.token, args.out,
                      data_root=args.data_root)
    elif args.source == 'pkl_grid':
        visualize_pkl_grid(
            pkl_paths=args.pkls,
            out_dir=args.out,
            num_per_pkl=args.num_per_pkl,
            data_root=args.data_root,
            one_per_scene=args.one_per_scene,
        )
    elif args.source == 'coco':
        visualize_coco(args.path, args.image_id, args.data_root, args.out)
    elif args.source == 'coco_batch':
        visualize_coco_batch(
            coco_path=args.path,
            data_root=args.data_root,
            out_dir=args.out,
            num_tokens=args.num_tokens,
            one_per_scene=args.one_per_scene,
        )
    else:  # pragma: no cover — argparse enforces choices
        raise SystemExit(f'unknown source {args.source!r}')


if __name__ == '__main__':
    main()
