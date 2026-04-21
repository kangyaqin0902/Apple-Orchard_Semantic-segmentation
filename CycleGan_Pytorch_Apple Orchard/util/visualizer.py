import os
import ntpath
import numpy as np
import rasterio
from . import util

# 全局存储原始影像的【完整地理信息+原始尺寸】
global_meta = None
global_affine = None
global_crs = None
global_ori_height = None
global_ori_width = None
global_ori_count = None


def set_geo_info(meta, affine, crs, height, width, count):
    """接收原始影像的所有信息，全程不变"""
    global global_meta, global_affine, global_crs, global_ori_height, global_ori_width, global_ori_count
    global_meta = meta.copy()  # 深度复制元数据，防止被修改
    global_affine = affine  # 原始仿射矩阵（坐标核心）
    global_crs = crs  # 原始投影坐标系
    global_ori_height = height  # 原始影像高度（像素）
    global_ori_width = width  # 原始影像宽度（像素）
    global_ori_count = count  # 原始影像通道数


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    webpage.add_header(name)
    ims, txts, links = [], []

    for label, im_data in visuals.items():
        # 1. tensor转原始像素值，反归一化，严格限制0-255（和原始影像一致）
        im_numpy = util.tensor2im(im_data)
        im_numpy = np.clip(im_numpy, 0, 255).astype(np.uint8)

        # 2. 严格匹配原始影像的维度和尺寸【核心修复】
        # 单通道影像 (灰度遥感图)
        if len(im_numpy.shape) == 2:
            im_final = np.expand_dims(im_numpy, axis=0)
        # 多通道影像 (RGB/3通道遥感图)
        else:
            im_final = np.transpose(im_numpy, (2, 0, 1))

        # ✅ 终极保障：强制让生成结果的尺寸 = 原始影像尺寸，像素数完全一致
        # 哪怕模型输出尺寸有偏差，也强制修正为原始尺寸，杜绝任何错位可能
        im_final = np.resize(im_final, (global_ori_count, global_ori_height, global_ori_width))

        # 3. 保存为带完整地理信息的GeoTIFF
        save_name = f"{name}_{label}.tif"
        save_path = os.path.join(image_dir, save_name)

        # ✅ 关键中的关键：写入原始影像的【完整元数据】，坐标100%对齐
        if global_meta is not None and global_crs is not None:
            # 更新元数据：只替换像素值，其余所有信息（坐标/投影/尺寸/波段）全部用原始的
            out_meta = global_meta.copy()
            out_meta.update({
                'dtype': im_final.dtype,
                'height': global_ori_height,
                'width': global_ori_width,
                'count': global_ori_count,
                'transform': global_affine,
                'crs': global_crs
            })
            # 写入GeoTIFF，无任何信息丢失
            with rasterio.open(save_path, 'w', **out_meta) as dst:
                dst.write(im_final)
        else:
            # 兼容普通TIF，无坐标
            with rasterio.open(save_path, 'w', driver='GTiff', height=im_final.shape[1], width=im_final.shape[2],
                               count=im_final.shape[0], dtype=im_final.dtype) as dst:
                dst.write(im_final)

        ims.append(save_name)
        txts.append(label)
        links.append(save_name)
    webpage.add_images(ims, txts, links, width=width)