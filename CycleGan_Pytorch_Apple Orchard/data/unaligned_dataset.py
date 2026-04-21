import os
import numpy as np
from data.base_dataset import BaseDataset, is_image_file
from PIL import Image
import rasterio
import torchvision.transforms as transforms


class UnalignedDataset(BaseDataset):
    """CycleGAN双向测试专用数据集，读取testA+testB，完美保存GeoTIFF地理信息"""

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # testA路径
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # testB路径

        self.A_paths = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A) if is_image_file(f)])
        self.B_paths = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B) if is_image_file(f)])

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform_A = self.get_transform(opt, grayscale=(opt.input_nc == 1))
        self.transform_B = self.get_transform(opt, grayscale=(opt.output_nc == 1))

        # ✅ 修复点1：在初始化函数中 预定义所有地理信息属性（全局属性，外部可读取）
        self.meta_A = None
        self.affine_A = None
        self.crs_A = None
        self.height_A = None
        self.width_A = None
        self.count_A = None
        self.meta_B = None
        self.affine_B = None
        self.crs_B = None
        self.height_B = None
        self.width_B = None
        self.count_B = None

        # ✅ 修复点2：提前读取第一张图片的地理信息，永久存入全局属性
        if len(self.A_paths) > 0:
            with rasterio.open(self.A_paths[0], 'r') as src_A:
                self.meta_A = src_A.meta
                self.affine_A = src_A.transform
                self.crs_A = src_A.crs
                self.height_A = src_A.height
                self.width_A = src_A.width
                self.count_A = src_A.count
        if len(self.B_paths) > 0:
            with rasterio.open(self.B_paths[0], 'r') as src_B:
                self.meta_B = src_B.meta
                self.affine_B = src_B.transform
                self.crs_B = src_B.crs
                self.height_B = src_B.height
                self.width_B = src_B.width
                self.count_B = src_B.count

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        # 读取图片像素数据，正常做数据增强/转换
        with rasterio.open(A_path, 'r') as src_A:
            img_A = src_A.read()
            img_A = np.transpose(img_A, (1, 2, 0))
            img_A = Image.fromarray(np.uint8(img_A))

        with rasterio.open(B_path, 'r') as src_B:
            img_B = src_B.read()
            img_B = np.transpose(img_B, (1, 2, 0))
            img_B = Image.fromarray(np.uint8(img_B))

        A = self.transform_A(img_A)
        B = self.transform_B(img_B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    # 封装transform函数，避免导入报错
    def get_transform(self, opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC,
                      convert=True):
        from data.base_dataset import get_transform
        return get_transform(opt, params, grayscale, method, convert)