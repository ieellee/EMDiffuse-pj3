import sys
import os
sys.path.append('RAFT/core')
from RAFT.core.raftConfig import RaftConfig
from RAFT.core.register import registration
import urllib
import zipfile
import glob
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import core.praser as Praser
import os
import warnings
import torch
import torch.multiprocessing as mp
import shutil
from PIL import Image
import cv2
import numpy as np

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_EMDiffuse
from emdiffuse_conifg import EMDiffuseConfig
from run import main_worker

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def prepare_training_data(source_path, target_path, categories):
    """
    准备训练数据，将PNG图像转换为TIFF格式，并组织为合适的目录结构
    
    Args:
        source_path: 原始数据集路径，如 /home/user9/reproduce/EMDiffuse/pj3/us_train2
        target_path: 准备好的训练数据保存路径
        categories: 类别列表，如 ["breast", "carotid", "kidney", "liver", "thyroid"]
    """
    # 确保目标目录是空的，防止之前的数据干扰
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path, exist_ok=True)
    
    # 注意：EMDiffuse需要特定的目录结构
    # 创建训练数据目录结构 - 按照RAFT与EMDiffuse的要求
    os.makedirs(os.path.join(target_path, 'denoise'), exist_ok=True)
    train_wf_dir = os.path.join(target_path, 'denoise', 'train_wf')  # 低质量图像
    train_gt_dir = os.path.join(target_path, 'denoise', 'train_gt')  # 高质量图像
    os.makedirs(train_wf_dir, exist_ok=True)
    os.makedirs(train_gt_dir, exist_ok=True)
    
    image_count = 0
    
    for category in categories:
        print(f"处理类别: {category}")
        
        # 源目录路径
        high_quality_dir = os.path.join(source_path, category, "high_quality")
        low_quality_dir = os.path.join(source_path, category, "low_quality")
        
        # 获取所有PNG图像
        high_quality_images = glob.glob(os.path.join(high_quality_dir, "*.png"))
        low_quality_images = glob.glob(os.path.join(low_quality_dir, "*.png"))
        
        # 确保两个目录中有图像
        if not high_quality_images or not low_quality_images:
            print(f"警告：{category} 类别下未找到图像文件")
            continue
        
        # 创建类别目录
        category_wf_dir = os.path.join(train_wf_dir, category)
        category_gt_dir = os.path.join(train_gt_dir, category)
        os.makedirs(category_wf_dir, exist_ok=True)
        os.makedirs(category_gt_dir, exist_ok=True)
        
        # 处理每一对图像
        for low_quality_image in low_quality_images:
            image_name = os.path.basename(low_quality_image)
            high_quality_image = os.path.join(high_quality_dir, image_name)
            
            # 检查对应的高质量图像是否存在
            if not os.path.exists(high_quality_image):
                print(f"警告：找不到与低质量图像 {image_name} 对应的高质量图像，跳过")
                continue
            
            # 读取图像
            low_img = cv2.imread(low_quality_image, cv2.IMREAD_GRAYSCALE)
            high_img = cv2.imread(high_quality_image, cv2.IMREAD_GRAYSCALE)
            
            if low_img is None or high_img is None:
                print(f"警告：无法读取图像对 {image_name}")
                continue
                
            # 确保图像尺寸一致（光流配准需要）
            if low_img.shape != high_img.shape:
                high_img = cv2.resize(high_img, (low_img.shape[1], low_img.shape[0]))
            
            # 创建子目录来模拟训练集结构
            sample_wf_dir = os.path.join(category_wf_dir, f'sample_{image_count}')
            sample_gt_dir = os.path.join(category_gt_dir, f'sample_{image_count}')
            os.makedirs(sample_wf_dir, exist_ok=True)
            os.makedirs(sample_gt_dir, exist_ok=True)
            
            # 保存为TIFF格式
            tiff_name = image_name.replace('.png', '.tif')
            imwrite(os.path.join(sample_wf_dir, tiff_name), low_img)
            imwrite(os.path.join(sample_gt_dir, tiff_name), high_img)
            
            image_count += 1
    
    print(f"共处理了 {image_count} 对图像")
    return target_path

def process_without_registration(source_path, categories):
    """
    跳过RAFT配准步骤，直接准备训练数据（当图像已经对齐时使用）
    """
    dataset_path = './pj3_train_data_direct'
    # 清理目标目录
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    
    # 创建目录结构
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'denoise'), exist_ok=True)
    denoise_dir = os.path.join(dataset_path, 'denoise')
    train_wf_dir = os.path.join(denoise_dir, 'train_wf')
    train_gt_dir = os.path.join(denoise_dir, 'train_gt')
    os.makedirs(train_wf_dir, exist_ok=True)
    os.makedirs(train_gt_dir, exist_ok=True)
    
    image_count = 0
    
    for category in categories:
        print(f"处理类别: {category}")
        
        # 源目录路径
        high_quality_dir = os.path.join(source_path, category, "high_quality")
        low_quality_dir = os.path.join(source_path, category, "low_quality")
        
        # 获取所有PNG图像
        high_quality_images = glob.glob(os.path.join(high_quality_dir, "*.png"))
        low_quality_images = glob.glob(os.path.join(low_quality_dir, "*.png"))
        
        if not high_quality_images or not low_quality_images:
            print(f"警告：{category} 类别下未找到图像文件")
            continue
            
        # 处理每一对图像
        for low_quality_image in low_quality_images:
            image_name = os.path.basename(low_quality_image)
            high_quality_image = os.path.join(high_quality_dir, image_name)
            
            if not os.path.exists(high_quality_image):
                continue
                
            # 读取图像
            low_img = cv2.imread(low_quality_image, cv2.IMREAD_GRAYSCALE)
            high_img = cv2.imread(high_quality_image, cv2.IMREAD_GRAYSCALE)
            
            if low_img is None or high_img is None:
                continue
            
            # 确保尺寸一致
            if low_img.shape != high_img.shape:
                high_img = cv2.resize(high_img, (low_img.shape[1], low_img.shape[0]))
            
            # 创建单独的样本目录
            low_sample_dir = os.path.join(train_wf_dir, f'sample_{image_count}')
            high_sample_dir = os.path.join(train_gt_dir, f'sample_{image_count}')
            os.makedirs(low_sample_dir, exist_ok=True)
            os.makedirs(high_sample_dir, exist_ok=True)
            
            # 保存为TIFF
            tiff_name = f"{image_count}.tif"
            imwrite(os.path.join(low_sample_dir, tiff_name), low_img)
            imwrite(os.path.join(high_sample_dir, tiff_name), high_img)
            
            image_count += 1
    
    print(f"共处理了 {image_count} 对图像（不经过配准）")
    return dataset_path

def check_data_structure(path):
    """检查数据结构是否符合EMDiffuse的要求"""
    required_dirs = [
        os.path.join(path, 'denoise'),
        os.path.join(path, 'denoise', 'train_wf'),
        os.path.join(path, 'denoise', 'train_gt')
    ]
    
    for d in required_dirs:
        if not os.path.exists(d) or not os.path.isdir(d):
            print(f"错误：目录 {d} 不存在")
            return False
    
    # 检查是否有样本
    samples_wf = glob.glob(os.path.join(path, 'denoise', 'train_wf', '*', '*', '*.tif'))
    samples_gt = glob.glob(os.path.join(path, 'denoise', 'train_gt', '*', '*', '*.tif'))
    
    if not samples_wf or not samples_gt:
        print(f"错误：找不到训练样本，WF样本：{len(samples_wf)}，GT样本：{len(samples_gt)}")
        return False
        
    print(f"数据结构检查通过，找到 {len(samples_wf)} 个WF样本和 {len(samples_gt)} 个GT样本")
    return True

def main():
    # 原始数据集路径
    source_dataset_path = '/home/user9/reproduce/EMDiffuse/pj3/us_train2'
    # 准备好的训练数据路径
    target_dataset_path = './pj3_train_data'
    # 类别
    categories = ["breast", "carotid", "kidney", "liver", "thyroid"]
    
    # 准备训练数据
    print("开始准备训练数据...")
    prepare_training_data(source_dataset_path, target_dataset_path, categories)
    
    # 尝试配准和裁剪
    try:
        print("开始光流配准和裁剪...")
        register_config = RaftConfig(path=target_dataset_path, patch_size=256, border=32, overlap=0.125)
        registration(register_config)
        
        # 检查配准和裁剪后的目录结构
        if not check_data_structure(target_dataset_path):
            print("光流配准后的数据结构不正确，尝试直接处理数据...")
            target_dataset_path = process_without_registration(source_dataset_path, categories)
    except Exception as e:
        print(f"光流配准时出错: {e}")
        print("尝试直接处理数据，跳过配准步骤...")
        target_dataset_path = process_without_registration(source_dataset_path, categories)
    
    # 查看处理后的数据
    noise_images = glob.glob(os.path.join(target_dataset_path, 'denoise', 'train_wf', '*', '*', '*.tif'))
    if len(noise_images) != 0:
        print(f"找到 {len(noise_images)} 个训练样本")
        for i in range(min(3, len(noise_images))):
            noisy_img = imread(noise_images[i])
            gt_img_path = noise_images[i].replace('train_wf', 'train_gt')
            if os.path.exists(gt_img_path):
                gt_img = imread(gt_img_path)
                print(f"样本 {i+1}: 形状 = {noisy_img.shape}, GT形状 = {gt_img.shape}")
            else:
                print(f"样本 {i+1}: 形状 = {noisy_img.shape}, GT不存在")
    else:
        print("警告：未找到训练样本，请检查数据准备过程")
        return
    
    # 创建EMDiffuse配置
    print("配置EMDiffuse模型...")
    config = EMDiffuseConfig(
        config='config/EMDiffuse-n.json', 
        phase='train', 
        path=os.path.join(target_dataset_path, 'denoise', 'train_wf'), 
        batch_size=4,  # 可根据GPU内存调整
        lr=5e-5,        # 学习率
        epochs=100,     # 训练轮数
        save_checkpoint_freq=10,  # 每隔多少轮保存一次
        val_freq=5      # 每隔多少轮验证一次
    )
    
    # 解析配置
    opt = Praser.parse(config)
    opt['world_size'] = 1
    
    # 设置随机种子
    Util.set_seed(opt['seed'])
    
    # 创建并训练模型
    print("开始创建并训练模型...")
    model = create_EMDiffuse(opt)
    print(f'训练数据加载器长度: {len(model.phase_loader)}, 验证数据加载器长度: {len(model.val_loader)}')
    
    # 训练模型
    model.train()
    
    print("训练完成！")
    print(f"模型保存在: {opt['path']['experiments_root']}")

if __name__ == "__main__":
    main() 