import os
import numpy as np
import torch
import random
from glob import glob
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
from scipy import signal
import shutil
from tifffile import imread

# 导入EMDiffuse相关模块
import core.praser as Praser
import core.util as Util
from models import create_EMDiffuse
from emdiffuse_conifg import EMDiffuseConfig
from crop_single_file import crop

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def calculate_mse(img1, img2):
    """计算均方误差 (Mean Squared Error)"""
    return np.mean((img1 - img2) ** 2)

def calculate_lncc(img1, img2, window_size=9):
    """计算局部归一化互相关 (Local Normalized Cross-Correlation)"""
    # 使用更高效的卷积方法计算LNCC
    if window_size % 2 == 0:
        window_size += 1
    
    # 创建高斯窗口
    kernel_size = window_size
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    # 计算局部均值
    img1_mean = signal.convolve2d(img1, kernel, mode='same')
    img2_mean = signal.convolve2d(img2, kernel, mode='same')
    
    # 减去局部均值
    img1_local = img1 - img1_mean
    img2_local = img2 - img2_mean
    
    # 计算局部标准差
    img1_var = signal.convolve2d(img1_local**2, kernel, mode='same')
    img2_var = signal.convolve2d(img2_local**2, kernel, mode='same')
    img1_std = np.sqrt(np.maximum(img1_var, 1e-6))
    img2_std = np.sqrt(np.maximum(img2_var, 1e-6))
    
    # 计算局部协方差
    img_cov = signal.convolve2d(img1_local * img2_local, kernel, mode='same')
    
    # 计算LNCC
    lncc = img_cov / (img1_std * img2_std + 1e-6)
    
    return np.mean(lncc)

def load_and_preprocess_image(image_path):
    """加载并预处理图像"""
    img = Image.open(image_path).convert('L')  # 转为灰度图
    img_np = np.array(img).astype(np.float32) / 255.0  # 归一化到 [0, 1]
    return img_np

def main():
    # 定义评测结果保存路径
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建vis文件夹用于保存三种图片
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 数据集基本路径
    base_path = "/home/user9/reproduce/EMDiffuse/pj3/us_test2"
    categories = ["breast", "carotid", "kidney", "liver", "thyroid"]
    
    # 存储所有结果
    all_results = {}
    category_results = {}
    
    for category in categories:
        print(f"正在处理类别: {category}")
        
        # 构建高质量和低质量图像路径
        high_quality_dir = os.path.join(base_path, category, "high_quality")
        low_quality_dir = os.path.join(base_path, category, "low_quality")
        
        # 获取所有图像文件
        high_quality_images = glob(os.path.join(high_quality_dir, "*.png"))
        low_quality_images = glob(os.path.join(low_quality_dir, "*.png"))
        
        # 确保有图像可用
        if not high_quality_images or not low_quality_images:
            print(f"警告：{category} 类别下没有找到图像")
            continue
        
        # 随机选择一张图像进行评估
        random_idx = random.randint(0, min(len(high_quality_images), len(low_quality_images)) - 1)
        
        # 获取图像名称
        low_quality_image = low_quality_images[random_idx]
        image_name = os.path.basename(low_quality_image)
        high_quality_image = os.path.join(high_quality_dir, image_name)
        
        # 确认高质量图像存在
        if not os.path.exists(high_quality_image):
            print(f"警告：找不到与低质量图像 {image_name} 对应的高质量图像")
            continue
        
        print(f"  选择的图像: {image_name}")
        
        # 创建临时目录
        temp_dir = os.path.join('./temp_data', category)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 读取低质量图像
        img = cv2.imread(low_quality_image)
        if img is None:
            print(f"警告：无法读取图像 {low_quality_image}")
            continue
        
        # 对于每个类别创建目录
        crop_save_path = os.path.join(temp_dir, f'noise_1')
        os.makedirs(crop_save_path, exist_ok=True)
        
        # 将PNG转换为TIFF格式保存，确保模型能够正确处理
        tiff_path = os.path.join(crop_save_path, image_name.replace('.png', '.tif'))
        cv2.imwrite(tiff_path, img)
        
        # 创建EMDiffuse配置
        config = EMDiffuseConfig(
            config='config/EMDiffuse-n.json', 
            phase='test',
            path=os.path.dirname(temp_dir),  # 使用临时目录的父目录
            batch_size=1, 
            mean=1, 
            resume='./experiments/EMDiffuse-n/best', 
            step=200
        )
        
        # 解析配置
        opt = Praser.parse(config)
        opt['world_size'] = 1
        
        # 设置随机种子
        Util.set_seed(opt['seed'])
        
        # 打印配置信息以便调试
        print(opt)
        print(type(opt))
        
        # 创建模型
        model = create_EMDiffuse(opt)
        print(f'len test data loader: {len(model.phase_loader)}')
        
        # 运行测试
        model.test()
        
        # 获取结果路径
        results_path = os.path.join(opt['path']['experiments_root'], 'results', 'test', '0')
        print(f"查找输出结果的路径: {results_path}")
        
        # 查找所有输出结果文件，尝试不同的扩展名
        result_files = []
        for ext in ['.tif', '.png', '.jpg']:
            result_files.extend(glob(os.path.join(results_path, f'*{ext}')))
        
        if not result_files:
            print(f"在 {results_path} 目录下未找到任何结果文件")
            # 列出目录中的所有文件
            if os.path.exists(results_path):
                all_files = os.listdir(results_path)
                print(f"目录中的文件: {all_files}")
            else:
                print(f"结果目录不存在")
            continue
        
        print(f"找到的结果文件: {result_files}")
        
        # 查找Input_*文件
        input_files = [f for f in result_files if os.path.basename(f).startswith('Input_')]
        if not input_files:
            print("未找到输入文件，尝试使用其他匹配模式")
            # 尝试其他命名模式
            input_files = [f for f in result_files if 'input' in os.path.basename(f).lower()]
        
        if not input_files:
            print("仍未找到输入文件，使用第一个找到的结果文件")
            if result_files:
                input_file = result_files[0]
                # 猜测输出文件名
                output_file = input_file.replace('Input', 'Out')
                if not os.path.exists(output_file):
                    # 如果没有找到对应的Out文件，查找任何可能的输出文件
                    output_candidates = [f for f in result_files if f != input_file]
                    if output_candidates:
                        output_file = output_candidates[0]
                    else:
                        print("无法找到输出文件，跳过此类别")
                        continue
            else:
                print("没有找到任何结果文件，跳过此类别")
                continue
        else:
            input_file = input_files[0]
            output_file = input_file.replace('Input', 'Out')
            if not os.path.exists(output_file):
                # 尝试找到任何包含'out'的文件
                out_files = [f for f in result_files if 'out' in os.path.basename(f).lower()]
                if out_files:
                    output_file = out_files[0]
                else:
                    print("无法找到输出文件，跳过此类别")
                    continue
        
        print(f"使用输入文件: {input_file}")
        print(f"使用输出文件: {output_file}")
        
        # 读取输入和输出图像
        try:
            # 尝试使用tifffile读取
            try:
                noisy_img = imread(input_file)
            except:
                # 如果失败，使用OpenCV读取
                noisy_img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
            
            try:
                denoised_img = imread(output_file)
            except:
                # 如果失败，使用OpenCV读取
                denoised_img = cv2.imread(output_file, cv2.IMREAD_GRAYSCALE)
            
            # 归一化
            if noisy_img.dtype != np.float32:
                noisy_img = noisy_img.astype(np.float32) / 255.0
            if denoised_img.dtype != np.float32:
                denoised_img = denoised_img.astype(np.float32) / 255.0
            
            # 确保值在[0, 1]范围内
            noisy_img = np.clip(noisy_img, 0, 1)
            denoised_img = np.clip(denoised_img, 0, 1)
        except Exception as e:
            print(f"读取图像时出错: {e}")
            continue
        
        # 加载高质量图像进行评估
        high_quality_img = load_and_preprocess_image(high_quality_image)
        
        # 确保图像尺寸一致
        if denoised_img.shape != high_quality_img.shape:
            denoised_img = cv2.resize(denoised_img, (high_quality_img.shape[1], high_quality_img.shape[0]))
        
        # 计算评估指标
        psnr_value = psnr(high_quality_img, denoised_img, data_range=1.0)
        ssim_value = ssim(high_quality_img, denoised_img, data_range=1.0)
        mse_value = calculate_mse(high_quality_img, denoised_img)
        lncc_value = calculate_lncc(high_quality_img, denoised_img)
        
        # 存储结果
        results = {
            'PSNR': psnr_value,
            'SSIM': ssim_value,
            'MSE': mse_value,
            'LNCC': lncc_value
        }
        
        category_results[category] = results
        all_results.update({f"{category}_{k}": v for k, v in results.items()})
        
        # 打印结果
        print(f"  PSNR: {results['PSNR']:.4f}")
        print(f"  SSIM: {results['SSIM']:.4f}")
        print(f"  MSE: {results['MSE']:.8f}")
        print(f"  LNCC: {results['LNCC']:.4f}")
        
        # 在vis文件夹中保存三种图片
        category_vis_dir = os.path.join(vis_dir, category)
        os.makedirs(category_vis_dir, exist_ok=True)
        
        # 保存低质量图片
        noisy_save_path = os.path.join(category_vis_dir, f"{image_name.split('.')[0]}_low_quality.png")
        noisy_img_uint8 = (noisy_img * 255).astype(np.uint8)
        cv2.imwrite(noisy_save_path, noisy_img_uint8)
        
        # 保存高质量图片
        high_quality_save_path = os.path.join(category_vis_dir, f"{image_name.split('.')[0]}_high_quality.png")
        high_quality_img_uint8 = (high_quality_img * 255).astype(np.uint8)
        cv2.imwrite(high_quality_save_path, high_quality_img_uint8)
        
        # 保存模型输出结果图片
        denoised_save_path = os.path.join(category_vis_dir, f"{image_name.split('.')[0]}_denoised.png")
        denoised_img_uint8 = (denoised_img * 255).astype(np.uint8)
        cv2.imwrite(denoised_save_path, denoised_img_uint8)
        
        # 创建可视化结果
        # 确保目录存在
        category_save_dir = os.path.join(results_dir, category)
        os.makedirs(category_save_dir, exist_ok=True)
        
        # 创建可视化
        plt.figure(figsize=(14, 7))
        
        # 显示原始噪声图像
        plt.subplot(1, 3, 1)
        plt.imshow(noisy_img, cmap='gray')
        plt.title('低质量')
        
        # 显示去噪后的图像
        plt.subplot(1, 3, 2)
        plt.imshow(denoised_img, cmap='gray')
        plt.title('去噪后')
        
        # 显示高质量参考图像
        plt.subplot(1, 3, 3)
        plt.imshow(high_quality_img, cmap='gray')
        plt.title('高质量参考')
        
        # 保存可视化结果
        plt.tight_layout()
        plt.savefig(os.path.join(category_save_dir, f"{image_name.split('.')[0]}_comparison.png"))
        plt.close()
        
        # 保存去噪结果
        denoised_save_path = os.path.join(category_save_dir, image_name)
        denoised_img_uint8 = (denoised_img * 255).astype(np.uint8)
        cv2.imwrite(denoised_save_path, denoised_img_uint8)
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        print()
    
    # 检查是否有结果
    if not category_results:
        print("警告：没有成功处理任何图像")
        return None
    
    # 计算平均指标
    avg_psnr = np.mean([results['PSNR'] for results in category_results.values()])
    avg_ssim = np.mean([results['SSIM'] for results in category_results.values()])
    avg_mse = np.mean([results['MSE'] for results in category_results.values()])
    avg_lncc = np.mean([results['LNCC'] for results in category_results.values()])
    
    # 打印总体平均结果
    print("=" * 50)
    print("平均指标结果:")
    print(f"  平均 PSNR: {avg_psnr:.4f}")
    print(f"  平均 SSIM: {avg_ssim:.4f}")
    print(f"  平均 MSE: {avg_mse:.8f}")
    print(f"  平均 LNCC: {avg_lncc:.4f}")
    
    # 将结果保存到文件
    result_file = os.path.join(results_dir, "evaluation_results.txt")
    with open(result_file, 'w') as f:
        f.write("类别评估结果:\n")
        for category, results in category_results.items():
            f.write(f"{category}:\n")
            f.write(f"  PSNR: {results['PSNR']:.4f}\n")
            f.write(f"  SSIM: {results['SSIM']:.4f}\n")
            f.write(f"  MSE: {results['MSE']:.8f}\n")
            f.write(f"  LNCC: {results['LNCC']:.4f}\n\n")
        
        f.write("平均指标结果:\n")
        f.write(f"  平均 PSNR: {avg_psnr:.4f}\n")
        f.write(f"  平均 SSIM: {avg_ssim:.4f}\n")
        f.write(f"  平均 MSE: {avg_mse:.8f}\n")
        f.write(f"  平均 LNCC: {avg_lncc:.4f}\n")
    
    print(f"评估结果已保存到 {result_file}")
    
    return all_results

if __name__ == "__main__":
    main() 