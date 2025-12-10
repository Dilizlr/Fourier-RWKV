import os
import yaml
import time
import torch
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning)



from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.append('.')
from utils import Adder
from custom_utils.data_loaders.dehaze_datasets import DeblurDataset
from model.model_builder import DehazeNet
import custom_utils

def parse_test_args():
    parser = argparse.ArgumentParser(description='FFTRWKV - Testing')
    parser.add_argument('--mode', type=str, default='ITS', help='Model type')
    parser.add_argument('--channel', type=int, default=24, help='Channel size')
    parser.add_argument('--model_path', type=str, default='./checkpoints/ITS_bestPSNR.pth')
    parser.add_argument('--yml_path', type=str, default='./configs/ITS_hazy.yaml', help='Path to config YAML file')
    parser.add_argument('--save_dir', type=str, default='./test_results/', help='Directory to save test results')
    parser.add_argument('--save_images', action='store_true', default=True, help='Whether to save output images')
    parser.add_argument('--save_metrics', action='store_true', default=True, help='Whether to save per-image metrics')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')
    return parser.parse_args()

def save_metrics_to_txt(metrics_data, summary_stats, txt_path):
    """保存指标数据到TXT文件"""
    with open(txt_path, 'w') as f:
        # 写入测试信息
        f.write("=" * 60 + "\n")
        f.write("TEST RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {summary_stats['test_info']['model_path']}\n")
        f.write(f"Test dataset: {summary_stats['test_info']['test_dataset']}\n")
        f.write(f"Timestamp: {summary_stats['test_info']['timestamp']}\n")
        f.write(f"Total images: {summary_stats['test_info']['total_images']}\n\n")
        
        # 写入汇总指标
        f.write("SUMMARY METRICS:\n")
        f.write("-" * 40 + "\n")
        metrics = summary_stats['summary_metrics']
        f.write(f"Average PSNR: {metrics['average_psnr']:.4f} dB\n")
        f.write(f"Average SSIM: {metrics['average_ssim']:.4f}\n")
        f.write(f"Total testing time: {metrics['total_testing_time']:.2f} s\n\n")
        
        # 写入每张图像的详细指标
        f.write("PER-IMAGE METRICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Image Name':<30} {'PSNR (dB)':<12} {'SSIM':<12}\n")
        f.write("-" * 60 + "\n")
        
        for data in metrics_data:
            f.write(f"{data['image_name']:<30} {data['psnr']:<12.4f} {data['ssim']:<12.4f}\n")
    
    print(f"Metrics saved to: {txt_path}")

def test_model():
    args = parse_test_args()
    
    # 设备配置
    if args.gpu is not None and args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    # 加载配置
    with open(args.yml_path, 'r') as f:
        config = yaml.safe_load(f)
    Test = config['TESTING']

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_images:
        output_dir = os.path.join(args.save_dir, args.mode)
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载模型
    print("Loading model...")
    model = DehazeNet(dim=args.channel).to(device)
    
    # 加载训练好的权重
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # 处理可能的DDP前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully!")
    
    # 2. 准备测试数据集
    print("Preparing test dataset...")
    test_dataset = DeblurDataset(Test['TEST_DIR'], mode=args.mode, is_test=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 3. 测试循环
    print("Starting testing...")
    psnr_adder = Adder()
    ssim_adder = Adder()
    
    # 存储每张图像的指标
    per_image_metrics = []
    
    # 记录总开始时间
    total_start_time = time.time()
    
    with torch.no_grad():
        for idx, (inp, tar, image_name) in enumerate(tqdm(test_loader, desc="Testing")):
            inp, tar = inp.to(device), tar.to(device)
            
            # 填充到8的倍数（与训练时保持一致）
            h, w = inp.shape[2], inp.shape[3]
            factor = 8
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
    
            if padh > 0 or padw > 0:
                inp_pad = F.pad(inp, (0, padw, 0, padh), 'reflect')
                out_pad = model(inp_pad)
                out = out_pad[:, :, :h, :w]
            else:
                out = model(inp)
            
            # 限制输出范围
            out = torch.clamp(out, 0, 1)
            
            # 计算指标
            psnr_val = custom_utils.torchPSNR(out, tar)
            ssim_val = custom_utils.torchSSIM(out, tar, H, W)
            
            psnr_adder(psnr_val)
            ssim_adder(ssim_val)
            
            # 保存每张图像的指标
            per_image_metrics.append({
                'image_name': image_name[0],
                'psnr': float(psnr_val),
                'ssim': float(ssim_val)
            })
            
            # 保存输出图像
            if args.save_images:
                from torchvision.utils import save_image
                # 确保文件名安全
                safe_name = os.path.basename(image_name[0])
                if not safe_name:  # 如果名称为空
                    safe_name = f"image_{idx:04d}.png"
                elif not safe_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    safe_name += '.png'
                    
                save_path = os.path.join(output_dir, safe_name)
                save_image(out, save_path)
    
    # 计算总时间
    total_time = time.time() - total_start_time
    
    # 4. 输出结果到控制台
    final_psnr = psnr_adder.average()
    final_ssim = ssim_adder.average()
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Test dataset: {Test['TEST_DIR']}")
    print(f"Model: {args.model_path}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Average PSNR: {final_psnr:.4f} dB")
    print(f"Average SSIM: {final_ssim:.4f}")
    print(f"Total testing time: {total_time:.2f} seconds")
    print("="*60)
    
    # 5. 保存每张图像的指标到TXT文件
    if args.save_metrics and per_image_metrics:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备汇总统计信息
        summary_stats = {
            'test_info': {
                'model_path': args.model_path,
                'test_dataset': Test['TEST_DIR'],
                'timestamp': timestamp,
                'total_images': len(per_image_metrics)
            },
            'summary_metrics': {
                'average_psnr': float(final_psnr),
                'average_ssim': float(final_ssim),
                'total_testing_time': float(total_time)
            }
        }
        
        # 保存到TXT文件
        txt_path = os.path.join(args.save_dir, f'test_metrics_{args.mode}.txt')
        save_metrics_to_txt(per_image_metrics, summary_stats, txt_path)
    
    if args.save_images:
        print(f"Output images saved to: {output_dir}")

if __name__ == '__main__':
    test_model()
