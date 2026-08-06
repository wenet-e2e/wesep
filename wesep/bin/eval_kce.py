"""
简化版评估入口脚本
重构后的eval.py，职责更清晰，代码更易维护
"""
import argparse
import os

import yaml

# 使用重构后的模块
from wesep.kce.eval.decoder import Decoder
from wesep.utils.kce_utils import set_seed
from wesep.models.aed_kws_asr_phone import AEDKWSASRPhone


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='KCE Model Evaluation')
    
    parser.add_argument(
        '--checkpoint_path',
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--decoding_config',
        required=True,
        help='Path to decoding configuration file (YAML)'
    )
    
    parser.add_argument(
        '--use_gpu_id',
        type=int,
        default=0,
        help='GPU ID to use (-1 for CPU)'
    )
    
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        help='Distributed rank'
    )
    
    parser.add_argument(
        '--world_size',
        type=int,
        default=1,
        help='Distributed world size'
    )
    
    parser.add_argument(
        '--inference_dirname_tag',
        type=str,
        default='',
        help='Tag for inference result directory'
    )
    
    parser.add_argument(
        '--test_list',
        type=str,
        default='',
        help='Override test list file path'
    )
    
    parser.add_argument(
        '--enable_attention_analysis',
        action='store_true',
        help='Enable attention map analysis'
    )

    parser.add_argument(
        '--extract_embedding',
        action='store_true',
        help='Extract and save speaker embeddings during evaluation'
    )

    parser.add_argument(
        '--embedding_save_dir',
        type=str,
        default=None,
        help='Directory to save extracted embeddings (used with --extract_embedding)'
    )

    return parser.parse_args()


def main():
    """主函数"""
    # 设置随机种子
    set_seed(5526)
    
    # 解析参数
    args = get_args()
    
    # 配置设备
    if args.use_gpu_id < 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        use_cuda = False
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.use_gpu_id)
        use_cuda = True
    
    # 加载解码配置
    with open(args.decoding_config, 'r') as f:
        decoding_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 准备路径
    checkpoint_path = args.checkpoint_path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    inference_dirname_tag = args.inference_dirname_tag or \
        decoding_config['inference_config']['inference_dirname_tag']
    inference_result_dir = os.path.join(checkpoint_dir, 'decode', inference_dirname_tag)
    
    # 测试数据配置
    test_data_config = decoding_config['data_config']
    test_list = args.test_list or decoding_config['test_list']
    
    # 注意力分析配置 (only when --enable_attention_analysis is set)
    attention_config = None
    if args.enable_attention_analysis:
        attention_config = {
            'nth_layers': [9],
            'high_att_threshold': 0.6,
            'min_high_att_frames': 4,
            'window_size_ratio': 3.5,
            'step_ratio': 0.5,
            'repeat_factor': 1,
            'top_percent': 0.05,
        }
    
    # 创建Decoder并运行
    decoder = Decoder(
        model_class=AEDKWSASRPhone,
        checkpoint_path=checkpoint_path,
        inference_result_dir=inference_result_dir,
        test_data_list=test_list,
        test_data_config=test_data_config,
        rank=args.rank,
        world_size=args.world_size,
        use_cuda=use_cuda,
        attention_config=attention_config
    )

    extract_embedding = args.extract_embedding
    embedding_save_dir = args.embedding_save_dir or (
        os.path.join(inference_result_dir, 'embeddings') if extract_embedding else None
    )

    decoder.run(extract_embedding=extract_embedding, embedding_save_dir=embedding_save_dir)
    
    print("Evaluation completed!")


if __name__ == '__main__':
    main()
