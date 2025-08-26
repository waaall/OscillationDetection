import os
import sys
import time
import logging
import json
import argparse
from typing import List, Tuple, Dict
from pathlib import Path
from SignalGenerator import SignalGenerator
from OscillationDetectionTester import OscillationDetectionTester


class ODDevFlow:
    def __init__(self, config_file: str = "oscillate_dev_settings.json"):
    
        config = self._load_and_validate_config(config_file)
        
        # 从配置文件读取参数
        self.csv_dir = config.get('csv_dir', './csv-data')
        self.output_dir = config.get('output_dir', './csv-output')
        self.target_points = config.get('target_points', [])
        self.days = config.get('days', 1)
        self.start_date = config.get('start_date', None)
        self.max_workers = config.get('max_workers', 4)
        self.point_threads = config.get('point_threads', 4)
        self.csv_format = config.get('csv_format', 'detailed')
        self.save_individual_points = config.get('save_individual_points', False)



    def _setup_logger(self):
        """设置高性能日志系统"""
        self.logger = logging.getLogger('HisToCsv')
        self.logger.setLevel(logging.INFO)
        
        # 清除现有的处理器
        self.logger.handlers.clear()
        
        # 创建文件处理器, 使用缓冲模式提高性能
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        file_formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 禁用传播到根logger
        self.logger.propagate = False

    def _load_and_validate_config(self, config_file: str) -> dict:
        """
        加载和验证配置文件, 如果不存在则自动创建模板
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            dict: 配置字典
        """
        config_path = Path(config_file)
        
        # 如果配置文件不存在, 创建模板并提示用户
        if not config_path.exists():
            self.logger.info(f"配置文件 {config_file} 不存在")
            self.logger.info(f"正在创建默认配置文件模板...")
            
            self.save_config_template(config_file)
            
            self.logger.info(f"==========================================")
            self.logger.info(f"已自动生成配置文件: {config_file}")
            self.logger.info(f"请编辑该文件设置您的参数:")
            self.logger.info(f"- csv_dir: 数据文件目录")
            self.logger.info(f"编辑完成后重新运行程序")
            self.logger.info(f"==========================================")
            sys.exit(0)
        
        # 加载配置文件
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read().strip()
                if not config_content:
                    self.logger.error(f"[ERROR] 配置文件 {config_file} 为空")
                    sys.exit(1)
                
                config = json.loads(config_content)
                self.logger.info(f"[CONFIG] 已加载配置文件: {config_file}")
                return config
                
        except json.JSONDecodeError as e:
            self.logger.error(f"[ERROR] 配置文件 {config_file} 格式错误: {e}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"[ERROR] 读取配置文件 {config_file} 失败: {e}")
            sys.exit(1)

    @staticmethod
    def save_config_template(config_file: str = "oscillate_dev_settings.json") -> None:
        """
        保存完整的配置文件模板
        
        Args:
            config_file: 配置文件路径
        """
        template_config = {
            "csv_dir": "./csv-data",
            "SignalGenerator": False,
            "output_dir": "./csv-output",
            "target_points": [
                "SYS_XCU001_Memory",
                "20MCS-UNITMW"
            ],
            "days": 1,
            "start_date": "20250702",
            "max_workers": 4,
            "point_threads": 4,
            "csv_format": "detailed",
            
            "description": "HIS文件批量CSV导出配置文件",
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(template_config, f, ensure_ascii=False, indent=4)
            print(f"[SUCCESS] 配置文件模板已保存: {config_file}")
        except Exception as e:
            print(f"[ERROR] 保存配置文件模板失败: {e}")