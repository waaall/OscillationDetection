import os
import sys
import time
import logging
import json
import argparse
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np

from SignalGenerator import SignalGenerator
from TestOscillationDetection import OscillationDetectionTester


class ODDevFlow:
    def __init__(self, config_file: str = "oscillate_dev_settings.json"):
        """
        振荡检测开发流程主类
        
        Args:
            config_file: 配置文件路径
        """
        # 首先设置基础日志，用于配置加载过程
        self._setup_basic_logger()
        
        # 加载配置
        self.config = self._load_and_validate_config(config_file)
        
        # 从配置文件读取参数
        self.csv_dir = self.config.get('csv_dir', './csv-data')
        self.window_size = self.config.get('window_size', 60)
        self.sampling_rate = self.config.get('sampling_rate', 1.0)
        self.threshold = self.config.get('threshold', 0.5)
        self.log_file = self.config.get('log_file', None)

        # 信号生成相关参数
        self._generate_signal = self.config.get('generate_signal', False)
        self._generate_duration = self.config.get('generate_duration', 10)
        self._noise_level = self.config.get('noise_level', 0.1)
        self._sin_freqs = self.config.get('sin_freqs', [0.1, 0.5])
        self._sin_amps = self.config.get('sin_amps', [1.0, 1.0])
        self._sin_phases = self.config.get('sin_phases', [0.0, 0.0])
        self._linear_slope = self.config.get('linear_slope', 0.2)
        self._linear_intercept = self.config.get('linear_intercept', 1.0)
        self._polynomial_parms = self.config.get('polynomial_parms', [1.0, 0.0, 0.0])
        self._exponential_amps = self.config.get('exponential_amps', 0.0)
        self._exponential_tau = self.config.get('exponential_tau', 0.0)
        
        # 设置完整的日志系统
        self._setup_logger()
        
        # 确保输出目录存在
        os.makedirs(self.csv_dir, exist_ok=True)
        
        self.logger.info("ODDevFlow 初始化完成")
        self.logger.info(f"配置参数: generate_signal={self._generate_signal}, "
                        f"window_size={self.window_size}, sampling_rate={self.sampling_rate}")

    def _setup_basic_logger(self):
        """设置基础日志系统"""
        self.logger = logging.getLogger('ODDevFlow')
        self.logger.setLevel(logging.INFO)
        
        # 清除现有的处理器
        self.logger.handlers.clear()
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 禁用传播到根logger
        self.logger.propagate = False



    def _setup_logger(self):
        """设置完整的日志系统"""
        self.logger = logging.getLogger('ODDevFlow')
        self.logger.setLevel(logging.INFO)
        
        # 清除现有的处理器
        self.logger.handlers.clear()
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 创建文件处理器（如果指定了日志文件）
        if self.log_file:
            try:
                # 确保日志目录存在
                log_dir = os.path.dirname(self.log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                
                file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                file_formatter = logging.Formatter('[%(asctime)s] %(message)s', 
                                                 datefmt='%Y-%m-%d %H:%M:%S')
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
                self.logger.info(f"日志文件设置为: {self.log_file}")
            except Exception as e:
                self.logger.warning(f"无法创建日志文件 {self.log_file}: {e}")
        
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
            print(f"配置文件 {config_file} 不存在")
            print(f"正在创建默认配置文件模板...")
            
            self.save_config_template(config_file)
            
            print(f"==========================================")
            print(f"已自动生成配置文件: {config_file}")
            print(f"请编辑该文件设置您的参数:")
            print(f"- generate_signal: 是否生成测试信号")
            print(f"- csv_dir: 数据文件目录")
            print(f"- window_size, sampling_rate, threshold: 检测参数")
            print(f"编辑完成后重新运行程序")
            print(f"==========================================")
            sys.exit(0)
        
        # 加载配置文件
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read().strip()
                if not config_content:
                    self.logger.error(f"[ERROR] 配置文件 {config_file} 为空")
                    sys.exit(1)
                
                config = json.loads(config_content)
                print(f"[CONFIG] 已加载配置文件: {config_file}")
                return config
                
        except json.JSONDecodeError as e:
            print(f"[ERROR] 配置文件 {config_file} 格式错误: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] 读取配置文件 {config_file} 失败: {e}")
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
            "window_size": 60,
            "sampling_rate": 1.0,
            "threshold": 0.5,
            "log_file": None,
            "generate_signal": False,
            "generate_duration": 10,
            "noise_level": 0.1,
            "sin_freqs": [0.1, 0.5],
            "sin_amps": [1.0, 1.0],
            "sin_phases": [0.0, 0.0],
            "linear_slope": 0.2,
            "linear_intercept": 1.0,
            "polynomial_parms": [1.0, 0.0, 0.0],
            "exponential_amps": 0.0,
            "exponential_tau": 0.0,
            "description": "test oscillation detection settings",
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(template_config, f, ensure_ascii=False, indent=4)
            print(f"[SUCCESS] 配置文件模板已保存: {config_file}")
        except Exception as e:
            print(f"[ERROR] 保存配置文件模板失败: {e}")

    def generate_test_signal(self) -> str:
        """
        根据配置参数生成测试信号
        
        Returns:
            str: 生成的CSV文件路径
        """
        try:
            self.logger.info("开始生成测试信号...")
            
            # 创建信号生成器
            generator = SignalGenerator(
                sampling_rate=int(self.sampling_rate),
                duration=self._generate_duration,
                noise_level=self._noise_level,
                seed=42,  # 固定种子保证可重复性
                log_file=self.log_file
            )
            
            # 生成基础信号（从0开始）
            signal = np.zeros(len(generator.signal_data))
            
            # 添加正弦波分量
            if self._sin_freqs and any(f > 0 for f in self._sin_freqs):
                sin_signal = generator.sine_wave(
                    freqs=self._sin_freqs,
                    amps=self._sin_amps,
                    phases=self._sin_phases
                )
                signal += sin_signal
                self.logger.info(f"已添加正弦波分量: freqs={self._sin_freqs}")
            
            # 添加线性分量
            if self._linear_slope != 0 or self._linear_intercept != 0:
                linear_signal = generator.linear(
                    slope=self._linear_slope,
                    intercept=self._linear_intercept
                )
                signal += linear_signal
                self.logger.info(f"已添加线性分量: slope={self._linear_slope}, intercept={self._linear_intercept}")
            
            # 添加多项式分量
            if self._polynomial_parms and any(p != 0 for p in self._polynomial_parms):
                poly_signal = generator.polynomial(coeffs=self._polynomial_parms)
                signal += poly_signal
                self.logger.info(f"已添加多项式分量: coeffs={self._polynomial_parms}")
            
            # 添加指数分量
            if self._exponential_amps != 0 and self._exponential_tau != 0:
                exp_signal = generator.exponential(
                    A=self._exponential_amps,
                    tau=self._exponential_tau
                )
                signal += exp_signal
                self.logger.info(f"已添加指数分量: A={self._exponential_amps}, tau={self._exponential_tau}")
            
            # 保存到CSV文件
            csv_file = os.path.join(self.csv_dir, "generated_signal.csv")
            generator.insert_into_csv(
                csv_path=csv_file,
                column="值",
                start_idx=0,
                new_signal=signal,
                isCUT=True
            )
            
            self.logger.info(f"测试信号已生成并保存到: {csv_file}")
            self.logger.info(f"信号长度: {len(signal)} 点, 持续时间: {self._generate_duration} 秒")
            
            return csv_file
            
        except Exception as e:
            self.logger.error(f"生成测试信号失败: {e}")
            raise

    def run_oscillation_detection(self, csv_file: str, mode: str = "animation") -> Optional[List[Dict]]:
        """
        运行振荡检测测试
        
        Args:
            csv_file: CSV文件路径
            mode: 运行模式 ("animation" 或 "static")
            
        Returns:
            Optional[List[Dict]]: 静态模式下返回检测结果，动画模式下返回None
        """
        try:
            self.logger.info(f"开始振荡检测测试: {csv_file}")
            self.logger.info(f"检测参数: window_size={self.window_size}, "
                           f"sampling_rate={self.sampling_rate}, threshold={self.threshold}")
            
            # 创建检测器
            tester = OscillationDetectionTester(
                csv_file=csv_file,
                window_size=self.window_size,
                overlap_ratio=0.5,  # 50%重叠
                sampling_rate=self.sampling_rate,
                threshold=self.threshold,
                col_name="值",
                log_file=self.log_file
            )
            
            if mode == "animation":
                self.logger.info("启动动画模式...")
                tester.run(interval=200)
                return None
            elif mode == "static":
                self.logger.info("启动静态分析模式...")
                results = tester.analyze_static(start_window=0, num_windows=50)
                oscillation_count = sum(1 for r in results if r['is_oscillation'])
                self.logger.info(f"静态分析完成，共分析 {len(results)} 个窗口，"
                               f"检测到振荡的窗口数: {oscillation_count}")
                return results
            else:
                raise ValueError(f"不支持的运行模式: {mode}")
                
        except Exception as e:
            self.logger.error(f"振荡检测测试失败: {e}")
            raise

    def run(self, mode: str = "animation"):
        """
        运行完整的开发流程
        
        Args:
            mode: 运行模式 ("animation" 或 "static")
        """
        try:
            self.logger.info("="*50)
            self.logger.info("开始振荡检测开发流程")
            self.logger.info("="*50)
            
            # 步骤1: 决定使用的CSV文件
            if self._generate_signal:
                self.logger.info("配置指定生成测试信号")
                csv_file = self.generate_test_signal()
            else:
                self.logger.info("配置指定使用现有数据文件")
                csv_file = os.path.join(self.csv_dir, "data.csv")
                
                # 检查文件是否存在
                if not os.path.exists(csv_file):
                    available_files = [f for f in os.listdir(self.csv_dir) 
                                     if f.endswith('.csv')] if os.path.exists(self.csv_dir) else []
                    if available_files:
                        csv_file = os.path.join(self.csv_dir, available_files[0])
                        self.logger.info(f"data.csv不存在，使用: {csv_file}")
                    else:
                        raise FileNotFoundError(f"在 {self.csv_dir} 目录中找不到CSV数据文件")
            
            # 步骤2: 运行振荡检测测试
            results = self.run_oscillation_detection(csv_file, mode=mode)
            
            if results is not None:
                # 输出静态分析结果
                self.logger.info("="*30 + " 检测结果 " + "="*30)
                for result in results[:10]:  # 显示前10个结果
                    status = "振荡" if result['is_oscillation'] else "正常"
                    if result['is_oscillation']:
                        self.logger.info(f"窗口 {result['window']:3d}: {status} - "
                                       f"频率={result['peak_frequency']:.2f}Hz, "
                                       f"幅值={result['peak_amplitude']:.3f}")
                    else:
                        self.logger.info(f"窗口 {result['window']:3d}: {status}")
            
            self.logger.info("="*50)
            self.logger.info("振荡检测开发流程完成")
            self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error(f"运行流程失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="振荡检测开发流程工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                          # 使用默认配置文件
  python main.py --config my_config.json  # 指定配置文件
  python main.py --mode static             # 静态分析模式
  python main.py --mode animation          # 动画演示模式（默认）
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='oscillate_dev_settings.json',
        help='配置文件路径 (默认: oscillate_dev_settings.json)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['animation', 'static'],
        default='animation',
        help='运行模式: animation(动画演示) 或 static(静态分析) (默认: animation)'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='创建配置文件模板并退出'
    )
    
    args = parser.parse_args()
    
    # 如果用户只想创建配置文件模板
    if args.create_config:
        ODDevFlow.save_config_template(args.config)
        return
    
    try:
        # 创建并运行开发流程
        dev_flow = ODDevFlow(config_file=args.config)
        dev_flow.run(mode=args.mode)
        
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断程序")
    except Exception as e:
        print(f"[ERROR] 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
