# coding=utf-8
"""
Modbus RTU 从站服务器模块

这个模块实现了一个Modbus RTU从站服务器，主要功能：
1. 从MySQL数据库读取点位配置信息
2. 作为Modbus RTU从站接收主站数据
3. 解析接收到的数据并存储到数据库

使用pymodbus库实现Modbus RTU从站功能，替代原来的modbus_tk库。
添加了完善的错误处理和日志记录功能。
"""

import datetime
import os
import struct
import time
import logging
import threading
from typing import List, Tuple

import pymysql
import requests
from pymodbus.server.sync import StartSerialServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
from pymodbus.transaction import ModbusRtuFramer

# 浮点数解析方式常量
FLOAT_AB_CD = 1
FLOAT_CD_AB = 2
FLOAT_BA_DC = 3
FLOAT_DC_BA = 4

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModbusRtuSlave:
    """Modbus RTU 从站服务器类"""

    def __init__(self):
        """初始化Modbus RTU从站服务器"""
        self.serial_port = None
        self.mysql_addr = None
        self.mysql_user = None
        self.mysql_password = None
        self.mysql_db = None
        self.server = None
        self.data_block = None
        self.context = None
        self.identity = None
        self.point_list = []
        self.last_update_time = 0
        self.update_interval = 30  # 点位列表更新间隔(秒)
        self.lock = threading.Lock()

    def read_config_from_system_env(self):
        """从系统环境变量读取配置"""
        try:
            self.mysql_addr = os.environ.get('MYSQL_ADDR')
            if self.mysql_addr is None:
                raise ValueError("无法找到 MYSQL_ADDR 环境变量")

            self.mysql_user = os.environ.get('MYSQL_USER')
            if self.mysql_user is None:
                raise ValueError("无法找到 MYSQL_USER 环境变量")

            self.mysql_password = os.environ.get('MYSQL_PASSWORD')
            if self.mysql_password is None:
                raise ValueError("无法找到 MYSQL_PASSWORD 环境变量")

            self.mysql_db = os.environ.get('MYSQL_DB')
            if self.mysql_db is None:
                raise ValueError("无法找到 MYSQL_DB 环境变量")

            self.serial_port = os.environ.get('SERIAL_PORT')
            if self.serial_port is None:
                raise ValueError("无法找到 SERIAL_PORT 环境变量")

            logger.info("成功从环境变量读取配置")

        except Exception as e:
            logger.error(f"读取环境变量配置失败: {e}")
            raise

    def read_config_from_local_test(self):
        """本地测试配置（用于开发和测试）"""
        self.mysql_addr = "192.168.50.11"
        self.mysql_user = "root"
        self.mysql_password = "z1Tx!6gHx40Gtaru"
        self.mysql_db = "modbus_server"
        self.serial_port = "COM3"
        logger.info("使用本地测试配置")

    def _get_point_list(self) -> List[Tuple]:
        """从数据库获取点位列表"""
        try:
            db = pymysql.connect(
                user=self.mysql_user,
                password=self.mysql_password,
                host=self.mysql_addr,
                db=self.mysql_db
            )

            with db.cursor() as cursor:
                sql = 'SELECT address, point_name, point_info FROM modbus_point WHERE crew_name = 2'
                cursor.execute(sql)
                point_list = list(cursor.fetchall())

            db.close()
            logger.info(f"成功从数据库获取 {len(point_list)} 个点位")
            return point_list

        except pymysql.Error as e:
            logger.error(f"数据库查询失败: {e}")
            return []
        except Exception as e:
            logger.error(f"获取点位列表失败: {e}")
            return []

    def _parse_data(self, a1: int, a2: int, parse_code: int) -> float:
        """解析两个寄存器值为浮点数"""
        try:
            if a1 != 0 or a2 != 0:
                if parse_code == FLOAT_DC_BA:
                    bin1 = bin(a1)[2:].zfill(16)
                    bin2 = bin(a2)[2:].zfill(16)
                    bin_ = bin2[8:12] + bin2[12:16] + bin2[0:4] + bin2[4:8] + (
                        bin1[8:12] + bin1[12:16] + bin1[0:4] + bin1[4:8])
                else:  # FLOAT_AB_CD
                    bin_ = bin(a1)[2:].zfill(16) + bin(a2)[2:].zfill(16)

                hex_ = hex(int(str(bin_), 2)).replace("0x", "").replace("L", "")
                return struct.unpack("!f", bytes.fromhex(hex_))[0]
            else:
                return 0.0

        except Exception as e:
            logger.error(f"解析数据失败: a1={a1}, a2={a2}, parse_code={parse_code}, 错误: {e}")
            return 0.0

    def _update_point_values(self):
        """更新点位值到数据库"""
        try:
            current_time = time.time()

            # 定期更新点位列表
            if current_time - self.last_update_time > self.update_interval:
                with self.lock:
                    self.point_list = self._get_point_list()
                self.last_update_time = current_time

            # 读取保持寄存器值
            slave_id = 0x01  # 从站ID
            start_address = 85  # 起始地址
            register_count = 92  # 寄存器数量

            # 获取从站上下文
            slave_context = self.context[slave_id]
            values = slave_context.getValues(3, start_address, register_count)  # 3=保持寄存器

            # 处理数据
            if len(values) >= 5:
                # 从下标为5开始取数据
                data_to_process = values[5:]
                date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 准备插入数据库的值
                value_records = []
                for i in range(0, len(data_to_process), 2):
                    if i + 1 >= len(data_to_process):
                        break

                    address = i
                    value = self._parse_data(data_to_process[i], data_to_process[i + 1], FLOAT_AB_CD)

                    # 查找对应的点位名称
                    point_name = None
                    with self.lock:
                        for point in self.point_list:
                            if point[0] == address:
                                point_name = point[1]
                                break

                    if point_name:
                        value_records.append(f"('{date}','{point_name}',{value})")

                # 如果有数据，插入数据库
                if value_records:
                    values_str = ",".join(value_records)
                    query = f"INSERT INTO yl.data (`dateTime`, pointName, pointValue) VALUES {values_str};"

                    try:
                        response = requests.post(
                            'http://127.0.0.1:8123',
                            data={"query": query},
                            timeout=5
                        )
                        logger.info(f"数据插入成功: {response.text}")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"HTTP请求失败: {e}")

        except Exception as e:
            logger.error(f"更新点位值失败: {e}")

    def setup_server(self):
        """设置Modbus服务器"""
        try:
            # 初始化数据存储
            # 创建保持寄存器块，初始值为0，数量为100
            self.data_block = ModbusSequentialDataBlock(0, [0] * 100)
            self.context = ModbusSlaveContext(hr=self.data_block)
            self.context = ModbusServerContext(slaves={1: self.context}, single=False)

            # 初始化设备标识
            self.identity = ModbusDeviceIdentification()
            self.identity.VendorName = 'Pymodbus'
            self.identity.ProductCode = 'PM'
            self.identity.VendorUrl = 'http://github.com/bashwork/pymodbus/'
            self.identity.ProductName = 'Pymodbus Server'
            self.identity.ModelName = 'Pymodbus Server'
            self.identity.MajorMinorRevision = '2.5.0'

            logger.info("Modbus服务器设置完成")

        except Exception as e:
            logger.error(f"设置Modbus服务器失败: {e}")
            raise

    def run(self):
        """运行Modbus RTU从站服务器"""
        try:
            # 读取配置
            try:
                self.read_config_from_system_env()
            except Exception as e:
                logger.warning(f"使用环境变量配置失败，尝试使用本地测试配置: {e}")
                self.read_config_from_local_test()

            # 获取初始点位列表
            self.point_list = self._get_point_list()
            self.last_update_time = time.time()

            # 设置服务器
            self.setup_server()

            # 启动Modbus服务器
            logger.info(f"启动Modbus RTU从站服务器，串口: {self.serial_port}")

            self.server = StartSerialServer(
                context=self.context,
                identity=self.identity,
                framer=ModbusRtuFramer,
                port=self.serial_port,
                timeout=1,
                baudrate=9600,
                bytesize=8,
                parity='N',
                stopbits=1
            )

        except Exception as e:
            logger.error(f"启动Modbus服务器失败: {e}")
            raise

    def start(self):
        """启动服务器和处理线程"""
        try:
            # 创建并启动服务器线程
            server_thread = threading.Thread(target=self.run, daemon=True)
            server_thread.start()

            logger.info("Modbus服务器线程已启动")

            # 主循环处理数据
            while True:
                try:
                    self._update_point_values()
                    time.sleep(1)  # 每秒处理一次数据
                except KeyboardInterrupt:
                    logger.info("接收到中断信号，停止服务器")
                    break
                except Exception as e:
                    logger.error(f"数据处理循环错误: {e}")
                    time.sleep(5)  # 出错后等待5秒再继续

        except Exception as e:
            logger.error(f"启动服务器失败: {e}")
            raise


if __name__ == '__main__':
    slave = ModbusRtuSlave()
    slave.start()
