# coding=utf-8
import datetime
import os
import struct
import time
import modbus_tk
import modbus_tk.defines as cst
import pymysql
import requests
import serial
from modbus_tk import modbus_rtu

FLOAT_AB_CD = 1
FLOAT_CD_AB = 2
FLOAT_BA_DC = 3
FLOAT_DC_BA = 4

global serial_port
global mysql_addr
global mysql_user
global mysql_password
global mysql_db


def read_config_from_local_test():
    global mysql_addr
    global mysql_user
    global mysql_password
    global mysql_db
    global serial_port
    mysql_addr = "192.168.50.11"
    mysql_user = "root"
    mysql_password = "z1Tx!6gHx40Gtaru"
    mysql_db = "modbus_server"
    serial_port = "COM3"
    print("init mysql_addr/mysql_user...")


# 读取环境变量的值
def read_config_from_system_env():
    global mysql_addr
    global mysql_user
    global mysql_password
    global mysql_db
    global serial_port
    mysql_addr = os.environ.get('MYSQL_ADDR')
    if mysql_addr is not None:
        print("mysql_addr env:", mysql_addr)
    else:
        print("can't find mysql_addr env")
        exit(-1)
    mysql_user = os.environ.get('MYSQL_USER')
    if mysql_user is not None:
        print("mysql_user env:", mysql_user)
    else:
        print("can't find mysql_user env")
        exit(-1)
    mysql_password = os.environ.get('MYSQL_PASSWORD')
    if mysql_password is not None:
        print("mysql_password env:", mysql_password)
    else:
        print("can't find mysql_password env")
        exit(-1)
        exit(-1)
    mysql_db = os.environ.get('MYSQL_DB')
    if mysql_db is not None:
        print("mysql_db env:", mysql_db)
    else:
        print("can't find mysql_db env")
        exit(-1)
    serial_port = os.environ.get('SERIAL_PORT')
    if serial_port is not None:
        print("serial_port env:", serial_port)
    else:
        print("can't find serial_port env")
        exit(-1)


def ModbusRTU_Slave():
    logger = modbus_tk.utils.create_logger("console")
    data = _get_point_list()
    try:
        server = modbus_rtu.RtuServer(
            serial.Serial(
                # port='/dev/ttyS1',
                port=serial_port,
                baudrate=9600,
                bytesize=8,
                parity='N',
                stopbits=1
            )
        )
        server.start()
        SLEVE1 = server.add_slave(1)
        SLEVE1.add_block("1", modbus_tk.defines.READ_HOLDING_REGISTERS, 0, 92)
        lastDate = datetime.datetime.now()
        time.sleep(3)
        tt = 0
        while True:
            time.sleep(0.3)
            tt = tt + 1
            if tt == 30:
                data = _get_point_list()
                tt = 0
            # 获取主站数据,去除第一个 从下标为4的开始拿
            a = list(SLEVE1.get_values("1", modbus_tk.defines.READ_HOLDING_REGISTERS, 85))[5:]
            thisDate = datetime.datetime.now()
            if thisDate.second > lastDate.second or (lastDate.second == 59 and thisDate.second == 0):
                lastDate = thisDate
                date = thisDate.strftime('%Y-%m-%d %H:%M:%S')
                # print date + str(a)
                r = ""
                for i in range(0, len(a), 2):
                    address = i
                    value = _parse_data(a[i], a[i + 1], FLOAT_AB_CD)
                    # r = r + "," + str(value)

                    for k in range(len(data)):
                        if data[k][0] == address:
                            if len(r) != 0:
                                r = r + "," + "('" + date + "'" + ",'" + str(data[k][1]) + "'," \
                                    + str(value) + ")"
                            else:
                                r = "('" + date + "'" + ",'" + str(data[k][1]) + "'," \
                                    + str(value) + ")"
                r = {"query": "INSERT INTO yl.data (`dateTime`, pointName, pointValue) VALUES" + r + ";"}
                print date + " " + str(r)
                print requests.post('http://127.0.0.1:8123', headers={'Content-Type': 'multipart/form-data: boundary'},
                                    data=r).text
    except Exception as err:
        print("modbus_rtu failed:", err)
    finally:
        server.stop()


def _get_point_list():
    # 连接数据库 查询点名列表
    db = pymysql.Connect(
        user=mysql_user,
        password=mysql_password,
        host=mysql_addr,
        db=mysql_db
    )

    cursor = db.cursor()
    sql = 'SELECT address,point_name,point_info FROM modbus_point where crew_name = 2'
    cursor.execute(sql)
    point_list = list(cursor.fetchall())
    db.close()
    return point_list


def _parse_data(a1, a2, parse_code):
    if a1 != 0 or a2 != 0:
        if parse_code == FLOAT_DC_BA:
            bin1 = bin(a1)[2:].zfill(16)
            bin2 = bin(a2)[2:].zfill(16)
            bin_ = bin2[8:12] + bin2[12:16] + bin2[0:4] + bin2[4:8] + (
                    bin1[8:12] + bin1[12:16] + bin1[0:4] + bin1[4:8])
        # elif parse_code == FLOAT_AB_CD:
        else:
            bin_ = bin(a1)[2:].zfill(16) + bin(a2)[2:].zfill(16)
        hex_ = hex(int(str(bin_), 2)).replace("0x", "").replace("L", "")
        return struct.unpack("!f", (hex_ + "").decode('hex'))[0]
    else:
        return 0


if __name__ == '__main__':
    # read_config_from_local_test()
    read_config_from_system_env()
    ModbusRTU_Slave()