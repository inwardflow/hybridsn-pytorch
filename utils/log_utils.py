# -*- coding: utf-8 -*-
# @Author: 辰阳星宇
# @Date:   2024-01-24
# @Description: 日志工具模块，用于记录日志和控制台输出重定向

import logging
import sys
from datetime import datetime


class Logger:
    """ 使用logging模块创建logger对象，记录由logger输出的日志信息
    """

    def __init__(self, LoggerName, FileName, CmdLevel, FileLevel):
        # LoggerName：实例化对象的名字  FileName:外部文件名   CmdLevel:设置控制台中日志输出的级别  FileLevel:设置文件日志输出的级别
        self.logger = logging.getLogger(LoggerName)
        # 设置日志的级别
        self.logger.setLevel(logging.DEBUG)
        # 设置日志的输出格式
        fmt = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')

        # 借助handle将日志输出到test.log文件中
        fh = logging.FileHandler(FileName, encoding='utf-8')
        fh.setLevel(FileLevel)

        # 借助handle将日志输出到控制台
        ch = logging.StreamHandler()
        # ch.setLevel(CmdLevel)

        # 配置logger
        fh.setFormatter(fmt)
        # ch.setFormatter(fmt)

        # 给logger添加handle
        self.logger.addHandler(fh)
        # self.logger.addHandler(ch)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def close(self):
        self.logger.disabled = True


class TerminalLogger(object):
    """ 将控制台上输出重定向，将控制台内容输入到log_path文件内
    """

    def __init__(self, log_path, stream=sys.stdout):
        self.terminal = stream
        self.log = open(log_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()  # 不启动缓冲,实时输出
        self.log.flush()

    def flush(self):
        pass


def record_terminal(log_path):
    """ 调用TerminalLogger的方法

    Args:
        log_path: log文件路径

    Returns:

    """
    # 记录正常的 print 信息
    sys.stdout = TerminalLogger(log_path, sys.stdout)
    # 记录 traceback 异常信息
    sys.stderr = TerminalLogger(log_path, sys.stderr)


def record_timestamp():
    """ 生成当前时间

    Returns: 当前时间

    """
    now = datetime.now()
    return now


if __name__ == '__main__':
    # 方式一：调用控制台重定向将控制台中信息记录到指定路径文件中
    record_terminal('../log/output_{}.log'.format(record_timestamp().date()))
    # 方式二：调用Logger对象，使用logging模块记录日志
    logger = Logger("my_log", "../log/output_{}.log".format(record_timestamp().date()), CmdLevel=logging.DEBUG,
                    FileLevel=logging.INFO)

    # 将时间信息输出到控制台，让日志记录中有时间信息
    new = record_timestamp()
