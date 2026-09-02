"""动态用户建模与模拟系统

使用示例：
from dytwin import DynamicUserSimulator, DEFAULT_SETTINGS

simulator = DynamicUserSimulator(settings=DEFAULT_SETTINGS, user_id="f33fdf31")
simulator.run(df)
"""

from .simulator import DynamicUserSimulator
from .settings import Settings, DEFAULT_SETTINGS

