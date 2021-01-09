from abc import ABCMeta, abstractmethod


class _BaseSpG(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, gp_handler):
        self._gp_handler = gp_handler

    @abstractmethod
    def generate(self, n):
        pass
    
    @abstractmethod
    def _generate_spline(self, ego_pose, target):
        pass