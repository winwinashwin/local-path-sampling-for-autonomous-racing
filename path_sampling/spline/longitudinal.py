from ._base import _BaseSpG


class LongitudinalSpG(_BaseSpG):

    def __init__(self, gp_handler):
        super(LongitudinalSpG, self).__init__(gp_handler)

    def generate(self):
        raise Exception('Not implemented yet')
