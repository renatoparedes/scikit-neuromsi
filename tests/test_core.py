import pytest

from prototype import SKNMSIMethodABC


def test_method():
    class Metodo(SKNMSIMethodABC):

        _sknms_abstract = False
        _sknms_run_method_config = [
            {"target": "auditory_position", "template": "${mode0}_position"},
            {"target": "visual_position", "template": "${mode1}_position"},
        ]

        def __init__(self, coso, mode0, mode1="visual"):
            pass

        def run(self, auditory_position, visual_position):
            print(visual_position, auditory_position)

    method = Metodo(1, mode0="aud", mode1="vis")
    import ipdb

    ipdb.set_trace()
    method.run("hola", vis_position="mundo")
    # import ipdb; ipdb.set_trace()
