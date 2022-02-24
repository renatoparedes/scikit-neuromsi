#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# IMPORTS
# =============================================================================

from skneuromsi import core


def test_method():
    class Metodo(core.SKNMSIMethodABC):

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
    method.run("hola", vis_position="mundo")
    # import ipdb; ipdb.set_trace()
