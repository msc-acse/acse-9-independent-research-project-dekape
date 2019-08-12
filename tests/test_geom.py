#!/usr/bin/env python
# Deborah Pelacani Cruz
# https://github.com/dekape
import context
import fullwaveqc.geom as geom
import os
import numpy as np


def test_thisfunction():
    assert 1


def test_boundarycalc():
    nabs, ndist = geom.boundarycalc(1190, 25, 1.5, 3400)
    assert(nabs == 182 and ndist == 33)
    return


def test_surveygeom():
    dir_path = os.path.abspath(os.path.dirname(__file__))
    rcvgeopath = os.path.join(dir_path, "test_data/PARBASE25_7-Receivers.geo")
    srcgeopath = os.path.join(dir_path, "test_data/PARBASE25_7-Sources.geo")
    src_list = [0, 200, 400, -1]
    g = geom.surveygeom(rcvgeopath, srcgeopath, src_list, plot=False, verbose=1)

    path = os.path.join(dir_path, "test_data/surveygeom_test.npy")
    g2 = np.load(path, allow_pickle=True)

    for i in range(0, len(g[0])):
        assert(np.allclose(g[0][i], g2[0][i]))

    for i in range(0, len(g[1])):
        assert(np.allclose(g[1][i], g2[1][i]))

    return