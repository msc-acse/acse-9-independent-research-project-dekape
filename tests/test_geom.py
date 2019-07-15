import context
import fullwaveqc.geom as geom


def test_thisfunction():
    assert 1


def test_boundarycalc():
    nabs, ndist = geom.boundarycalc(1190, 25, 1.5, 3400)
    assert(nabs == 182 and ndist == 33)
    return
