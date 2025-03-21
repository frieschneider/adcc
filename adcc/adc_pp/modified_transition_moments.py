#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
from math import sqrt

from adcc import block as b
from adcc.LazyMp import LazyMp
from adcc.AdcMethod import AdcMethod
from adcc.functions import einsum, evaluate
from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector


def mtm_adc0(mp, op, intermediates):
    f1 = op.ov if op.is_symmetric else op.vo.transpose()
    return AmplitudeVector(ph=f1)


def mtm_adc1(mp, op, intermediates):
    ampl = mtm_adc0(mp, op, intermediates)
    f1 = - 1.0 * einsum("ijab,jb->ia", mp.t2(b.oovv), op.ov)
    return ampl + AmplitudeVector(ph=f1)


def mtm_adc2(mp, op, intermediates):
    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm

    op_vo = op.ov.transpose() if op.is_symmetric else op.vo

    ampl = mtm_adc1(mp, op, intermediates)
    f1 = (
        + 0.5 * einsum("ijab,jkbc,ck->ia", t2, t2, op_vo)
        + 0.5 * einsum("ij,aj->ia", p0.oo, op_vo)
        - 0.5 * einsum("bi,ab->ia", op_vo, p0.vv)
        + 1.0 * einsum("ib,ab->ia", p0.ov, op.vv)
        - 1.0 * einsum("ji,ja->ia", op.oo, p0.ov)
        - 1.0 * einsum("ijab,jb->ia", mp.td2(b.oovv), op.ov)
    )
    f2 = (
        + 1.0 * einsum("ijac,bc->ijab", t2, op.vv).antisymmetrise(2, 3)
        + 1.0 * einsum("ki,jkab->ijab", op.oo, t2).antisymmetrise(0, 1)
    )
    return ampl + AmplitudeVector(ph=f1, pphh=f2)


def mtm_cvs_adc0(mp, op, intermediates):
    f1 = op.cv if op.is_symmetric else op.vc.transpose()
    return AmplitudeVector(ph=f1)


def mtm_cvs_adc2(mp, op, intermediates):
    op_vc = op.cv.transpose() if op.is_symmetric else op.vc
    op_oc = op.co.transpose() if op.is_symmetric else op.oc

    ampl = mtm_cvs_adc0(mp, op, intermediates)
    f1 = (
        - 0.5 * einsum("bI,ab->Ia", op_vc, intermediates.cvs_p0.vv)
        - 1.0 * einsum("jI,ja->Ia", op_oc, intermediates.cvs_p0.ov)
    )
    f2 = (1 / sqrt(2)) * einsum("kI,kjab->jIab", op_oc, mp.t2(b.oovv))
    return ampl + AmplitudeVector(ph=f1, pphh=f2)


DISPATCH = {
    "adc0": mtm_adc0,
    "adc1": mtm_adc1,
    "adc2": mtm_adc2,
    "adc2x": mtm_adc2,  # Identical to ADC(2)
    "cvs-adc0": mtm_cvs_adc0,
    "cvs-adc1": mtm_cvs_adc0,  # Identical to CVS-ADC(0)
    "cvs-adc2": mtm_cvs_adc2,
}


def modified_transition_moments(method, ground_state, operator=None,
                                intermediates=None):
    """Compute the modified transition moments (MTM) for the provided
    ADC method with reference to the passed ground state.

    Parameters
    ----------
    method: adc.Method
        Provide a method at which to compute the MTMs
    ground_state : adcc.LazyMp
        The MP ground state
    operator : adcc.OneParticleOperator or list, optional
        Only required if different operators than the standard
        electric dipole operators in the MO basis should be used.
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse

    Returns
    -------
    adcc.AmplitudeVector or list of adcc.AmplitudeVector
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    unpack = False
    if operator is None:
        operator = ground_state.reference_state.operators.electric_dipole
    elif not isinstance(operator, list):
        unpack = True
        operator = [operator]
    if method.name not in DISPATCH:
        raise NotImplementedError("modified_transition_moments is not "
                                  f"implemented for {method.name}.")

    ret = [DISPATCH[method.name](ground_state, op, intermediates)
           for op in operator]
    if unpack:
        assert len(ret) == 1
        ret = ret[0]
    return evaluate(ret)
