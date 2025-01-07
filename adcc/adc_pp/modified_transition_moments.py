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
import numpy as np

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

def mtm_adc3(mp, op, intermediates):
    # t2 = mp.t2(b.oovv)
    # p0 = mp.mp2_diffdm
    # td2 = mp.td2(b.oovv)
    # tt2 = mp.tt2(b.ooovvv)
    # ts3 = mp.ts3(b.ov)
    # td3 = mp.td3(b.oovv)

    # op_vo = op.ov.transpose() if op.is_symmetric else op.vo

    # ampl = mtm_adc2(mp, op, intermediates)
    # f1_rouven = (
    #     # op.vv
    #     + 1.0 * einsum('ib,ab->ia', ts3, op.vv)
    #     - 1.0 * einsum('ijbc,jc,ab->ia', t2, p0.ov, op.vv)
    #     - 1.0 * einsum('ijab,jc,cb->ia', t2, p0.ov, op.vv)
    #     + 0.5 * einsum('ijkabc,jkcd,db->ia', tt2, t2, op.vv)
    #     - 0.25 * einsum('ijkbcd,jkcd,ab->ia', tt2, t2, op.vv)
    #     # op.oo
    #     - 1.0 * einsum('ja,ji->ia', ts3, op.oo)
    #     + 1.0 * einsum('ijab,kb,jk->ia', t2, p0.ov, op.oo)
    #     + 1.0 * einsum('jkab,kb,ji->ia', t2, p0.ov, op.oo)
    #     + 0.5 * einsum('ijkabc,klbc,jl->ia', tt2, t2, op.oo)
    #     + 0.25 * einsum('jklabc,klbc,ji->ia', tt2, t2, op.oo)
    #     # op_vo
    #     - 0.25 * einsum('jkbc,ikbc,aj->ia', t2, td2, op_vo)
    #     - 0.25 * einsum('jkac,jkbc,bi->ia', t2, td2, op_vo)
    #     - 0.25 * einsum('jkbc,jibc,ak->ia', td2, t2, op_vo)
    #     - 0.25 * einsum('jkba,jkbc,ci->ia', td2, t2, op_vo)
    #     + 0.5 * einsum('jiac,jkbc,bk->ia', t2, td2, op_vo)
    #     - 0.5 * einsum('ijba,jkbc,ck->ia', td2, t2, op_vo)
    #     # op.ov
    #     - 1.0 * einsum('ijab,jb->ia', td3, op.ov)
    #     + 0.25 * einsum('ijac,klbd,klcd,jb->ia', t2, t2, t2, op.ov)
    #     - 0.25 * einsum('ijad,klbc,klcd,jb->ia', t2, t2, t2, op.ov)
    #     + 0.25 * einsum('ijbd,klac,klcd,jb->ia', t2, t2, t2, op.ov)
    #     - 0.25 * einsum('ijcd,klab,klcd,jb->ia', t2, t2, t2, op.ov)
    #     + 0.25 * einsum('ikab,jlcd,klcd,jb->ia', t2, t2, t2, op.ov)
    #     - 0.25 * einsum('ikac,jlbd,klcd,jb->ia', t2, t2, t2, op.ov)
    #     + 0.25 * einsum('ikad,jlbc,klcd,jb->ia', t2, t2, t2, op.ov)
    #     + 0.25 * einsum('ikbc,jlad,klcd,jb->ia', t2, t2, t2, op.ov)
    #     - 0.25 * einsum('ikbd,jlac,klcd,jb->ia', t2, t2, t2, op.ov)
    #     + 0.25 * einsum('ikcd,jlab,klcd,jb->ia', t2, t2, t2, op.ov)
    #     - 0.25 * einsum('ilab,jkcd,klcd,jb->ia', t2, t2, t2, op.ov)
    #     + 0.25 * einsum('ilac,jkbd,klcd,jb->ia', t2, t2, t2, op.ov)
    #     + 0.25 * einsum('ilad,jkbc,klcd,jb->ia', t2, t2, t2, op.ov)
    #     - 0.25 * einsum('ilbc,jkad,klcd,jb->ia', t2, t2, t2, op.ov)
    #     + 0.25 * einsum('ilbd,jkac,klcd,jb->ia', t2, t2, t2, op.ov)
    # )
    # f2_rouven = (
    #     - 1.0 * einsum('ijbc,ac->ijab', td2, op.vv).antisymmetrise(2, 3)
    #     - 1.0 * einsum('ikab,kj->ijab', td2, op.oo).antisymmetrise(0, 1)
    #     - 0.5 * einsum('ijkabc,kc->ijab', tt2, op.ov)
    # )

    ampl = mtm_adc2(mp, op, intermediates)
    t2_1 = mp.t2(b.oovv)
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)
    t2_3 = mp.td3(b.oovv)
    op_vo = op.ov.transpose() if op.is_symmetric else op.vo
    p0 = mp.mp3_diffdm - mp.mp2_diffdm  # 2nd + 3rd order MP density contribution
    p0_oo, p0_ov, p0_vv = p0.oo, p0.ov, p0.vv
    p0_2 = mp.mp2_diffdm  # 2nd order MP density contribution
    p0_2_oo, p0_2_ov, p0_2_vv = p0_2.oo, p0_2.ov, p0_2.vv    

    f1 = (
        # op.oo
        - 1.0 * einsum('ia,ij->ja', p0_ov, op.oo)
        + 0.5 * einsum('jkbc,iklabc,ij->la', t2_1, t3_2, op.oo)
        - 1.0 * einsum('jb,ikab,ij->ka', p0_2_ov, t2_1, op.oo)
        # op.vv
        + 1.0 * einsum('ib,ab->ia', p0_ov, op.vv)
        + 1.0 * einsum('ja,ijbc,ab->ic', p0_2_ov, t2_1, op.vv)
        + 0.5 * einsum('jkad,ijkbcd,ab->ic', t2_1, t3_2, op.vv)
        # op.vo
        + 0.5 * einsum('ij,ai->ja', p0_oo, op_vo)
        - 0.5 * einsum('ab,ai->ib', p0_vv, op_vo)
        - 0.5 * einsum('ikab,jkbc,ai->jc', t2_1, t2_2, op_vo)
        - 0.5 * einsum('ikab,jkbc,ai->jc', t2_2, t2_1, op_vo)
        # op.ov
        - 1.0 * einsum('ijab,ia->jb', t2_3, op.ov)
        + 1.0 * einsum('ik,jkab,ia->jb', p0_2_oo, t2_1, op.ov)
        + 0.5 * einsum('ijac,bc,ia->jb', t2_1, p0_2_vv, op.ov)
        - 1.0 * einsum('ac,ijbc,ia->jb', p0_2_vv, t2_1, op.ov)
        - 0.5 * einsum('ikab,jk,ia->jb', t2_1, p0_2_oo, op.ov)
        + 1.0 * einsum('jlad,klcd,ikbc,ia->jb', t2_1, t2_1, t2_1, op.ov)
        + 0.5 * einsum('ilac,klcd,jkbd,ia->jb', t2_1, t2_1, t2_1, op.ov)
        -0.25 * einsum('klab,ijcd,klcd,ia->jb', t2_1, t2_1, t2_1, op.ov)

    )
    f2 = (
        # op.oo
        - 1.0 * einsum('ikab,ij->jkab', t2_2, op.oo).antisymmetrise(0,1)
        # op.vv
        + 1.0 * einsum('ijbc,ab->ijac', t2_2, op.vv).antisymmetrise(2,3)
        # op.ov
        - 0.5 * einsum('ijkabc,ia->jkbc', t3_2, op.ov)
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
    "adc2x": mtm_adc2, # identical to ADC(2)
    "adc3": mtm_adc3,
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
