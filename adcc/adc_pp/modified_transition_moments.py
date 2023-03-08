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


def mtm_adc0(mp, dipop, intermediates):
    #return AmplitudeVector(ph=dipop.ov)
    if dipop.is_symmetric:
        dipop_vo = dipop.ov.transpose((1, 0))
    else:
        dipop_vo = dipop.vo.copy()
    return AmplitudeVector(ph=dipop_vo.transpose((1, 0)))


def mtm_adc1(mp, dipop, intermediates):
    if dipop.is_symmetric:
        dipop_vo = dipop.ov.transpose((1, 0))
    else:
        dipop_vo = dipop.vo.copy()
    f1 = dipop_vo.transpose((1, 0)) + einsum("rkjs,rs->kj", mp.t2(b.oovv), dipop.ov)
    return AmplitudeVector(ph=f1)


def mtm_adc2(mp, dipop, intermediates):
    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm
    hf = mp.reference_state
    if dipop.is_symmetric:
        dipop_vo = dipop.ov.transpose((1, 0))
    else:
        dipop_vo = dipop.vo.copy()
    f1 = (
        dipop_vo.transpose((1, 0))
        + einsum("rkjs,rs->kj", mp.t2(b.oovv), dipop.ov)
        
        # antonia 1:
        #- einsum("krjs,rs->kj", t2, dipop.ov)
        -0.25 * einsum("nvjr,nvsr,sk->kj", t2, t2, dipop_vo)
        -0.25 * einsum("knrs,vnrs,jv->kj", t2, t2, dipop_vo)
        +0.5 * einsum("knjr,vnsr,sv->kj", t2, t2, dipop_vo)
        -1.0 * einsum("vk,vj->kj", dipop.oo, p0.ov)
        +1.0 * einsum("jr,kr->kj", dipop.vv, p0.ov)
        - einsum("ijab,jb->ia", mp.td2(b.oovv), dipop.ov)
        # adcc:
        # - einsum("ijab,jb->ia", t2,
        #          + dipop.ov - 0.5 * einsum("jkbc,kc->jb", t2, dipop.ov))
        # + 0.5 * einsum("ij,ja->ia", p0.oo, dipop.ov)
        # - 0.5 * einsum("ib,ab->ia", dipop.ov, p0.vv)
        # + einsum("ib,ab->ia", p0.ov, dipop.vv)
        # - einsum("ij,ja->ia", dipop.oo, p0.ov)
        # - einsum("ijab,jb->ia", mp.td2(b.oovv), dipop.ov)
    )
    f2 = 0.5*(
        - 2*einsum("vlij,vk->klij", t2, dipop.oo).antisymmetrise(0, 1)
        # + einsum("vkij,vl->klij", t2, dipop.oo)
        + 2*einsum("klsj,is->klij", t2, dipop.vv).antisymmetrise(2, 3)
        # - einsum("klsi,js->klij", t2, dipop.vv)
        # + einsum("ijac,bc->ijab", t2, dipop.vv).antisymmetrise(2, 3)
        # - einsum("ik,kjab->ijab", dipop.oo, t2).antisymmetrise(0, 1)
    )
    return AmplitudeVector(ph=f1, pphh=f2)


def mtm_adc3(mp, dipop, intermediates):
    second_order = mtm_adc2(mp, dipop, intermediates)
    f1 = second_order.ph
    f2 = second_order.pphh

    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm
    td2 = mp.td2(b.oovv)
    tt2 = mp.tt2(b.ooovvv)
    ts3 = mp.ts3(b.ov)
    td3 = mp.td3(b.oovv)

    f1 += (
        + 1.0 * einsum('ib,ab->ia', ts3, dipop.vv)
        - 1.0 * einsum('ja,ji->ia', ts3, dipop.oo)
        - 1.0 * einsum('ijab,jb->ia', td3, dipop.ov)
        + 1.0 * einsum('ijab,kb,jk->ia', t2, p0.ov, dipop.oo)
        + 1.0 * einsum('jkab,kb,ji->ia', t2, p0.ov, dipop.oo)
        - 1.0 * einsum('ijbc,jc,ab->ia', t2, p0.ov, dipop.vv)
        - 1.0 * einsum('ijab,jc,cb->ia', t2, p0.ov, dipop.vv)
        - 0.25 * einsum('ikbc,jkbc,ja->ia', t2, td2, dipop.ov)
        - 0.25 * einsum('jkac,jkbc,ib->ia', t2, td2, dipop.ov)
        + 0.5 * einsum('ijab,jkbc,kc->ia', t2, td2, dipop.ov)
        - 0.25 * einsum('ikbc,jkbc,ja->ia', td2, t2, dipop.ov)
        - 0.25 * einsum('jkac,jkbc,ib->ia', td2, t2, dipop.ov)
        + 0.5 * einsum('ijab,jkbc,kc->ia', td2, t2, dipop.ov)
        + 0.5 * einsum('ijkabc,jkcd,db->ia', tt2, t2, dipop.vv)
        - 0.25 * einsum('ijkbcd,jkcd,ab->ia', tt2, t2, dipop.vv)
        + 0.5 * einsum('ijkabc,klbc,jl->ia', tt2, t2, dipop.oo)
        + 0.25 * einsum('jklabc,klbc,ji->ia', tt2, t2, dipop.oo)
        + 0.25 * einsum('ijac,klbd,klcd,jb->ia', t2, t2, t2, dipop.ov)
        - 0.25 * einsum('ijad,klbc,klcd,jb->ia', t2, t2, t2, dipop.ov)
        + 0.25 * einsum('ijbd,klac,klcd,jb->ia', t2, t2, t2, dipop.ov)
        - 0.25 * einsum('ijcd,klab,klcd,jb->ia', t2, t2, t2, dipop.ov)
        + 0.25 * einsum('ikab,jlcd,klcd,jb->ia', t2, t2, t2, dipop.ov)
        - 0.25 * einsum('ikac,jlbd,klcd,jb->ia', t2, t2, t2, dipop.ov)
        + 0.25 * einsum('ikad,jlbc,klcd,jb->ia', t2, t2, t2, dipop.ov)
        + 0.25 * einsum('ikbc,jlad,klcd,jb->ia', t2, t2, t2, dipop.ov)
        - 0.25 * einsum('ikbd,jlac,klcd,jb->ia', t2, t2, t2, dipop.ov)
        + 0.25 * einsum('ikcd,jlab,klcd,jb->ia', t2, t2, t2, dipop.ov)
        - 0.25 * einsum('ilab,jkcd,klcd,jb->ia', t2, t2, t2, dipop.ov)
        + 0.25 * einsum('ilac,jkbd,klcd,jb->ia', t2, t2, t2, dipop.ov)
        + 0.25 * einsum('ilad,jkbc,klcd,jb->ia', t2, t2, t2, dipop.ov)
        - 0.25 * einsum('ilbc,jkad,klcd,jb->ia', t2, t2, t2, dipop.ov)
        + 0.25 * einsum('ilbd,jkac,klcd,jb->ia', t2, t2, t2, dipop.ov)
    )
    f2 += (
        - 1.0 * einsum('ijbc,ac->ijab', td2, dipop.vv).antisymmetrise(2, 3)
        - 1.0 * einsum('ikab,kj->ijab', td2, dipop.oo).antisymmetrise(0, 1)
        - 0.5 * einsum('ijkabc,kc->ijab', tt2, dipop.ov)
    )
    return AmplitudeVector(ph=f1, pphh=f2)


def mtm_cvs_adc0(mp, dipop, intermediates):
    return AmplitudeVector(ph=dipop.cv)


def mtm_cvs_adc2(mp, dipop, intermediates):
    f1 = (
        + dipop.cv
        - einsum("Ib,ba->Ia", dipop.cv, intermediates.cvs_p0.vv)
        - einsum("Ij,ja->Ia", dipop.co, intermediates.cvs_p0.ov)
    )
    f2 = (1 / sqrt(2)) * einsum("Ik,kjab->jIab", dipop.co, mp.t2(b.oovv))
    return AmplitudeVector(ph=f1, pphh=f2)


DISPATCH = {
    "adc0": mtm_adc0,
    "adc1": mtm_adc1,
    "adc2": mtm_adc2,
    "adc2x": mtm_adc2, # Identical to ADC(2)
    "adc3": mtm_adc3,
    "cvs-adc0": mtm_cvs_adc0,
    "cvs-adc1": mtm_cvs_adc0,  # Identical to CVS-ADC(0)
    "cvs-adc2": mtm_cvs_adc2,
}


def modified_transition_moments(method, ground_state, dipole_operator=None,
                                intermediates=None):
    """Compute the modified transition moments (MTM) for the provided
    ADC method with reference to the passed ground state.

    Parameters
    ----------
    method: adc.Method
        Provide a method at which to compute the MTMs
    ground_state : adcc.LazyMp
        The MP ground state
    dipole_operator : adcc.OneParticleOperator or list, optional
        Only required if different dipole operators than the standard
        dipole operators in the MO basis should be used.
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
    if dipole_operator is None:
        dipole_operator = ground_state.reference_state.operators.electric_dipole
    elif not isinstance(dipole_operator, list):
        unpack = True
        dipole_operator = [dipole_operator]
    if method.name not in DISPATCH:
        raise NotImplementedError("modified_transition_moments is not "
                                  f"implemented for {method.name}.")

    ret = [DISPATCH[method.name](ground_state, dipop, intermediates)
               for dipop in dipole_operator]
    if unpack:
        assert len(ret) == 1
        ret = ret[0]
    return evaluate(ret)
