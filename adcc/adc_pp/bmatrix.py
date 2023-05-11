#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2022 by the adcc authors
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
from collections import namedtuple
from adcc import block as b
from adcc.functions import einsum
from adcc.AmplitudeVector import AmplitudeVector


__all__ = ["block"]


#
# Dispatch routine
#

"""
`apply` is a function mapping an AmplitudeVector to the contribution of this
block to the result of applying the ADC matrix.
"""
AdcBlock = namedtuple("AdcBlock", ["apply"])


def block(ground_state, operator, spaces, order, variant=None):
    """
    Gets ground state, one-particle matrix elements associated
    with a one-particle operator, spaces (ph, pphh and so on)
    and the perturbation theory order for the block,
    variant is "cvs" or sth like that.

    The matrix-vector product was derived up to second order
    using the original equations from
    J. Schirmer and A. B. Trofimov, J. Chem. Phys. 120, 11449–11464 (2004).
    """
    if isinstance(variant, str):
        variant = [variant]
    elif variant is None:
        variant = []

    if ground_state.has_core_occupied_space and "cvs" not in variant:
        raise ValueError("Cannot run a general (non-core-valence approximated) "
                         "ADC method on top of a ground state with a "
                         "core-valence separation.")
    if not ground_state.has_core_occupied_space and "cvs" in variant:
        raise ValueError("Cannot run a core-valence approximated ADC method on "
                         "top of a ground state without a "
                         "core-valence separation.")

    fn = "_".join(["block"] + variant + spaces + [str(order)])

    if fn not in globals():
        raise ValueError("Could not dispatch: "
                         f"spaces={spaces} order={order} variant=variant")
    return globals()[fn](ground_state, operator)


#
# 0th order main
#
def block_ph_ph_0(ground_state, op):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + 1.0 * einsum('ic,ac->ia', ampl.ph, op.vv)
            - 1.0 * einsum('ka,ki->ia', ampl.ph, op.oo)
        ))
    return AdcBlock(apply)


def block_pphh_pphh_0(ground_state, op):
    def apply(ampl):
        return AmplitudeVector(pphh=0.5 * (
            (
                + 2.0 * einsum('ijcb,ac->ijab', ampl.pphh, op.vv)
                - 2.0 * einsum('ijca,bc->ijab', ampl.pphh, op.vv)
            ).antisymmetrise(2, 3)
            + (
                - 2.0 * einsum('kjab,ki->ijab', ampl.pphh, op.oo)
                + 2.0 * einsum('kiab,kj->ijab', ampl.pphh, op.oo)
            ).antisymmetrise(0, 1)
        ))
    return AdcBlock(apply)


#
# 0th order coupling
#
def block_ph_pphh_0(ground_state, op):
    return AdcBlock(lambda ampl: 0, 0)


def block_pphh_ph_0(ground_state, op):
    return AdcBlock(lambda ampl: 0, 0)


#
# 1st order main
#
block_ph_ph_1 = block_ph_ph_0

block_pphh_pphh_1 = block_pphh_pphh_0

#
# 1st order coupling
#
def block_ph_pphh_1(ground_state, op):
    if op.is_symmetric:
        op_vo = op.ov.transpose()
    else:
        op_vo = op.vo
    t2 = ground_state.t2(b.oovv)

    def apply(ampl):
        f1 = 0.5 * (
            - 2.0 * einsum('ilad,ld->ia', ampl.pphh, op.ov)
            + 2.0 * einsum('ilad,lndf,fn->ia', ampl.pphh, t2, op_vo)
            + 2.0 * einsum('ilca,lc->ia', ampl.pphh, op.ov)
            - 2.0 * einsum('ilca,lncf,fn->ia', ampl.pphh, t2, op_vo)
            - 2.0 * einsum('klad,kled,ei->ia', ampl.pphh, t2, op_vo)
            - 2.0 * einsum('ilcd,nlcd,an->ia', ampl.pphh, t2, op_vo)
        )
        return AmplitudeVector(ph=f1)
    return AdcBlock(apply)


def block_pphh_ph_1(ground_state, op):
    if op.is_symmetric:
        op_vo = op.ov.transpose()
    else:
        op_vo = op.vo
    t2 = ground_state.t2(b.oovv)

    def apply(ampl):
        f2 = 0.5 * (
            4.0 * (
            + 1.0 * einsum('ia,kc,jkbc->ijab', ampl.ph, op.ov, t2) 
            + 1.0 * einsum('ja,bi->ijab', ampl.ph, op_vo) 
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 2.0 * (
            + 1.0 * einsum('ka,kc,ijbc->ijab', ampl.ph, op.ov, t2) 
            ).antisymmetrise(2, 3)
            + 2.0 * (
            + 1.0 * einsum('ic,kc,jkab->ijab', ampl.ph, op.ov, t2)  
            ).antisymmetrise(0, 1)
        )
        return AmplitudeVector(pphh=f2)
    return AdcBlock(apply)


#
# 2nd order main
#
def block_ph_ph_2(ground_state, op):
    if op.is_symmetric:
        op_vo = op.ov.transpose()
    else:
        op_vo = op.vo
    p0 = ground_state.mp2_diffdm
    t2 = ground_state.t2(b.oovv)

    def apply(ampl):
        first_order = block_ph_ph_1(ground_state, op).apply(ampl)
        f1 = first_order.ph
        f1 += (
        # (2,1)
        - 1.0 * einsum('ic,jc,aj->ia', ampl.ph, p0.ov, op_vo)
        - 1.0 * einsum('ka,kb,bi->ia', ampl.ph, p0.ov, op_vo)
        - 1.0 * einsum('ic,ja,jc->ia', ampl.ph, p0.ov, op.ov)  # h.c.
        - 1.0 * einsum('ka,ib,kb->ia', ampl.ph, p0.ov, op.ov)  # h.c.
        # (2,2)
        - 0.25 * einsum('ic,mnef,mnaf,ec->ia', ampl.ph, t2, t2, op.vv)
        - 0.25 * einsum('ic,mnef,mncf,ae->ia', ampl.ph, t2, t2, op.vv)  # h.c.
        # (2,3)
        - 0.5 * einsum('ic,mnce,mnaf,ef->ia', ampl.ph, t2, t2, op.vv)
        + 1.0 * einsum('ic,mncf,jnaf,jm->ia', ampl.ph, t2, t2, op.oo)
        # (2,4)
        + 0.25 * einsum('ka,mnef,inef,km->ia', ampl.ph, t2, t2, op.oo)
        + 0.25 * einsum('ka,mnef,knef,mi->ia', ampl.ph, t2, t2, op.oo)  # h.c.
        # (2,5)
        - 1.0 * einsum('ka,knef,indf,ed->ia', ampl.ph, t2, t2, op.vv)
        + 0.5 * einsum('ka,knef,imef,mn->ia', ampl.ph, t2, t2, op.oo)
        # (2,6)
        + 0.5 * einsum('kc,knef,inaf,ec->ia', ampl.ph, t2, t2, op.vv)
        - 0.5 * einsum('kc,mncf,inaf,km->ia', ampl.ph, t2, t2, op.oo)
        + 0.5 * einsum('kc,inef,kncf,ae->ia', ampl.ph, t2, t2, op.vv)  # h.c.
        - 0.5 * einsum('kc,mnaf,kncf,mi->ia', ampl.ph, t2, t2, op.oo)  # h.c.
        # (2,7)
        - 1.0 * einsum('kc,kncf,imaf,mn->ia', ampl.ph, t2, t2, op.oo)
        + 1.0 * einsum('kc,knce,inaf,ef->ia', ampl.ph, t2, t2, op.vv)
        )
        return AmplitudeVector(ph=f1)
    return AdcBlock(apply)

#
# 2nd order couplings
#
def block_ph_pphh_2(ground_state, op):
    if op.is_symmetric:
        op_vo = op.ov.transpose((1,0))
    else:
        op_vo = op.vo.copy()
    p0 = ground_state.mp2_diffdm
    t2 = ground_state.t2(b.oovv)
    td2 = ground_state.td2(b.oovv)
    
    def apply(ampl):
        first_order = block_ph_pphh_1(ground_state, op).apply(ampl)
        f1 = first_order.ph
        f1 += (
            # op.vv
            - 2.0 * einsum('ijac, jb, bc -> ia', ampl.pphh, p0.ov, op.vv)
            # op.oo
            + 2.0 * einsum('ikab, jb, kj -> ia', ampl.pphh, p0.ov, op.oo)
            # op_vo
            + 2.0 * einsum('ikac, jkbc, bj -> ia', ampl.pphh, td2, op_vo)
            + 1.0 * einsum('ijbc, jkbc, ak -> ia', ampl.pphh, td2, op_vo)
            - 1.0 * einsum('jkac, jkbc, bi -> ia', ampl.pphh, td2, op_vo)
            # op.ov
            + 1.0 * einsum('ijcd, jlcb, klba, kd -> ia', ampl.pphh, t2, t2, op.ov)
            + 1.0 * einsum('ilba, jkdc, jlbc, kd -> ia', ampl.pphh, t2, t2, op.ov)
            - 1.0 * einsum('jkba, ildc, klbc, jd -> ia', ampl.pphh, t2, t2, op.ov)
            - 0.5 * einsum('jkcd, ilab, klcd, jb -> ia', ampl.pphh, t2, t2, op.ov)
            + 0.5 * einsum('ikad, jlbc, jlbd, kc -> ia', ampl.pphh, t2, t2, op.ov)
            + 0.5 * einsum('jlad, ikbc, jlbd, kc -> ia', ampl.pphh, t2, t2, op.ov)
            - 0.5 * einsum('jkcd, ilba, jkbd, lc -> ia', ampl.pphh, t2, t2, op.ov)
            + 0.5 * einsum('ijad, jlcb, klcb, kd -> ia', ampl.pphh, t2, t2, op.ov)
            - 0.5 * einsum('ijbc, klda, jlbc, kd -> ia', ampl.pphh, t2, t2, op.ov)
            - 0.25 * einsum('jlad, ikcb, jlcb, kd -> ia', ampl.pphh, t2, t2, op.ov)
            + 0.25 * einsum('ijbc, klda, klbc, jd -> ia', ampl.pphh, t2, t2, op.ov)
               )
        return AmplitudeVector(ph = f1)
    return AdcBlock(apply)

def block_pphh_ph_2(ground_state, op):
    if op.is_symmetric:
        op_vo = op.ov.transpose((1,0))
    else:
        op_vo = op.vo.copy()
    p0 = ground_state.mp2_diffdm
    t2 = ground_state.t2(b.oovv)
    td2 = ground_state.td2(b.oovv)

    def apply(ampl):
        first_order = block_pphh_ph_1(ground_state, op).apply(ampl)
        f2 = first_order.pphh
        f2 += 0.5 * (
            4.0 * (
                + 1.0 * einsum('ia,kc,jkbc->ijab', ampl.ph, op.ov, td2)  
                + 0.5 * einsum('ja,dl,ikbc,klcd->ijab', ampl.ph, op_vo, t2, t2)  
                + 0.5 * einsum('la,di,jkbc,klcd->ijab', ampl.ph, op_vo, t2, t2)  
                + 0.5 * einsum('id,al,jkbc,klcd->ijab', ampl.ph, op_vo, t2, t2)  
                - 0.25 * einsum('ia,dj,klbc,klcd->ijab', ampl.ph, op_vo, t2, t2) 
                - 0.25 * einsum('ja,bk,ilcd,klcd->ijab', ampl.ph, op_vo, t2, t2) 
                + 1.0 * einsum('ja,bc,ic->ijab', ampl.ph, op.vv, p0.ov)  
                + 1.0 * einsum('ia,kj,kb->ijab', ampl.ph, op.oo, p0.ov)  
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 2.0 * (
                + 1.0 * einsum('ic,kc,jkab->ijab', ampl.ph, op.ov, td2)  
                + 0.5 * einsum('jc,dl,ikab,klcd->ijab', ampl.ph, op_vo, t2, t2)  
                + 0.5 * einsum('lc,di,jkab,klcd->ijab', ampl.ph, op_vo, t2, t2)  
                - 0.25 * einsum('ic,dj,klab,klcd->ijab', ampl.ph, op_vo, t2, t2) 
            ).antisymmetrise(0, 1)
            + 2.0 * (
                + 1.0 * einsum('ka,kc,ijbc->ijab', ampl.ph, op.ov, td2) 
                + 0.5 * einsum('kb,dl,ijac,klcd->ijab', ampl.ph, op_vo, t2, t2) 
                + 0.5 * einsum('kd,al,ijbc,klcd->ijab', ampl.ph, op_vo, t2, t2) 
                - 0.25 * einsum('ka,bl,ijcd,klcd->ijab', ampl.ph, op_vo, t2, t2) 
            ).antisymmetrise(2, 3)
        )
        return AmplitudeVector(pphh = f2)
    return AdcBlock(apply)

# 
# 3rd order main
#
def block_ph_ph_3(ground_state, op):
    if op.is_symmetric:
        op_vo = op.ov.transpose((1,0))
    else:
        op_vo = op.vo.copy()
    p0 = ground_state.mp2_diffdm
    t2 = ground_state.t2(b.oovv)
    td2 = ground_state.td2(b.oovv)
    tt2 = ground_state.tt2(b.ooovvv)
    ts3 = ground_state.ts3(b.ov)

    def apply(ampl):
        second_order = block_ph_ph_2(ground_state, op).apply(ampl)
        f1 = second_order.ph
        f1 += (
            (
            # op.vv
            + 1.0 * einsum('ka,cd,ijbd,jkbc->ia', ampl.ph, op.vv, t2, td2)  
            + 1.0 * einsum('kd,cb,ijab,jkcd->ia', ampl.ph, op.vv, t2, td2)  
            + 0.5 * einsum('jc,ab,ikbd,jkcd->ia', ampl.ph, op.vv, t2, td2) 
            + 0.5 * einsum('jc,ad,jkbc,ikbd->ia', ampl.ph, op.vv, t2, td2) 
            + 0.5 * einsum('kc,dc,ijab,jkbd->ia', ampl.ph, op.vv, t2, td2) 
            + 0.5 * einsum('id,cb,jkab,jkcd->ia', ampl.ph, op.vv, t2, td2) 
            - 1.0 * einsum('ja,dc,jkbd,ikbc->ia', ampl.ph, op.vv, t2, td2)  
            - 1.0 * einsum('kb,dc,jkbd,ijac->ia', ampl.ph, op.vv, t2, td2)  
            - 0.5 * einsum('ib,dc,jkbd,jkac->ia', ampl.ph, op.vv, t2, td2) 
            - 0.5 * einsum('kb,cb,jkcd,ijad->ia', ampl.ph, op.vv, t2, td2) 
            - 0.25 * einsum('ib,cb,jkcd,jkad->ia', ampl.ph, op.vv, t2, td2) 
            - 0.25 * einsum('ic,ab,jkbd,jkcd->ia', ampl.ph, op.vv, t2, td2) 
            - 0.25 * einsum('ic,ab,jkcd,jkbd->ia', ampl.ph, op.vv, t2, td2) 
            + 0.25 * einsum('id,cd,jkab,jkbc->ia', ampl.ph, op.vv, t2, td2) 
            # op.oo
            + 1.0 * einsum('kb,lj,jkbc,ilac->ia', ampl.ph, op.oo, t2, td2)  
            + 1.0 * einsum('ic,lj,klab,jkbc->ia', ampl.ph, op.oo, t2, td2)  
            + 0.5 * einsum('ja,lk,jkbc,ilbc->ia', ampl.ph, op.oo, t2, td2) 
            - 1.0 * einsum('ib,jl,klbc,jkac->ia', ampl.ph, op.oo, t2, td2)  
            - 1.0 * einsum('lc,kj,ikab,jlbc->ia', ampl.ph, op.oo, t2, td2)  
            - 0.5 * einsum('la,kj,ikbc,jlbc->ia', ampl.ph, op.oo, t2, td2) 
            - 0.5 * einsum('jb,jk,klbc,ilac->ia', ampl.ph, op.oo, t2, td2) 
            - 0.5 * einsum('jb,ki,jlbc,klac->ia', ampl.ph, op.oo, t2, td2) 
            - 0.5 * einsum('kb,ji,jlac,klbc->ia', ampl.ph, op.oo, t2, td2) 
            - 0.5 * einsum('kc,kl,ijab,jlbc->ia', ampl.ph, op.oo, t2, td2) 
            - 0.25 * einsum('ja,jl,ikbc,klbc->ia', ampl.ph, op.oo, t2, td2) 
            - 0.25 * einsum('ka,kl,jlbc,ijbc->ia', ampl.ph, op.oo, t2, td2) 
            + 0.25 * einsum('ja,ki,jlbc,klbc->ia', ampl.ph, op.oo, t2, td2) 
            + 0.25 * einsum('ka,ji,jlbc,klbc->ia', ampl.ph, op.oo, t2, td2) 
            # op_vo
            - 1.0 * einsum('ja,bi,jb->ia', ampl.ph, op_vo, ts3) 
            - 1.0 * einsum('ib,aj,jb->ia', ampl.ph, op_vo, ts3) 
            + 1.0 * einsum('ib,aj,jkbc,kc->ia', ampl.ph, op_vo, t2, p0.ov) 
            + 1.0 * einsum('jb,ak,jkbc,ic->ia', ampl.ph, op_vo, t2, p0.ov) 
            + 1.0 * einsum('jb,ci,jkbc,ka->ia', ampl.ph, op_vo, t2, p0.ov) 
            + 1.0 * einsum('jd,ck,ilab,jklbcd->ia', ampl.ph, op_vo, t2, tt2) 
            + 0.5 * einsum('kd,ci,jlab,jklbcd->ia', ampl.ph, op_vo, t2, tt2) 
            - 1.0 * einsum('ja,ci,jkbc,kb->ia', ampl.ph, op_vo, t2, p0.ov) 
            - 1.0 * einsum('ka,cj,jkbc,ib->ia', ampl.ph, op_vo, t2, p0.ov) 
            - 1.0 * einsum('ib,cj,jkbc,ka->ia', ampl.ph, op_vo, t2, p0.ov) 
            - 0.5 * einsum('ja,cl,ikbd,jklbcd->ia', ampl.ph, op_vo, t2, tt2) 
            - 0.5 * einsum('jc,ak,ilbd,jklbcd->ia', ampl.ph, op_vo, t2, tt2) 
            - 0.5 * einsum('id,ck,jlab,jklbcd->ia', ampl.ph, op_vo, t2, tt2) 
            - 0.25 * einsum('ja,ci,klbd,jklbcd->ia', ampl.ph, op_vo, t2, tt2) 
            - 0.25 * einsum('ic,aj,klbd,jklbcd->ia', ampl.ph, op_vo, t2, tt2) 
            # op.ov
            - 1.0 * einsum('ja,jb,ib->ia', ampl.ph, op.ov, ts3) 
            - 1.0 * einsum('ib,jb,ja->ia', ampl.ph, op.ov, ts3) 
            + 1.0 * einsum('ka,jc,ijbc,kb->ia', ampl.ph, op.ov, t2, p0.ov) 
            + 1.0 * einsum('ka,kb,ijbc,jc->ia', ampl.ph, op.ov, t2, p0.ov) 
            + 1.0 * einsum('ic,jc,jkab,kb->ia', ampl.ph, op.ov, t2, p0.ov) 
            + 1.0 * einsum('ic,kb,jkab,jc->ia', ampl.ph, op.ov, t2, p0.ov) 
            + 1.0 * einsum('jc,jb,ikab,kc->ia', ampl.ph, op.ov, t2, p0.ov) 
            + 1.0 * einsum('jc,kc,ikab,jb->ia', ampl.ph, op.ov, t2, p0.ov) 
            + 0.5 * einsum('ka,jc,klbd,ijlbcd->ia', ampl.ph, op.ov, t2, tt2) 
            + 0.5 * einsum('kc,jc,klbd,ijlabd->ia', ampl.ph, op.ov, t2, tt2) 
            + 0.5 * einsum('lc,ld,jkbc,ijkabd->ia', ampl.ph, op.ov, t2, tt2) 
            - 1.0 * einsum('kc,jd,klbc,ijlabd->ia', ampl.ph, op.ov, t2, tt2)    
            - 0.5 * einsum('ic,lb,jkcd,jklabd->ia', ampl.ph, op.ov, t2, tt2)   
            - 0.25 * einsum('la,lc,jkbd,ijkbcd->ia', ampl.ph, op.ov, t2, tt2)  
            + 0.25 * einsum('ic,jc,klbd,jklabd->ia', ampl.ph, op.ov, t2, tt2)  
            ))
        return AmplitudeVector(ph = f1)
    return AdcBlock(apply)
