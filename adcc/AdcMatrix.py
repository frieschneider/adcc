#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import libadcc
import numpy as np

from .LazyMp import LazyMp
from .adc_pp import matrix as ppmatrix
from .timings import Timer, timed_member_call
from .AdcMethod import AdcMethod
from .functions import empty_like, evaluate
from .Intermediates import Intermediates
from .AmplitudeVector import AmplitudeVector


class AdcMatrixlike:
    """
    Class implementing minimal functionality of AdcMatrixlike objects.

    Note: This is not the user-facing high-level object. Use adcc.AdcMatrix
    if you want to construct an ADC matrix object yourself.
    """
    def __init__(self, innermatrix):
        self.innermatrix = innermatrix

    @property
    def ndim(self):
        return 2

    def __len__(self):
        return self.shape[0]

    def matvec(self, v):
        out = empty_like(v)
        self.compute_matvec(v, out)
        return out

    def rmatvec(self, v):
        # ADC matrix is symmetric
        return self.matvec(v)

    def __matmul__(self, other):
        if isinstance(other, AmplitudeVector):
            return self.compute_matvec(other)
        if isinstance(other, list):
            if all(isinstance(elem, AmplitudeVector) for elem in other):
                return [self.compute_matvec(ov) for ov in other]
        return NotImplemented

    @property
    def intermediates(self):
        return self.innermatrix.intermediates

    @intermediates.setter
    def intermediates(self, new):
        self.innermatrix.intermediates = new

    def construct_symmetrisation_for_blocks(self):
        """
        Construct the symmetrisation functions, which need to be
        applied to relevant blocks of an AmplitudeVector in order
        to symmetrise it to the right symmetry in order to be used
        with the various matrix-vector-products of this function.

        Most importantly the returned functions antisymmetrise
        the occupied and virtual parts of the doubles parts
        if this is sensible for the method behind this adcmatrix.

        Returns a dictionary block identifier -> function
        """
        ret = {}
        if self.is_core_valence_separated:
            # CVS doubles part is antisymmetric wrt. (i,K,a,b) <-> (i,K,b,a)
            ret["d"] = lambda v: v.antisymmetrise([(2, 3)])
        else:
            def symmetrise_generic_adc_doubles(invec):
                # doubles part is antisymmetric wrt. (i,j,a,b) <-> (i,j,b,a)
                scratch = invec.antisymmetrise([(2, 3)])
                # doubles part is symmetric wrt. (i,j,a,b) <-> (j,i,b,a)
                return scratch.symmetrise([(0, 1), (2, 3)])
            ret["d"] = symmetrise_generic_adc_doubles
        return ret

    def dense_basis(self, blocks=None, ordering="adcc"):
        """
        Return the list of indices and their values
        of the dense basis representation

        ordering: adcc, spin, spatial
        """
        ret = []
        if blocks is None:
            blocks = self.blocks

        # Define function to impose the order in the basis
        if ordering == "adcc":
            def reduce_index(n_orbsa, idx):
                return idx, idx
        elif ordering == "spin":
            def reduce_index(n_orbsa, idx):
                is_beta = [idx[i] >= n_orbsa[i] for i in range(len(idx))]
                spatial = [idx[i] - n_orbsa[i] if is_beta[i] else idx[i]
                           for i in range(len(idx))]
                # Sort first by spin, then by spatial
                return (is_beta, spatial)
        elif ordering == "spatial":
            def reduce_index(n_orbsa, idx):
                is_beta = [idx[i] >= n_orbsa[i] for i in range(len(idx))]
                spatial = [idx[i] - n_orbsa[i] if is_beta[i] else idx[i]
                           for i in range(len(idx))]
                # Sort first by spatial, then by spin
                return (spatial, is_beta)

        if "s" in blocks:
            ret_s = []
            sp_s = self.block_spaces("s")
            n_orbs_s = [self.mospaces.n_orbs(sp) for sp in sp_s]
            n_orbsa_s = [self.mospaces.n_orbs_alpha(sp) for sp in sp_s]
            for i in range(n_orbs_s[0]):
                for a in range(n_orbs_s[1]):
                    ret_s.append([((i, a), 1)])

            def sortfctn(x):
                return min(reduce_index(n_orbsa_s, idx) for idx, factor in x)
            ret_s.sort(key=sortfctn)
            ret_s.sort(key=sortfctn)
            ret.extend(ret_s)

        if "d" in blocks:
            ret_d = []
            sp_d = self.block_spaces("d")
            n_orbsa_d = [self.mospaces.n_orbs_alpha(sp) for sp in sp_d]

            if sp_d[0] == sp_d[1] and sp_d[2] == sp_d[3]:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsv = self.mospaces.n_orbs(sp_d[2])
                ret_d.extend([[((i, j, a, b), +1 / 2),
                               ((j, i, a, b), -1 / 2),
                               ((i, j, b, a), -1 / 2),
                               ((j, i, b, a), +1 / 2)]
                              for i in range(nso) for j in range(i)
                              for a in range(nsv) for b in range(a)])
            elif sp_d[2] == sp_d[3]:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsc = self.mospaces.n_orbs(sp_d[1])
                nsv = self.mospaces.n_orbs(sp_d[2])
                ret_d.extend([[((i, j, a, b), +1 / np.sqrt(2)),
                               ((i, j, b, a), -1 / np.sqrt(2))]
                              for i in range(nso) for j in range(nsc)
                              for a in range(nsv) for b in range(a)])
            else:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsc = self.mospaces.n_orbs(sp_d[1])
                nsv = self.mospaces.n_orbs(sp_d[2])
                nsw = self.mospaces.n_orbs(sp_d[3])
                ret_d.append([((i, j, b, a), 1)
                              for i in range(nso) for j in range(nsc)
                              for a in range(nsv) for b in range(nsw)])

            def sortfctn(x):
                return min(reduce_index(n_orbsa_d, idx) for idx, factor in x)
            ret_d.sort(key=sortfctn)
            ret_d.sort(key=sortfctn)
            ret.extend(ret_d)

        if any(b not in "sd" for b in self.blocks):
            raise NotImplementedError("Blocks other than s and d "
                                      "not implemented")
        return ret

    def to_dense_matrix(self, out=None):
        """
        Return the ADC matrix object as a dense numpy array. Converts the sparse
        internal representation of the ADC matrix to a dense matrix and return
        as a numpy array.

        Notes
        -----

        This method is only intended to be used for debugging and
        visualisation purposes as it involves computing a large amount of
        matrix-vector products and the returned array consumes a considerable
        amount of memory.

        The resulting matrix has no spin symmetry imposed, which means that
        its eigenspectrum may contain non-physical excitations (e.g. with linear
        combinations of α->β and α->α components in the excitation vector).

        This function has not been sufficiently tested to be considered stable.
        """
        import tqdm

        from adcc import guess_zero

        # Get zero amplitude of the appropriate symmetry
        # (TODO: Only true for C1, where there is only a single irrep)
        ampl_zero = guess_zero(self)
        assert self.mospaces.point_group == "C1"

        # Build the shape of the returned array
        # Since the basis of the doubles block is not the unit vectors
        # this *not* equal to the shape of the AdcMatrix object
        basis = {b: self.dense_basis(b) for b in self.blocks}
        mat_len = sum(len(basis[b]) for b in basis)

        if out is None:
            out = np.zeros((mat_len, mat_len))
        else:
            if out.shape != (mat_len, mat_len):
                raise ValueError("Output array has shape ({0:}, {1:}), but "
                                 "shape ({2:}, {2:}) is required."
                                 "".format(*out.shape, mat_len))
            out[:] = 0  # Zero all data in out.

        # Check for the cases actually implemented
        if any(b not in "sd" for b in self.blocks):
            raise NotImplementedError("Blocks other than s and d "
                                      "not implemented")
        if "s" not in self.blocks:
            raise NotImplementedError("Block 's' needs to be present")

        # Extract singles-singles block (contiguous)
        assert "s" in self.blocks
        n_orbs_s = [self.mospaces.n_orbs(sp) for sp in self.block_spaces("s")]
        n_s = np.prod(n_orbs_s)
        assert len(basis["s"]) == n_s
        view_ss = out[:n_s, :n_s].reshape(*n_orbs_s, *n_orbs_s)
        for i in range(n_orbs_s[0]):
            for a in range(n_orbs_s[1]):
                ampl = ampl_zero.copy()
                ampl["s"][i, a] = 1
                view_ss[:, :, i, a] = (self @ ampl)["s"].to_ndarray()

        # Extract singles-doubles and doubles-doubles block
        if "d" in self.blocks:
            assert self.blocks == ["s", "d"]
            view_sd = out[:n_s, n_s:].reshape(*n_orbs_s, len(basis["d"]))
            view_dd = out[n_s:, n_s:]
            for j, bas1 in tqdm.tqdm(enumerate(basis["d"]),
                                     total=len(basis["d"])):
                ampl = ampl_zero.copy()
                for idx, val in bas1:
                    ampl["d"][idx] = val
                ret_ampl = self @ ampl
                view_sd[:, :, j] = ret_ampl["s"].to_ndarray()

                for i, bas2 in enumerate(basis["d"]):
                    view_dd[i, j] = sum(val * ret_ampl["d"][idx]
                                        for idx, val in bas2)

            out[n_s:, :n_s] = np.transpose(out[:n_s, n_s:])
        return out


for prop in ["reference_state", "ground_state", "mospaces",
             "is_core_valence_separated", "shape", "blocks", "timer"]:
    def caller(self, propcopy=prop):
        return getattr(self.innermatrix, propcopy)
    setattr(AdcMatrixlike, prop, property(caller))


# TODO Get rid of the core class
class AdcMatrix(AdcMatrixlike):
    def __init__(self, *args, **kwargs):
        super().__init__(AdcMatrixCore(*args, **kwargs))
        self.method = self.innermatrix.method

    def __repr__(self):
        return f"AdcMatrix({self.method.name})"


class AdcMatrixCore:
    def __init__(self, method, hf_or_mp, block_orders=None, intermediates=None):
        """
        Initialise an ADC matrix.

        Parameters
        ----------
        method : str or AdcMethod
            Method to use.
        hf_or_mp : adcc.ReferenceState or adcc.LazyMp
            HF reference or MP ground state
        block_orders : optional
            The order of perturbation theory to employ for each matrix block.
            If not set, defaults according to the selected ADC method are chosen.
        intermediates : adcc.Intermediates or NoneType
            Allows to pass intermediates to re-use to this class.
        """
        if isinstance(hf_or_mp, (libadcc.ReferenceState,
                                 libadcc.HartreeFockSolution_i)):
            hf_or_mp = LazyMp(hf_or_mp)
        if not isinstance(hf_or_mp, LazyMp):
            raise TypeError("mp_results is not a valid object. It needs to be "
                            "either a LazyMp, a ReferenceState or a "
                            "HartreeFockSolution_i.")

        if not isinstance(method, AdcMethod):
            method = AdcMethod(method)

        self.timer = Timer()
        self.method = method
        self.ground_state = hf_or_mp
        self.reference_state = hf_or_mp.reference_state
        self.mospaces = hf_or_mp.reference_state.mospaces
        self.is_core_valence_separated = method.is_core_valence_separated

        # Determine block orders
        new_orders = {
            #             ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),
            "adc0":  dict(ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
            "adc1":  dict(ph_ph=1, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
            "adc2":  dict(ph_ph=2, ph_pphh=1,    pphh_ph=1,    pphh_pphh=0),     # noqa: E501
            "adc2x": dict(ph_ph=2, ph_pphh=1,    pphh_ph=1,    pphh_pphh=1),     # noqa: E501
            "adc3":  dict(ph_ph=3, ph_pphh=2,    pphh_ph=2,    pphh_pphh=1),     # noqa: E501
        }[method.base_method.name]
        if block_orders is not None:
            new_orders.update(block_orders)
        block_orders = new_orders

        # Sanity checks on block_orders
        for block in block_orders.keys():
            if block not in ("ph_ph", "ph_pphh", "pphh_ph", "pphh_pphh"):
                raise ValueError(f"Invalid block order key: {block}")
        if block_orders["ph_pphh"] != block_orders["pphh_ph"]:
            raise ValueError("ph_pphh and pphh_ph should always have "
                             "the same order")
        if block_orders["ph_pphh"] is not None \
           and block_orders["pphh_pphh"] is None:
            raise ValueError("pphh_pphh cannot be node if ph_pphh isn't.")

        if intermediates is None:
            self.intermediates = Intermediates(self.ground_state)
        else:
            self.intermediates = intermediates

        # Build the blocks
        variant = None
        if method.is_core_valence_separated:
            variant = "cvs"
        self.__blocks = {
            block: ppmatrix.block(self.ground_state, block.split("_"), order=order,
                                  intermediates=self.intermediates,
                                  variant=variant)
            for block, order in block_orders.items() if order is not None
        }

        # TODO Remove this
        # automatically get the shape and subspace
        # blocks of the matrix from the orders
        # TODO once the AmplitudeVector object support missing blocks
        #      this would be self.diagonal.blocks and self.diagonal.ph.subspaces
        #      and self.diagonal.pphh.subspaces
        self.__block_spaces = {}
        if variant == "cvs":
            self.__block_spaces["s"] = ["o2", "v1"]
            if block_orders.get("pphh_pphh", None) is not None:
                self.__block_spaces["d"] = ["o1", "o2", "v1", "v1"]
        else:
            self.__block_spaces["s"] = ["o1", "v1"]
            if block_orders.get("pphh_pphh", None) is not None:
                self.__block_spaces["d"] = ["o1", "o1", "v1", "v1"]
        self.blocks = list(self.__block_spaces.keys())

    @timed_member_call()
    def diagonal(self, block):
        if block not in self.blocks:
            raise ValueError("block not in blocks")
        # TODO The block argument should be brought more
        #      in line with the ph, pphh stuff

        # TODO Once the AmplitudeVector object support missing blocks this would
        #      just be return sum(bl.diagonal for bl in self.__blocks)

        if block == "s":
            return evaluate(sum(bl.diagonal.ph
                                for bl in self.__blocks.values()
                                if bl.diagonal and "s" in bl.diagonal.blocks
                                and bl.diagonal.ph))
        else:
            return evaluate(sum(bl.diagonal.pphh
                                for bl in self.__blocks.values()
                                if bl.diagonal and "d" in bl.diagonal.blocks
                                and bl.diagonal.pphh))

    def compute_apply(self, block, tensor):
        # TODO A lot of junk code to get compatibility to the old interface
        key = {"ss": "ph_ph", "sd": "ph_pphh",
               "ds": "pphh_ph", "dd": "pphh_pphh"}[block]
        if block[1] == "s":
            ampl = AmplitudeVector(ph=tensor)
        else:
            ampl = AmplitudeVector(pphh=tensor)
        with self.timer.record("apply/" + block):
            ret = self.__blocks[key].apply(ampl)

        if block[0] == "s":
            return ret.ph
        else:
            return ret.pphh

    @timed_member_call()
    def compute_matvec(self, ampl):
        """
        Compute the matrix-vector product of the ADC matrix
        with an excitation amplitude and return the result
        in the out_ampl if it is given, else the result
        will be returned.
        """
        # TODO Once properly supported in AmplitudeVector, this should be
        # return sum(bl.apply(ampl) for bl in self.__blocks.values())

        res = [bl.apply(ampl) for bl in self.__blocks.values()]
        ph = sum(v.ph for v in res if "s" in v.blocks and v.ph)
        pphh = None
        if "d" in self.blocks:
            pphh = sum(v.pphh for v in res if "d" in v.blocks and v.pphh)
        return AmplitudeVector(ph=ph, pphh=pphh)

    def has_block(self, block):  # TODO This should be deprecated
        return block in self.blocks

    @property
    def shape(self):
        length = 0
        for bl in self.blocks:
            prod = 1
            for sp in self.block_spaces(bl):
                prod *= self.mospaces.n_orbs(sp)
            length += prod
        return (length, length)

    def block_spaces(self, block):  # TODO This should be deprecated
        return self.__block_spaces[block]


# Redirect some functions and properties of AdcMatrixlike to the innermatrix
for wfun in ["compute_apply", "compute_matvec", "diagonal",
             "has_block", "block_spaces"]:
    def caller(self, *args, wfuncopy=wfun, **kwargs):
        return getattr(self.innermatrix, wfuncopy)(*args, **kwargs)
    if hasattr(AdcMatrixCore, wfun):
        caller.__doc__ = getattr(AdcMatrixCore, wfun).__doc__
    setattr(AdcMatrixlike, wfun, caller)


class AdcMatrixShifted(AdcMatrixlike):
    def __init__(self, matrix, shift=0.0):
        """
        Initialise a shifted ADC matrix. Applying this class to a vector ``v``
        represents an efficient version of ``matrix @ v + shift * v``.

        Parameters
        ----------
        matrix : AdcMatrixlike
            Matrix which is shifted
        shift : float
            Value by which to shift the matrix
        """
        super().__init__(matrix)
        self.shift = shift

    def compute_matvec(self, in_ampl):
        out = self.innermatrix.compute_matvec(in_ampl)
        out = out + self.shift * in_ampl
        return out

    def to_dense_matrix(self, out=None):
        self.innermatrix.to_dense_matrix(self, out)
        out = out + self.shift * np.eye(*out.shape)
        return out

    def compute_apply(self, block, in_vec):
        ret = self.innermatrix.compute_apply(block, in_vec)
        if block[0] == block[1]:  # Diagonal block
            ret += self.shift * in_vec
        return ret

    def diagonal(self, block):
        out = self.innermatrix.diagonal(block)
        out = out + self.shift  # Shift the diagonal
        return out
