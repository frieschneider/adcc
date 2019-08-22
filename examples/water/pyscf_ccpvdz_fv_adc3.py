#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from pyscf import gto, scf

# Run SCF in pyscf
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='cc-pvdz',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-13
scfres.kernel()

#
# Some more advanced memory tampering options
#
# Initialise ADC memory (512 MiB)
# Use a tensor block size parameter of 16 and
# a specific allocator (in this case std::allocator)
adcc.memory_pool.initialise(max_memory=512 * 1024 * 1024,
                            tensor_block_size=12, allocator="standard")

# Run an adc3 calculation:
singlets = adcc.adc3(scfres, frozen_virtual=3, n_singlets=3)
triplets = adcc.adc3(singlets.matrix, n_triplets=3)

print(singlets.describe())
print()
print(triplets.describe())
