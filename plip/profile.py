"""
Python API for profiling a System given as mmCIF file(s).

    from plip.profile import profile_system
    arrays = profile_system(receptor=['rec.cif'], ligands=['lig1.cif', 'lig2.cif'], npz='system.npz')
"""
import contextlib
import copy
import gzip
import os
import tempfile

import lxml.etree as et
import numpy as np

from plip.basic import config, logger
from plip.exchange.npz import interaction_arrays
from plip.exchange.report import StructureReport
from plip.structure.cif import CIFSystem
from plip.structure.preparation import PDBComplex

logger = logger.get_logger()


@contextlib.contextmanager
def config_scope():
    """Restores all global PLIP settings after the block. PLIP keeps its settings in the module
    plip.basic.config, so profiling is not thread-safe; use processes for parallel runs."""
    saved = {k: copy.deepcopy(v) for k, v in vars(config).items() if k.isupper()}
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def profile_system(structure=None, receptor=None, ligands=None, *, chains=None, peptides=None, intra=None,
                   regions=None, model=1, keepmod=False, nohydro=False, breakcomposite=False, altlocation=False,
                   dnareceptor=False, nopdbcanmap=False, name=None, npz=None, xml=None, txt=None):
    """Profiles the interactions of a System given as mmCIF file(s) and returns them as numpy arrays.

    Either `structure` (Single-file input: one mmCIF file with receptor and ligands) or `receptor`
    (Split input: list of receptor mmCIF files, with an optional list of `ligands` mmCIF files) must be given.
    In a Split input, each ligand file is one Ligand, except a file with a single metal atom, which is a
    Cofactor and stays in the surroundings.

    Chains are given with their original IDs from the input files:
      chains: [['1.A'], ['1.B']] -- receptor chains, ligand chains (inter-chain interactions)
      peptides: ['1.B'] -- chains taken as peptide ligands
      intra: '1.A' -- chain for intra-chain interactions
      regions: [({'1.A': [(1, 20), 25]}, {'1.B': [(17, 46)]})] -- (ligand region, receptor region or None)
    The other keyword arguments correspond to the command line options of the same name.
    `npz`, `xml` and `txt` are optional output paths (xml/txt ending with .gz are compressed).

    Returns a dict of numpy arrays with the same content as the NPZ file (see plip.exchange.npz).
    """
    if (structure is None) == (receptor is None):
        raise ValueError('either structure or receptor must be given')
    if structure is not None and ligands:
        raise ValueError('ligands can only be given together with receptor')
    if sum(x is not None and x != [] for x in (chains, peptides, intra, regions)) > 1:
        raise ValueError('chains, peptides, intra and regions are mutually exclusive')

    with config_scope():
        config.MODEL = model
        config.KEEPMOD = keepmod
        config.NOHYDRO = nohydro
        config.BREAKCOMPOSITE = breakcomposite
        config.ALTLOC = altlocation
        config.DNARECEPTOR = dnareceptor
        config.NOFIXFILE = True
        if structure is not None:
            files, roles = [structure], ['complex']
        else:
            files = list(receptor) + list(ligands or [])
            roles = ['receptor'] * len(receptor) + ['ligand'] * len(ligands or [])
        system = CIFSystem(files, roles, model=model, keep_hydrogens=nohydro)

        config.PEPTIDES = [system.resolve_chain(c) for c in peptides] if peptides else []
        config.INTRA = system.resolve_chain(intra) if intra is not None else None
        config.CHAINS = [[system.resolve_chain(c) for c in group] for group in chains] if chains else None
        config.REGIONS = _resolve_regions(system, regions) if regions else None
        config.NOPDBCANMAP = bool(nopdbcanmap or config.INTRA or config.PEPTIDES)
        ppi_mode = config.PEPTIDES or config.INTRA is not None or config.CHAINS or config.REGIONS
        ligand_groups = [g for _, g in system.ligand_groups] if system.is_split and not ppi_mode else None
        if name is None:
            name = _basename(files[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            pdbpath = os.path.join(tmpdir, f'{name}.pdb')
            with open(pdbpath, 'w') as f:
                f.write(system.pdb_text)
            mol = PDBComplex()
            mol.output_path = tmpdir
            mol.identifier_map = system.identifier_map
            mol.cif_system = system
            mol.load_pdb(pdbpath, ligand_groups=ligand_groups)
            if mol.protcomplex.OBMol.NumAtoms() < len(system.serial_file):
                raise RuntimeError('OpenBabel did not read all atoms of the system')
            if system.is_split and not ppi_mode and not mol.ligands:
                logger.warning('the system contains no ligands (only cofactors)')
            mol.filetype = 'mmcif'
            mol.pymol_name = name
            for ligand in mol.ligands:
                mol.characterize_complex(ligand)
            mol.sourcefiles['pdbcomplex'] = ','.join(files)
            mol.sourcefiles['filename'] = ','.join(os.path.basename(f) for f in files)

            arrays = interaction_arrays(mol, system)
            if xml is not None or txt is not None:
                report = StructureReport(mol)
                if xml is not None:
                    _write(xml, et.tostring(et.ElementTree(report.xmlreport), pretty_print=True,
                                            xml_declaration=True, encoding='utf-8'))
                if txt is not None:
                    _write(txt, ''.join(line + '\n' for line in report.txtreport).encode('utf-8'))
        if npz is not None:
            np.savez_compressed(npz, **arrays)
    return arrays


def _resolve_regions(system, regions):
    """Translates chain IDs in regions to internal ones and expands (start, end) residue ranges."""
    if isinstance(regions, tuple):
        regions = [regions]

    def expand(region):
        if region is None:
            return None
        expanded = {}
        for chain, residues in region.items():
            numbers = []
            for r in residues:
                numbers.extend(range(r[0], r[1] + 1) if isinstance(r, (tuple, list)) else [r])
            expanded[system.resolve_chain(chain)] = numbers
        return expanded

    return [(expand(lig_rec[0]), expand(lig_rec[1]) if len(lig_rec) > 1 else None) for lig_rec in regions]


def _basename(path):
    name = os.path.basename(path)
    for ext in ('.gz', '.cif', '.mmcif'):
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
    return name


def _write(path, content):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'wb') as f:
        f.write(content)
