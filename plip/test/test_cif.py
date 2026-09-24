"""
Protein-Ligand Interaction Profiler - Analyze and visualize protein-ligand interactions in PDB files.
test_cif.py - Unit Tests for mmCIF input (Single-file and Split input) and NPZ output.
"""
import os
import subprocess
import sys
import tempfile
import unittest

import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import lxml.etree as et
import numpy as np

from plip.basic import config
from plip.exchange.npz import INTERACTION_TYPES
from plip.profile import profile_system
from plip.structure.cif import CIFSystem, read_cif
from plip.structure.preparation import PDBComplex

CIF_DIR = './cif'


def system_files(system):
    """Receptor file and ligand files of a PLINDER system directory."""
    path = os.path.join(CIF_DIR, system)
    ligand_dir = os.path.join(path, 'ligand_structures')
    ligands = [os.path.join(ligand_dir, chain, 'ligand_model_0.cif') for chain in sorted(os.listdir(ligand_dir))]
    return os.path.join(path, 'receptor_structures', 'receptor_model_0.cif'), ligands


def merged_atoms(files):
    atoms = [read_cif(f) for f in files]
    merged = atoms[0]
    for a in atoms[1:]:
        merged += a
    return merged


def distance(arrays, i):
    for name in ('dist', 'dist_d_a', 'dist_a_w', 'centdist'):
        if not np.isnan(arrays[name][i]):
            return round(float(arrays[name][i]), 2)


def signature_from_arrays(arrays):
    """Interactions as comparable tuples (type, residue, ligand residue, distance)."""
    return sorted((str(arrays['interaction_types'][arrays['interaction_type'][i]]),
                   int(arrays['resnr'][i]), str(arrays['reschain'][i]),
                   int(arrays['resnr_lig'][i]), str(arrays['reschain_lig'][i]), distance(arrays, i))
                  for i in range(len(arrays['interaction_type'])))


def signature_from_complex(mol, chain_names):
    signature = []
    for site in mol.interaction_sets.values():
        for ix in site.all_itypes:
            itype = _itype_of(ix)
            dist = next(getattr(ix, n) for n in ('distance', 'distance_ad', 'distance_aw') if hasattr(ix, n))
            signature.append((itype, int(ix.resnr), chain_names[ix.reschain], int(ix.resnr_l),
                              chain_names[ix.reschain_l], round(float(dist), 2)))
    return sorted(signature)


def _itype_of(ix):
    return {'hydroph_interaction': 'hydrophobic', 'hbond': 'hbond', 'waterbridge': 'waterbridge',
            'saltbridge': 'saltbridge', 'pistack': 'pistacking', 'pication': 'pication',
            'halogenbond': 'halogen', 'metal_complex': 'metal'}[type(ix).__name__]


class CIFInputTest(unittest.TestCase):
    """Checks mmCIF input against PDB input and the NPZ output against the input files."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def assert_split_equals_single_file_and_pdb(self, system):
        receptor, ligands = system_files(system)
        split = profile_system(receptor=[receptor], ligands=ligands)
        self.assertGreater(len(split['interaction_type']), 0)

        # Single-file input: the same atoms written as one mmCIF file
        atoms = merged_atoms([receptor] + ligands)
        atoms.del_annotation('altloc_id')
        cif = pdbx.CIFFile()
        pdbx.set_structure(cif, atoms)
        cifpath = os.path.join(self.tmp_dir.name, 'complex.cif')
        cif.write(cifpath)
        single = profile_system(structure=cifpath)
        self.assertEqual(signature_from_arrays(split), signature_from_arrays(single))

        # PDB input: the same atoms with single-character chain IDs and three-character residue names
        chain_names = {}
        for i, chain in enumerate(dict.fromkeys(atoms.chain_id)):
            chain_names['ABCDEFGH'[i]] = str(chain)
            atoms.chain_id[atoms.chain_id == chain] = 'ABCDEFGH'[i]
        atoms.res_name[np.char.str_len(atoms.res_name.astype(str)) > 3] = 'LIG'
        atoms.b_factor[np.isnan(atoms.b_factor)] = 0
        pdbfile = pdb.PDBFile()
        pdbfile.set_structure(atoms)
        pdbpath = os.path.join(self.tmp_dir.name, 'complex.pdb')
        pdbfile.write(pdbpath)
        mol = PDBComplex()
        mol.output_path = self.tmp_dir.name
        mol.load_pdb(pdbpath)
        for ligand in mol.ligands:
            mol.characterize_complex(ligand)
        self.assertEqual(signature_from_arrays(split), signature_from_complex(mol, chain_names))

    def test_split_equals_single_file_and_pdb_1a0f(self):
        """Split input, Single-file input and PDB input of the same System yield the same interactions."""
        self.assert_split_equals_single_file_and_pdb('1a0f__1__1.A_1.B__1.C')

    def test_split_equals_single_file_and_pdb_1a0t(self):
        """Same for a System with two Ligands whose residue name (GLC-F) does not fit PDB format."""
        self.assert_split_equals_single_file_and_pdb('1a0t__1__1.A_1.B__1.H_1.I')

    def test_long_residue_names(self):
        """Residue names longer than three characters are reported with their original name."""
        receptor, ligands = system_files('1a0t__1__1.A_1.B__1.H_1.I')
        xmlpath = os.path.join(self.tmp_dir.name, 'report.xml')
        arrays = profile_system(receptor=[receptor], ligands=ligands, xml=xmlpath)
        self.assertEqual(list(arrays['ligand_hetid']), ['GLC-F', 'GLC-F'])
        self.assertEqual(list(arrays['ligand_chain']), ['1.H', '1.I'])
        self.assertEqual(set(arrays['restype_lig']), {'GLC-F'})
        xml = et.parse(xmlpath)
        self.assertEqual([e.text for e in xml.findall('.//identifiers/hetid')], ['GLC-F', 'GLC-F'])
        self.assertEqual({e.text for e in xml.findall('.//restype_lig')}, {'GLC-F'})
        self.assertTrue({e.text for e in xml.findall('.//reschain')} <= {'1.A', '1.B'})

    def test_split_equals_single_file_and_pdb_4dlu(self):
        """Same for a System with a Ligand file containing a single metal ion."""
        self.assert_split_equals_single_file_and_pdb('4dlu__2__2.A_3.A__3.B_3.D')

    def test_metal_ion_ligand(self):
        """A ligand file with a single metal ion is a Ligand of type ION with metal complexes."""
        receptor, ligands = system_files('4dlu__2__2.A_3.A__3.B_3.D')
        arrays = profile_system(receptor=[receptor], ligands=ligands)
        self.assertEqual(list(arrays['ligand_hetid']), ['GNP', 'MG'])
        self.assertEqual(list(arrays['ligand_type']), ['SMALLMOLECULE', 'ION'])
        self.assertEqual(list(arrays['ligand_file']), [1, 2])
        metal = arrays['interaction_type'] == INTERACTION_TYPES.index('metal')
        self.assertGreater(metal.sum(), 0)
        self.assertTrue(np.all(arrays['interaction_ligand'][metal] == 1))
        metal_pairs = arrays['pairs'][arrays['pairs'][:, 4] == INTERACTION_TYPES.index('metal')]
        self.assertTrue(np.all((metal_pairs[:, 2] == 2) & (metal_pairs[:, 3] == 0)))

    def test_no_ligands(self):
        """A System without Ligands yields no interactions, but can still be profiled between chains."""
        receptor, _ = system_files('4dlu__2__2.A_3.A__3.B_3.D')
        arrays = profile_system(receptor=[receptor])
        self.assertEqual(len(arrays['ligand_hetid']), 0)
        self.assertEqual(arrays['pairs'].shape, (0, 6))
        arrays = profile_system(receptor=[receptor], chains=[['2.A'], ['3.A']])
        self.assertGreater(len(arrays['interaction_type']), 0)

    def test_inter_chain_with_original_chain_ids(self):
        """Chains are selected by their original IDs; receptor and ligand side are the requested chains."""
        receptor, _ = system_files('1a0f__1__1.A_1.B__1.C')
        arrays = profile_system(receptor=[receptor], chains=[['1.A'], ['1.B']])
        self.assertEqual(list(arrays['ligand_type']), ['PEPTIDE'])
        self.assertEqual(list(arrays['ligand_chain']), ['1.B'])
        self.assertEqual(set(arrays['rec_chain']), {'1.A'})
        self.assertEqual(set(arrays['lig_chain']), {'1.B'})
        peptides = profile_system(receptor=[receptor], peptides=['1.B'])
        self.assertEqual(signature_from_arrays(arrays), signature_from_arrays(peptides))
        intra = profile_system(receptor=[receptor], intra='1.A')
        self.assertEqual(set(intra['reschain']) | set(intra['reschain_lig']), {'1.A'})
        regions = profile_system(receptor=[receptor], regions=[({'1.B': [(1, 201)]}, {'1.A': [(1, 201)]})])
        self.assertEqual(signature_from_arrays(arrays), signature_from_arrays(regions))

    def test_chain_collision(self):
        """Equal chain IDs in different files are renamed internally and are ambiguous for selection."""
        receptor1, _ = system_files('1a0f__1__1.A_1.B__1.C')
        receptor2, _ = system_files('1a0t__1__1.A_1.B__1.H_1.I')
        system = CIFSystem([receptor1, receptor2], ['receptor', 'receptor'])
        self.assertEqual([(e.file_idx, e.original) for e in system.chain_mapping],
                         [(0, '1.A'), (0, '1.B'), (1, '1.A'), (1, '1.B')])
        self.assertEqual(len({e.internal for e in system.chain_mapping}), 4)
        with self.assertRaises(ValueError):
            system.resolve_chain('1.A')
        with self.assertRaises(ValueError):
            system.resolve_chain('1.X')

    def test_npz_indices_point_to_input_atoms(self):
        """Atom indices in pairs refer to the atoms of the input files as read by biotite."""
        receptor, ligands = system_files('1a0f__1__1.A_1.B__1.C')
        npzpath = os.path.join(self.tmp_dir.name, 'system.npz')
        profile_system(receptor=[receptor], ligands=ligands, npz=npzpath)
        arrays = np.load(npzpath)
        files = list(arrays['files'])
        atoms = [pdbx.get_structure(pdbx.CIFFile.read(f), model=1, altloc='all') for f in files]
        pairs = arrays['pairs']
        self.assertGreater(len(pairs), 0)
        for rec_file, rec_atom, lig_file, lig_atom, itype, interaction_id in pairs:
            self.assertEqual(arrays['file_role'][lig_file], 'ligand')
            self.assertEqual(atoms[lig_file].chain_id[lig_atom], arrays['lig_chain'][
                np.flatnonzero((pairs[:, 2] == lig_file) & (pairs[:, 3] == lig_atom))[0]])
            if INTERACTION_TYPES[itype] in ('hydrophobic', 'hbond', 'halogen'):  # single atoms on both sides
                np.testing.assert_allclose(atoms[lig_file].coord[lig_atom], arrays['ligcoo'][interaction_id],
                                           atol=1e-3)
                np.testing.assert_allclose(atoms[rec_file].coord[rec_atom], arrays['protcoo'][interaction_id],
                                           atol=1e-3)
                self.assertEqual(atoms[rec_file].res_id[rec_atom], arrays['resnr'][interaction_id])

    def test_config_is_restored(self):
        """Profiling does not change the global PLIP settings."""
        receptor, _ = system_files('1a0f__1__1.A_1.B__1.C')
        profile_system(receptor=[receptor], chains=[['1.A'], ['1.B']], keepmod=True)
        self.assertIsNone(config.CHAINS)
        self.assertFalse(config.KEEPMOD)
        self.assertEqual(config.PEPTIDES, [])

    def test_invalid_arguments(self):
        receptor, ligands = system_files('1a0f__1__1.A_1.B__1.C')
        with self.assertRaises(ValueError):
            profile_system()
        with self.assertRaises(ValueError):
            profile_system(structure=receptor, receptor=[receptor])
        with self.assertRaises(ValueError):
            profile_system(receptor=[receptor], chains=[['1.A'], ['1.B']], intra='1.A')


class CIFCommandLineTest(unittest.TestCase):
    """Checks the command line options for mmCIF input."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_split_input(self):
        receptor, ligands = system_files('1a0f__1__1.A_1.B__1.C')
        exitcode = subprocess.call(f'{sys.executable} ../plipcmd.py --receptor {receptor} --ligand {" ".join(ligands)}'
                                   f' -x -t --npz --name sys -o {self.tmp_dir.name} -s', shell=True)
        self.assertEqual(exitcode, 0)
        self.assertEqual(sorted(os.listdir(self.tmp_dir.name)), ['sys.npz', 'sys.txt', 'sys.xml'])

    def test_single_file_input_with_chains(self):
        receptor, _ = system_files('1a0f__1__1.A_1.B__1.C')
        exitcode = subprocess.call(f'{sys.executable} ../plipcmd.py -f {receptor} --chains "[[1.A], [1.B]]" --npz'
                                   f' -o {self.tmp_dir.name} -s', shell=True)
        self.assertEqual(exitcode, 0)
        arrays = np.load(os.path.join(self.tmp_dir.name, 'receptor_model_0_report.npz'))
        self.assertEqual(list(arrays['ligand_chain']), ['1.B'])

    def test_invalid_options(self):
        receptor, ligands = system_files('1a0f__1__1.A_1.B__1.C')
        for options in [f'-f ./pdb/1acj.pdb --npz', f'--ligand {ligands[0]}', f'--receptor {receptor} -y']:
            exitcode = subprocess.call(f'{sys.executable} ../plipcmd.py {options} -o {self.tmp_dir.name} -s',
                                       shell=True, stderr=subprocess.DEVNULL)
            self.assertEqual(exitcode, 2, options)
