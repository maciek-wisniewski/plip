"""
Reading of mmCIF input with biotite and conversion into PDB text for the OpenBabel-based pipeline.

A System is either a Single-file input (one mmCIF file with receptor and ligands) or a Split input
(receptor files and ligand files). All files are merged into one PDB text with single-character chain IDs
and residue names of at most three characters. The IdentifierMap translates these internal identifiers back
to the original ones (see docs/adr/0001-mmcif-read-with-biotite-routed-through-pdb-text.md).
"""
import gzip
import re
import string
import warnings
from collections import namedtuple

import numpy as np
import biotite.structure.io.pdbx as pdbx

from plip.basic import config, logger

logger = logger.get_logger()

CIF_EXTENSIONS = ('.cif', '.cif.gz', '.mmcif', '.mmcif.gz')
CHAIN_POOL = string.ascii_uppercase + string.ascii_lowercase + string.digits
VALID_RESNAME = re.compile(r'^[A-Za-z0-9]{1,3}$')

ChainMappingEntry = namedtuple('ChainMappingEntry', 'file_idx original internal')
Cofactor = namedtuple('Cofactor', 'file_idx chain resname resnr atom_idx')


def is_cif(path):
    """Checks by file extension whether the given path is an mmCIF file."""
    return path.lower().endswith(CIF_EXTENSIONS)


def read_cif(path, model=1):
    """Reads one model of an mmCIF file as a biotite AtomArray, including all alternate locations.
    Chains and residues come from the auth_* fields, the label_* fields are only used if auth_* are missing.
    The position of an atom in the returned array is its atom index within the file."""
    if path.lower().endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            cif = pdbx.CIFFile.read(f)
    else:
        cif = pdbx.CIFFile.read(path)
    atom_site = cif.block['atom_site']
    use_author_fields = 'auth_asym_id' in atom_site and 'auth_seq_id' in atom_site
    models = np.unique(atom_site['pdbx_PDB_model_num'].as_array(int)) if 'pdbx_PDB_model_num' in atom_site else [1]
    if model not in models:
        logger.warning(f'invalid model number specified for {path}, using first model instead')
        model = int(models[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # missing optional fields (e.g. charges) are expected
        atoms = pdbx.get_structure(cif, model=model, altloc='all', use_author_fields=use_author_fields,
                                   extra_fields=['b_factor', 'occupancy', 'charge'])
    return atoms


class IdentifierMap:
    """Translates the internal single-character chain IDs and residue name aliases back to the original ones."""

    def __init__(self, chains, resnames):
        self.chains = chains  # internal chain character -> original chain ID
        self.resnames = resnames  # internal residue name alias -> original residue name

    def chain(self, internal):
        return self.chains.get(internal, internal)

    def resname(self, internal):
        return self.resnames.get(internal, internal)


class CIFSystem:
    """A System read from mmCIF files and merged into PDB text."""

    def __init__(self, files, roles, model=1, keep_hydrogens=False):
        self.files = list(files)
        self.roles = list(roles)  # 'complex', 'receptor' or 'ligand' for each file
        self.atoms = [read_cif(f, model) for f in self.files]
        self.keep_hydrogens = keep_hydrogens
        for path, atoms in zip(self.files, self.atoms):
            if atoms.array_length() == 0:
                raise ValueError(f'no atoms found in {path}')
        self.chain_mapping = self._map_chains()
        self._internal_chain = {(e.file_idx, e.original): e.internal for e in self.chain_mapping}
        self.resname_aliases = self._alias_resnames()
        self.cofactors = self._find_cofactors()
        self.identifier_map = IdentifierMap(
            chains={e.internal: e.original for e in self.chain_mapping},
            resnames={alias: name for name, alias in self.resname_aliases.items()})
        self.pdb_text, self.serial_file, self.serial_atom = self._to_pdb()

    @property
    def is_split(self):
        return 'complex' not in self.roles

    def _map_chains(self):
        """Assigns a unique single-character chain ID to each chain of each file. Original single-character IDs
        which occur in only one file are kept, all others get the next free character."""
        chains = []  # (file_idx, original chain) in order of appearance
        for file_idx, atoms in enumerate(self.atoms):
            for chain in dict.fromkeys(atoms.chain_id):
                chains.append((file_idx, str(chain)))
        if len(chains) > len(CHAIN_POOL):
            raise ValueError(f'the system contains {len(chains)} chains, at most {len(CHAIN_POOL)} are supported')
        originals = [chain for _, chain in chains]
        keep = {chain for chain in originals if chain in CHAIN_POOL and originals.count(chain) == 1}
        free = iter([c for c in CHAIN_POOL if c not in keep])
        mapping = [ChainMappingEntry(file_idx, chain, chain if chain in keep else next(free))
                   for file_idx, chain in chains]
        for entry in mapping:
            if entry.internal != entry.original:
                logger.debug(f'chain {entry.original} of {self.files[entry.file_idx]} renamed to {entry.internal}')
        return mapping

    def _alias_resnames(self):
        """Assigns three-character aliases to residue names which do not fit into PDB format."""
        names = set()
        for atoms in self.atoms:
            names.update(str(n) for n in np.unique(atoms.res_name))
        aliases = {}
        candidates = (f'Z{a}{b}' for a in string.digits + string.ascii_uppercase
                      for b in string.digits + string.ascii_uppercase)
        for name in sorted(n for n in names if not VALID_RESNAME.match(n)):
            alias = next(c for c in candidates if c not in names)
            aliases[name] = alias
            logger.debug(f'residue name {name} is represented as {alias}')
        return aliases

    def _find_cofactors(self):
        """Ligand files consisting of exactly one metal atom are Cofactors instead of Ligands."""
        cofactors = []
        for file_idx, atoms in enumerate(self.atoms):
            if self.roles[file_idx] != 'ligand':
                continue
            heavy = atoms[atoms.element != 'H']
            if heavy.array_length() == 1 and heavy.element[0].upper() in config.METAL_IONS:
                atom_idx = int(np.flatnonzero(atoms.element != 'H')[0])
                cofactors.append(Cofactor(file_idx=file_idx, chain=str(heavy.chain_id[0]),
                                          resname=str(heavy.res_name[0]), resnr=int(heavy.res_id[0]),
                                          atom_idx=atom_idx))
        return cofactors

    @property
    def ligand_groups(self):
        """For a Split input, the internal chain IDs of each Ligand file (Cofactors excluded)."""
        cofactor_files = {c.file_idx for c in self.cofactors}
        return [(file_idx, {self._internal_chain[(file_idx, str(c))] for c in np.unique(atoms.chain_id)})
                for file_idx, atoms in enumerate(self.atoms)
                if self.roles[file_idx] == 'ligand' and file_idx not in cofactor_files]

    def internal_chains(self, original):
        """Returns all internal chain IDs for a chain ID as given in the input files."""
        return [e.internal for e in self.chain_mapping if e.original == original]

    def resolve_chain(self, original):
        """Translates a chain ID as given in the input files to the internal chain ID."""
        internal = self.internal_chains(original)
        if len(internal) == 1:
            return internal[0]
        mapping = ', '.join(f'{e.original} ({self.files[e.file_idx]}) -> {e.internal}' for e in self.chain_mapping)
        if not internal:
            raise ValueError(f'chain {original} does not exist in the input files. Chain mapping: {mapping}')
        raise ValueError(f'chain {original} occurs in several input files and is ambiguous. Chain mapping: {mapping}')

    def _to_pdb(self):
        """Writes all files as one PDB text. Serial numbers are consecutive and without TER records, so the
        serial number of an atom equals its OpenBabel index. Returns the text and, for each serial number - 1,
        the index of the source file and the atom index within that file."""
        lines, serial_file, serial_atom = [], [], []
        serial = 0
        for file_idx, atoms in enumerate(self.atoms):
            for atom_idx in range(atoms.array_length()):
                element = str(atoms.element[atom_idx]).upper()
                if element == 'H' and not self.keep_hydrogens:
                    continue  # polar hydrogens are added by OpenBabel, as for PDB input
                serial += 1
                if serial > 99999:
                    raise ValueError('systems with more than 99999 atoms are not supported')
                resnr = int(atoms.res_id[atom_idx])
                if not -999 <= resnr <= 9999:
                    raise ValueError(f'residue number {resnr} in {self.files[file_idx]} does not fit PDB format')
                lines.append(self._pdb_line(atoms, atom_idx, serial, file_idx, element))
                serial_file.append(file_idx)
                serial_atom.append(atom_idx)
        lines.append('END\n')
        return ''.join(lines), np.array(serial_file, dtype=np.int32), np.array(serial_atom, dtype=np.int32)

    def _pdb_line(self, atoms, i, serial, file_idx, element):
        record = 'HETATM' if atoms.hetero[i] else 'ATOM'
        name = str(atoms.atom_name[i])
        name = f' {name:<3}' if len(name) < 4 and len(element) == 1 else f'{name:<4}'
        altloc = str(atoms.altloc_id[i])
        altloc = ' ' if altloc in ('.', '?', '') else altloc[0]
        resname = str(atoms.res_name[i])
        resname = self.resname_aliases.get(resname, resname)
        chain = self._internal_chain[(file_idx, str(atoms.chain_id[i]))]
        ins_code = str(atoms.ins_code[i])[:1] or ' '
        x, y, z = atoms.coord[i]
        occupancy = float(atoms.occupancy[i])
        occupancy = 1.0 if np.isnan(occupancy) else occupancy
        b_factor = float(atoms.b_factor[i])
        b_factor = 0.0 if np.isnan(b_factor) else b_factor
        charge = int(atoms.charge[i])
        charge = f'{abs(charge)}{"+" if charge > 0 else "-"}' if charge != 0 else '  '
        return (f'{record:<6}{serial:>5} {name:4}{altloc}{resname:>3} {chain}{int(atoms.res_id[i]):>4}{ins_code}   '
                f'{x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}          {element:>2}{charge}\n')

    def source_of(self, serial):
        """Returns the source file index and atom index within the file for a PDB serial number."""
        return int(self.serial_file[serial - 1]), int(self.serial_atom[serial - 1])

    def original_chain_of(self, serial):
        file_idx, atom_idx = self.source_of(serial)
        return str(self.atoms[file_idx].chain_id[atom_idx])
