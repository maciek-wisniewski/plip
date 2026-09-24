"""
Interactions of a System as numpy arrays (written as NPZ file), with atoms indexed per input file.

Every interaction is expanded into atom pairs in `pairs` with the columns
[rec_file, rec_atom, lig_file, lig_atom, type, interaction_id]. Files index into `files`, atoms are
positions within that file (see plip.structure.cif.read_cif), type indexes into `interaction_types`.
Interactions between groups of atoms (rings, charged groups) contribute all pairs of their atoms.
Per-interaction features are arrays indexed by interaction_id; missing values are NaN, -1 or ''.
"""
import numpy as np

INTERACTION_TYPES = ('hydrophobic', 'hbond', 'waterbridge', 'saltbridge', 'pistacking', 'pication', 'halogen',
                     'metal')

FLOAT_FEATURES = ('dist', 'centdist', 'dist_h_a', 'dist_d_a', 'dist_a_w', 'dist_d_w', 'don_angle', 'acc_angle',
                  'water_angle', 'angle', 'offset', 'rms')
INT_FEATURES = ('resnr', 'resnr_lig', 'sidechain', 'protisdon', 'protispos', 'protcharged', 'coordination',
                'complexnum', 'water_file', 'water_atom')
STR_FEATURES = ('restype', 'reschain', 'restype_lig', 'reschain_lig', 'donortype', 'acceptortype', 'stacking_type',
                'lig_group', 'metal_type', 'target_type', 'location', 'geometry')
COORD_FEATURES = ('ligcoo', 'protcoo', 'watercoo')


def _interactions(pli):
    """Yields (type, receptor-side serials, ligand-side serials, features) for all interactions of a binding site.
    For metal complexes the metal is on the ligand side and the coordinated atom on the receptor side."""
    for h in pli.hydrophobic_contacts:
        yield 'hydrophobic', [h.bsatom_orig_idx], [h.ligatom_orig_idx], dict(
            dist=h.distance, ligcoo=h.ligatom.coords, protcoo=h.bsatom.coords)
    for hb in pli.hbonds_pdon + pli.hbonds_ldon:
        rec, lig = (hb.d_orig_idx, hb.a_orig_idx) if hb.protisdon else (hb.a_orig_idx, hb.d_orig_idx)
        ligatom, protatom = (hb.a, hb.d) if hb.protisdon else (hb.d, hb.a)
        yield 'hbond', [rec], [lig], dict(
            sidechain=hb.sidechain, dist_h_a=hb.distance_ah, dist_d_a=hb.distance_ad, don_angle=hb.angle,
            protisdon=hb.protisdon, donortype=hb.dtype, acceptortype=hb.atype,
            ligcoo=ligatom.coords, protcoo=protatom.coords)
    for wb in pli.water_bridges:
        rec, lig = (wb.d_orig_idx, wb.a_orig_idx) if wb.protisdon else (wb.a_orig_idx, wb.d_orig_idx)
        ligatom, protatom = (wb.a, wb.d) if wb.protisdon else (wb.d, wb.a)
        yield 'waterbridge', [rec], [lig], dict(
            dist_a_w=wb.distance_aw, dist_d_w=wb.distance_dw, don_angle=wb.d_angle, water_angle=wb.w_angle,
            protisdon=wb.protisdon, donortype=wb.dtype, acceptortype=wb.atype, water_serial=wb.water_orig_idx,
            ligcoo=ligatom.coords, protcoo=protatom.coords, watercoo=wb.water.coords)
    for sb in pli.saltbridge_lneg + pli.saltbridge_pneg:
        prot, lig = (sb.positive, sb.negative) if sb.protispos else (sb.negative, sb.positive)
        yield 'saltbridge', prot.atoms_orig_idx, lig.atoms_orig_idx, dict(
            dist=sb.distance, protispos=sb.protispos, lig_group=lig.fgroup,
            ligcoo=lig.center, protcoo=prot.center)
    for st in pli.pistacking:
        yield 'pistacking', st.proteinring.atoms_orig_idx, st.ligandring.atoms_orig_idx, dict(
            centdist=st.distance, angle=st.angle, offset=st.offset, stacking_type=st.type,
            ligcoo=st.ligandring.center, protcoo=st.proteinring.center)
    for pc in pli.pication_laro + pli.pication_paro:
        prot, lig = (pc.charge, pc.ring) if pc.protcharged else (pc.ring, pc.charge)
        yield 'pication', prot.atoms_orig_idx, lig.atoms_orig_idx, dict(
            dist=pc.distance, offset=pc.offset, protcharged=pc.protcharged,
            lig_group='Aromatic' if pc.protcharged else pc.charge.fgroup,
            ligcoo=lig.center, protcoo=prot.center)
    for hal in pli.halogen_bonds:
        yield 'halogen', [hal.acc_orig_idx], [hal.don_orig_idx], dict(
            sidechain=hal.sidechain, dist=hal.distance, don_angle=hal.don_angle, acc_angle=hal.acc_angle,
            donortype=hal.donortype, acceptortype=hal.acctype,
            ligcoo=hal.don.x.coords, protcoo=hal.acc.o.coords)
    for m in pli.metal_complexes:
        yield 'metal', [m.target_orig_idx], [m.metal_orig_idx], dict(
            metal_type=m.metal_type, target_type=m.target_type, coordination=m.coordination_num, dist=m.distance,
            location=m.location, rms=m.rms, geometry=m.geometry, complexnum=m.complexnum,
            ligcoo=m.metal.coords, protcoo=m.target.atom.coords)


def interaction_arrays(mol, system):
    """Builds all arrays for a profiled PDBComplex `mol` that was loaded from the CIFSystem `system`."""
    idmap = system.identifier_map
    pairs, rec_chain, lig_chain = [], [], []
    features = []
    lig_rows = []
    ligand_files = {e.internal: e.file_idx for e in system.chain_mapping if system.roles[e.file_idx] == 'ligand'}

    for ligand_idx, site in enumerate(sorted(mol.interaction_sets)):
        pli = mol.interaction_sets[site]
        lig = pli.ligand
        lig_rows.append(dict(
            hetid=idmap.resname(lig.hetid), chain=idmap.chain(lig.chain), resnr=lig.position,
            longname='-'.join(idmap.resname(member[0]) for member in lig.members), type=lig.type,
            file=ligand_files.get(lig.chain, -1), smiles=lig.smiles, inchikey=lig.inchikey.strip()))
        for itype, rec_serials, lig_serials, feats in _interactions(pli):
            interaction_id = len(features)
            feats = dict(feats)
            feats['type'] = INTERACTION_TYPES.index(itype)
            feats['ligand'] = ligand_idx
            water_serial = feats.pop('water_serial', None)
            if water_serial is not None:
                feats['water_file'], feats['water_atom'] = system.source_of(water_serial)
            rec_file, rec_atom = system.source_of(rec_serials[0])
            lig_file, lig_atom = system.source_of(lig_serials[0])
            feats.update(_residue_features(system, rec_file, rec_atom, lig_file, lig_atom))
            features.append(feats)
            for rs in rec_serials:
                for ls in lig_serials:
                    rf, ra = system.source_of(rs)
                    lf, la = system.source_of(ls)
                    pairs.append((rf, ra, lf, la, feats['type'], interaction_id))
                    rec_chain.append(system.original_chain_of(rs))
                    lig_chain.append(system.original_chain_of(ls))

    arrays = {
        'pairs': np.array(pairs, dtype=np.int32).reshape(-1, 6),
        'rec_chain': _str_array(rec_chain),
        'lig_chain': _str_array(lig_chain),
        'files': _str_array(system.files),
        'file_role': _str_array(system.roles),
        'interaction_types': _str_array(INTERACTION_TYPES),
        'interaction_type': np.array([f['type'] for f in features], dtype=np.int8),
        'interaction_ligand': np.array([f['ligand'] for f in features], dtype=np.int32),
    }
    for name in FLOAT_FEATURES:
        arrays[name] = np.array([float(f.get(name, np.nan)) for f in features], dtype=np.float64)
    for name in INT_FEATURES:
        arrays[name] = np.array([int(f.get(name, -1)) for f in features], dtype=np.int32)
    for name in STR_FEATURES:
        arrays[name] = _str_array([str(f.get(name, '')) for f in features])
    for name in COORD_FEATURES:
        arrays[name] = np.array([f.get(name, (np.nan, np.nan, np.nan)) for f in features],
                                dtype=np.float64).reshape(-1, 3)

    for key, dtype in [('hetid', str), ('chain', str), ('resnr', np.int32), ('longname', str), ('type', str),
                       ('file', np.int32), ('smiles', str), ('inchikey', str)]:
        values = [row[key] for row in lig_rows]
        arrays[f'ligand_{key}'] = _str_array(values) if dtype is str else np.array(values, dtype=dtype)

    arrays['chain_mapping_file'] = np.array([e.file_idx for e in system.chain_mapping], dtype=np.int32)
    arrays['chain_mapping_original'] = _str_array([e.original for e in system.chain_mapping])
    arrays['chain_mapping_internal'] = _str_array([e.internal for e in system.chain_mapping])
    return arrays


def _residue_features(system, rec_file, rec_atom, lig_file, lig_atom):
    """Residue of the first receptor-side and ligand-side atom of an interaction, with original identifiers."""
    rec, lig = system.atoms[rec_file], system.atoms[lig_file]
    return dict(resnr=int(rec.res_id[rec_atom]), restype=str(rec.res_name[rec_atom]),
                reschain=str(rec.chain_id[rec_atom]), resnr_lig=int(lig.res_id[lig_atom]),
                restype_lig=str(lig.res_name[lig_atom]), reschain_lig=str(lig.chain_id[lig_atom]))


def _str_array(values):
    values = [str(v) for v in values]
    return np.array(values, dtype=f'<U{max([len(v) for v in values] + [1])}')
