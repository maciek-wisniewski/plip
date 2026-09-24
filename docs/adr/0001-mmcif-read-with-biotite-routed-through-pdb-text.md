# mmCIF is read with biotite and routed through PDB text into the existing OpenBabel pipeline

mmCIF input (a single file, or a Split input of receptor and ligand files) is read and merged with biotite. The result is written as temporary PDB text, with chain IDs renamed to single characters and residue names longer than three characters given 3-character aliases. That text goes through the existing `PDBParser` and pybel path, and OpenBabel does all of the chemistry, exactly as it does for PDB input. Reports translate the aliases back to the original chain IDs and residue names.

We rejected three alternatives:
- **Reading mmCIF with OpenBabel directly.** Tested on PLINDER files, the reader drops chain IDs (it re-letters them A, B, …), sets the HETATM flag to false everywhere, sets the side-chain flag on every atom, adds no implicit hydrogens (so there is no aromaticity and no polar H), ignores `_chem_comp_bond` and charges, and silently fails on single-atom files.
- **Building an OBMol directly from biotite arrays.** This would keep bond orders from the file, but it means re-implementing what OpenBabel's PDB reader and PLIP's `PDBParser` do, which risks subtle divergence from PDB results.
- **Rewriting the core on biotite.** This amounts to a rewrite of PLIP.

## Consequences

- Bond orders from `_chem_comp_bond` are discarded and OpenBabel perceives them again, the same as for PDB input.
- A System can hold at most 62 chains, the number of single characters available for a PDB chain ID.
- A CIF file and a PDB file of the same structure yield identical interactions, and PDB input serves as the test oracle.
