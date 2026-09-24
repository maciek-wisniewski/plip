# PLIP

Profiling of non-covalent interactions between a receptor and its ligands in 3D structures.

## Language

**System**:
A single unit of analysis: one Receptor together with its Ligand(s), regardless of whether it is supplied as one structure file or as several.
_Avoid_: complex, structure (when meaning the whole input)

**Receptor**:
The protein part of a System, made of one or more chains.
_Avoid_: protein (when meaning the role), target

**Ligand**:
A molecule whose interactions with the Receptor are profiled. When supplied as its own file, the Ligand is the entire content of that file, whether it is a small molecule or a peptide; a lone metal ion is a Ligand as well.
_Avoid_: hetero group (when meaning the role)

**Surroundings**:
Everything in the System other than the Ligand currently being profiled — including the other Ligands — exactly as it would be present in a single structure file.

**Split input**:
A System supplied as several files, each explicitly declared as a Receptor file or a Ligand file.
_Avoid_: multi-file mode, PLINDER mode

**Single-file input**:
A System supplied as one file containing both Receptor and Ligand(s); Ligands are detected the same way as for PDB input.

**Chain mapping**:
The record of how chain identifiers from the source files were renamed when merging a Split input, so that each chain in the System is unique and traceable to its source file.

**Inter-chain interaction**:
An interaction between two chains of a System, one of which is treated as the Ligand (e.g. protein–protein or protein–peptide).
_Avoid_: PPI (when the partner may be a peptide)

**Intra-chain interaction**:
An interaction between residues of the same chain.

## Relationships

- A **System** has exactly one **Receptor** and one or more **Ligands**
- A **Split input** yields exactly one **System** and exactly one **Chain mapping**
- Every chain in a **System** traces back to exactly one source file via the **Chain mapping**
- Each **Ligand** is profiled separately, against the **Receptor**, with the other **Ligands** present in the **Surroundings**
- A **Split input** and a **Single-file input** of the same structure describe the same **System** and must yield the same interactions
- In a **Split input**, each Ligand file is exactly one **Ligand** and is never discarded as an artifact, buffer or crystallisation additive; in a **Single-file input**, Ligands are detected and filtered as for PDB
- A **System** with no **Ligands** is still valid: it yields no ligand interactions but can be profiled for **Inter-chain** and **Intra-chain interactions**
- Users always name chains by their original identifiers; the **Chain mapping** is internal and only surfaces when an identifier is ambiguous

## Flagged ambiguities

- "structure file" vs "System": one System may come from many files; resolved — the System is the unit, files are only its sources.
