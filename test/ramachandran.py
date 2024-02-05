from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral
import matplotlib.pyplot as plt

def compute_phi_psi(structure):
    for model in structure:
        for chain in model:
            polypeptides = list(chain.get_residues())
            for i, residue in enumerate(polypeptides):
                
                if i > 0 and i < len(polypeptides) - 1:
                    prev_residue = polypeptides[i-1]
                    next_residue = polypeptides[i+1]

                    if "C" in prev_residue \
                        and "N" in residue and "CA" in residue and "C" in residue \
                        and "N" in next_residue:

                        phi = calc_dihedral(
                            prev_residue["C"].get_vector(),
                            residue["N"].get_vector(),
                            residue["CA"].get_vector(),
                            residue["C"].get_vector()
                        ) * 57.3

                        psi = calc_dihedral(
                            residue["N"].get_vector(),
                            residue["CA"].get_vector(),
                            residue["C"].get_vector(),
                            next_residue["N"].get_vector()
                        ) * 57.3

                    yield phi, psi

pdb_parser = PDBParser()
structure = pdb_parser.get_structure("3b8e", "3b8e.pdb")

angles = []
for phi, psi in compute_phi_psi(structure):
    angles.append((phi, psi))

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(
    [phi for phi, psi in angles],
    [psi for phi, psi in angles],
    s=1,
    c="black"
)
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\psi$")
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_yticks([-180, -90, 0, 90, 180])
ax.grid()
plt.tight_layout()
ax.set_title("Ramachandran plot of 3b8e")
plt.savefig("3b8e_ramachandran.png", dpi=300)