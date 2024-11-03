import pymol
from pymol import cmd, stored
import numpy as np
import os

def visualize_smotif(pdb_file, chain_id, sse1_start, sse1_end, sse2_start, sse2_end):
    # Initialize PyMOL
    pymol.finish_launching()

    # Load the PDB file
    cmd.load(pdb_file, "protein")

    # Select and color the SSEs
    cmd.select("sse1", f"chain {chain_id} and resi {sse1_start}-{sse1_end}")
    cmd.select("sse2", f"chain {chain_id} and resi {sse2_start}-{sse2_end}")
    cmd.color("red", "sse1")
    cmd.color("blue", "sse2")

    # Get coordinates
    stored.sse1_coords = []
    cmd.iterate_state(1, "sse1 and name CA", "stored.sse1_coords.append([x,y,z])")
    stored.sse2_coords = []
    cmd.iterate_state(1, "sse2 and name CA", "stored.sse2_coords.append([x,y,z])")

    sse1_coords = np.array(stored.sse1_coords)
    sse2_coords = np.array(stored.sse2_coords)

    # Calculate vectors
    M1 = calculate_moment_of_inertia(sse1_coords)
    M2 = calculate_moment_of_inertia(sse2_coords)
    L = sse2_coords[0] - sse1_coords[-1]
    L_norm = L / np.linalg.norm(L)

    # Calculate D, delta, theta, and rho
    D = L
    delta = np.arccos(np.clip(np.dot(L_norm, M1), -1.0, 1.0))
    theta = np.arccos(np.clip(np.dot(M1, M2), -1.0, 1.0))

    # Calculate plane P and its normal
    normal_P = np.cross(M1, L_norm)
    normal_P = normal_P / np.linalg.norm(normal_P)

    # Calculate plane C (perpendicular to both M1 and normal_P)
    normal_C = np.cross(M1, normal_P)
    normal_C = normal_C / np.linalg.norm(normal_C)

    # Calculate ρ (meridian): angle between M2 and plane C
    rho = 90 - np.degrees(np.arccos(np.clip(np.dot(M2, normal_P), -1.0, 1.0)))

    # Calculate points for visualization
    sse1_end_point = sse1_coords[-1]
    sse2_start_point = sse2_coords[0]
    sse1_mid = np.mean(sse1_coords, axis=0)
    sse2_mid = np.mean(sse2_coords, axis=0)

    # Create vector D
    create_vector("D", sse1_end_point, sse2_start_point, "yellow")

    # Create vectors M1 and M2
    scale = 5.0  # Adjust this value to change the length of M1 and M2 vectors
    create_vector("M1", sse1_mid, sse1_mid + scale * M1, "orange")
    create_vector("M2", sse2_mid, sse2_mid + scale * M2, "purple")

    # Create arcs for delta, theta, and rho
    create_arc("delta", sse1_end_point, L_norm, M1, "cyan", 3.0)
    create_arc("theta", sse1_mid, M1, M2, "magenta", 4.0)
    create_arc("rho", sse2_mid, M2, normal_P, "green", 5.0)

    # Label vectors and arcs
    label_vector("D", (sse1_end_point + sse2_start_point) / 2)
    label_vector("M1", sse1_mid + scale * M1)
    label_vector("M2", sse2_mid + scale * M2)
    label_arc("delta", sse1_end_point, L_norm, M1, 3.0)
    label_arc("theta", sse1_mid, M1, M2, 4.0)
    label_arc("rho", sse2_mid, M2, normal_P, 5.0)

    # Set view
    cmd.zoom("all")
    cmd.center("all")

def create_vector(name, start, end, color):
    cmd.pseudoatom(name + "_start", pos=tuple(start))
    cmd.pseudoatom(name + "_end", pos=tuple(end))
    cmd.distance(name, name + "_start", name + "_end")
    cmd.color(color, name)
    cmd.hide("labels", name)
    cmd.hide("spheres", name + "_start")
    cmd.hide("spheres", name + "_end")

def create_arc(name, center, vec1, vec2, color, radius):
    cmd.pseudoatom(name + "_center", pos=tuple(center))
    cmd.pseudoatom(name + "_start", pos=tuple(center + radius * vec1))
    cmd.pseudoatom(name + "_end", pos=tuple(center + radius * vec2))
    cmd.angle(name, name + "_start", name + "_center", name + "_end")
    cmd.color(color, name)
    cmd.hide("labels", name)
    cmd.hide("spheres", name + "_center")
    cmd.hide("spheres", name + "_start")
    cmd.hide("spheres", name + "_end")

def label_vector(name, pos):
    cmd.pseudoatom(f"label_{name}", pos=tuple(pos))
    cmd.label(f"label_{name}", f"'{name}'")

def label_arc(name, center, vec1, vec2, radius):
    mid_vec = (vec1 + vec2) / 2
    mid_vec = mid_vec / np.linalg.norm(mid_vec)
    label_pos = center + (radius + 0.5) * mid_vec
    cmd.pseudoatom(f"label_{name}", pos=tuple(label_pos))
    cmd.label(f"label_{name}", f"'{name}'")

def calculate_moment_of_inertia(coords):
    coords = np.array(coords)
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid
    inertia_tensor = np.dot(centered_coords.T, centered_coords)
    eigenvalues, eigenvectors = np.linalg.eig(inertia_tensor)
    return eigenvectors[:, np.argmax(eigenvalues)]

# Example usage
pdb_file = "/home/kalabharath/projects/dingo_fold/cath_db/non-redundant-data-sets/dompdb/1h65A00.pdb"
chain_id = "A"
sse1_start = 19
sse1_end = 35
sse2_start = 39
sse2_end = 43

visualize_smotif(pdb_file, chain_id, sse1_start, sse1_end, sse2_start, sse2_end)

# Keep the PyMOL session open
pymol.cmd.mplay()