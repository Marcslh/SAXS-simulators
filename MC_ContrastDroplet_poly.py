#!/usr/bin/env python3
"""
Monte Carlo scattering model of PDB proteins on an oil droplet surface.

Version with
  * droplet-size polydispersity (explicit ensemble of droplets, incoherent average)
  * constant protein surface coverage (N_proteins scales with droplet area)
  * safe handling of N_proteins = 0 (bare oil droplet)
  * faster oil/protein cross term (radial histogram) and vectorised Debye sum

Polydispersity model (dilute emulsion, no droplet-droplet S(q)):
    I_poly(q) = sum_i w_i * I_i(q)             (number-averaged, per droplet)
    I_vol(q)  = sum_i w_i * I_i(q) / sum_i w_i * V_i   (per unit oil volume)
where I_i(q) is the full MC calculation for a droplet of radius R_i carrying
N_i = round(Gamma * 4*pi*R_i^2) proteins.
"""

import numpy as np
import matplotlib.pyplot as plt

# ================= USER PARAMETERS =================
PDB_FILE = "lys.pdb"
Ro = 1000.0                        # droplet radius (Å); MEAN radius if POLYDISPERSE = True
N_proteins = 5000                    # proteins on a droplet of radius Ro (0 is allowed = bare oil droplet)

penetration_min = 0.2
penetration_max = 0.8

rho_oil = -0.06
rho_protein = 0.12
rho_water = 0.0

USE_OIL_SPHERE = True
rho_oil_bulk = -0.06

# ===== SIZE POLYDISPERSITY =====
POLYDISPERSE = True               # False = single droplet radius Ro (original behaviour)
SIZE_DISTRIBUTION = 'lognormal'   # 'lognormal' or 'gaussian' (number-weighted, mean = Ro)
SIZE_PDI = 0.1                   # relative width sigma_R / <R>  (0.10-0.20 typical for emulsions)
N_RADII = 25                      # number of radii in the grid (>=2). The averaged curve is free of
                                  # aliasing ripples only up to q ~ pi/dR_max - 1.5/sigma_R (printed at start)
N_SIGMA = 2.5                     # grid spans +/- N_SIGMA standard deviations

# ===== CONSTANT SURFACE COVERAGE =====
# Proteins per nm^2 of droplet surface. N_i = round(SURFACE_DENSITY * 4*pi*R_i^2 [nm^2]).
# None -> derived from N_proteins at the mean radius: Gamma = N_proteins / (4*pi*Ro^2).
# (Example: ~2-3 mg/m^2 of a 66 kDa protein is ~0.018-0.027 proteins/nm^2.)
SURFACE_DENSITY = None

RANDOM_SEED = None                # integer for reproducible runs, None for random
VERBOSE = False                   # True = detailed per-droplet printout

# ===== PROTEIN ORIENTATION CONTROL =====
ALIGNED_PROTEINS = False           # True = all proteins aligned, False = random orientations
ALIGNMENT_AXIS = 'radial'          # 'radial' = point outward, 'tangential' = lie flat, 'custom' = specify vector
CUSTOM_ALIGNMENT = [0, 0, 1]       # Used if ALIGNMENT_AXIS = 'custom'
ALIGNMENT_NOISE_DEGREES = 0.0      # Angular spread around alignment (0 = perfect, 15 = slight, 45 = high disorder)

# ===== PROTEIN AXIS SELECTION =====
PROTEIN_ALIGNMENT_AXIS = 'z'       # Which protein axis aligns with target: 'x', 'y', or 'z'
                                   # 'z' = protein Z-axis points radially (X,Y are tangential), etc.
CONTROL_SPIN = False               # True = fix rotation around alignment axis, False = random
SPIN_ANGLE_DEGREES = 0.0           # If CONTROL_SPIN=True, rotation angle around alignment axis (0-360°)

# ===== THERMAL/EQUILIBRATION PERTURBATION =====
ADD_THERMAL_NOISE = False          # Add small random displacements to protein positions
THERMAL_NOISE_AMPLITUDE = 2.0      # RMS displacement in Angstroms (typical: 1-5 Å)
PERTURB_PENETRATION = False        # Also randomize penetration depth slightly
PENETRATION_NOISE_AMPLITUDE = 0.1  # Fraction of R_protein (0.1 = ±10% variation)

qmin, qmax, qN = 0.002, 0.45, 2000
dr_histogram = 1.0
max_pairs_sample = 100_000_000     # per droplet; noise averages out over the N_RADII droplets, so this can be lower than for one droplet
dr_cross = 0.25                    # bin width (Å) of the radial histogram used for the oil-protein cross term

N_oil_vis_points = 20000
proteins_vis_subsample = 5
VISUALIZE_CUTOUT = True
cutout_angle_deg = 40

OUT_PREFIX = "pdb_droplet"


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


# ================= PDB LOADING =================
def load_pdb(filename):
    positions = []
    elements = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    elem = line[76:78].strip()
                    if not elem:
                        elem = line[12:16].strip()[0]

                    positions.append([x, y, z])
                    elements.append(elem)
                except (ValueError, IndexError):
                    continue

    positions = np.array(positions)
    positions -= np.mean(positions, axis=0)

    print(f"Loaded {len(positions)} atoms from {filename}")
    print(f"Protein bounding box: {np.ptp(positions, axis=0)} Å")

    return positions, elements


def get_protein_radius(positions):
    distances = np.linalg.norm(positions, axis=1)
    return np.max(distances)


# ================= GEOMETRY =================
def random_unit_vectors(N):
    v = np.random.normal(size=(N, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def random_rotation_matrix():
    """Uniformly random 3D rotation matrix from a random unit quaternion."""
    u = np.random.uniform(0, 1, 3)
    q0 = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
    q1 = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
    q2 = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
    q3 = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])

    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R


def aligned_rotation_matrix(radial_direction, alignment_type='radial',
                            custom_axis=None, noise_degrees=0.0,
                            protein_axis='z', control_spin=False, spin_angle_deg=0.0):
    """
    Rotation matrix that aligns a chosen protein axis with a target direction
    (radial / tangential / custom), with optional angular noise and spin control.
    """
    radial = radial_direction / np.linalg.norm(radial_direction)

    # Determine target alignment direction
    if alignment_type == 'radial':
        target = radial.copy()
    elif alignment_type == 'tangential':
        if abs(radial[2]) < 0.9:
            arbitrary = np.array([0, 0, 1])
        else:
            arbitrary = np.array([1, 0, 0])
        target = np.cross(radial, arbitrary)
        target = target / np.linalg.norm(target)
    elif alignment_type == 'custom':
        if custom_axis is None:
            raise ValueError("custom_axis must be provided when alignment_type='custom'")
        target = np.array(custom_axis) / np.linalg.norm(custom_axis)
    else:
        raise ValueError(f"Unknown alignment_type: {alignment_type}")

    # Add angular noise if requested
    if noise_degrees > 0:
        noise_rad = np.deg2rad(noise_degrees)

        random_perp = np.random.randn(3)
        random_perp -= np.dot(random_perp, target) * target
        if np.linalg.norm(random_perp) > 1e-10:
            random_perp /= np.linalg.norm(random_perp)

            angle = np.random.normal(0, noise_rad)

            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            target = (target * cos_a +
                      np.cross(random_perp, target) * sin_a +
                      random_perp * np.dot(random_perp, target) * (1 - cos_a))
            target /= np.linalg.norm(target)

    # Select which protein axis to align with target
    if protein_axis.lower() == 'z':
        protein_vec = np.array([0, 0, 1])
    elif protein_axis.lower() == 'x':
        protein_vec = np.array([1, 0, 0])
    elif protein_axis.lower() == 'y':
        protein_vec = np.array([0, 1, 0])
    else:
        raise ValueError(f"protein_axis must be 'x', 'y', or 'z', got '{protein_axis}'")

    # Rotation that aligns protein_vec with target (Rodrigues)
    v = np.cross(protein_vec, target)
    c = np.dot(protein_vec, target)

    if np.linalg.norm(v) < 1e-10:
        if c > 0:
            R_align = np.eye(3)
        else:
            if abs(protein_vec[2]) < 0.9:
                flip_axis = np.array([0, 0, 1])
            else:
                flip_axis = np.array([1, 0, 0])
            flip_axis = flip_axis - np.dot(flip_axis, protein_vec) * protein_vec
            flip_axis /= np.linalg.norm(flip_axis)
            R_align = 2 * np.outer(flip_axis, flip_axis) - np.eye(3)
    else:
        v_normalized = v / np.linalg.norm(v)
        s = np.linalg.norm(v)

        vx = np.array([[0, -v_normalized[2], v_normalized[1]],
                       [v_normalized[2], 0, -v_normalized[0]],
                       [-v_normalized[1], v_normalized[0], 0]])

        R_align = np.eye(3) + s * vx + (1 - c) * (vx @ vx)

    # Spin around the alignment axis
    if control_spin:
        spin_angle = np.deg2rad(spin_angle_deg)
    else:
        spin_angle = np.random.uniform(0, 2*np.pi)

    cos_spin = np.cos(spin_angle)
    sin_spin = np.sin(spin_angle)

    ux, uy, uz = target
    R_spin = np.array([
        [cos_spin + ux**2*(1-cos_spin), ux*uy*(1-cos_spin) - uz*sin_spin, ux*uz*(1-cos_spin) + uy*sin_spin],
        [uy*ux*(1-cos_spin) + uz*sin_spin, cos_spin + uy**2*(1-cos_spin), uy*uz*(1-cos_spin) - ux*sin_spin],
        [uz*ux*(1-cos_spin) - uy*sin_spin, uz*uy*(1-cos_spin) + ux*sin_spin, cos_spin + uz**2*(1-cos_spin)]
    ])

    return R_spin @ R_align


def apply_thermal_perturbation(positions, center, amplitude):
    """Small Gaussian displacements of atom positions, preserving the mean position."""
    displacements = np.random.normal(0, amplitude, size=positions.shape)
    perturbed_positions = positions + displacements
    current_center = np.mean(perturbed_positions, axis=0)
    perturbed_positions += (center - current_center)
    return perturbed_positions


def random_points_on_sphere_surface(N, R):
    """Uniformly distributed points on a sphere surface."""
    phi = np.random.uniform(0, 2*np.pi, N)
    costheta = np.random.uniform(-1, 1, N)
    theta = np.arccos(costheta)

    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * costheta

    return np.column_stack([x, y, z])


# ================= SIZE DISTRIBUTION =================
def make_size_distribution():
    """
    Radius grid and number weights for the droplet size distribution (mean = Ro).
    A uniform grid in z (standard-normal variable) with Gaussian weights gives the
    exact number weights for the lognormal (z = (ln R - mu)/s) and Gaussian cases.
    Returns (radii, weights) with weights summing to 1.
    """
    if (not POLYDISPERSE) or SIZE_PDI <= 0 or N_RADII < 2:
        return np.array([float(Ro)]), np.array([1.0])

    n = int(N_RADII)
    z = np.linspace(-N_SIGMA, N_SIGMA, n)
    w = np.exp(-0.5 * z**2)
    w /= w.sum()

    if SIZE_DISTRIBUTION == 'lognormal':
        s = np.sqrt(np.log(1.0 + SIZE_PDI**2))
        mu = np.log(Ro) - 0.5 * s**2
        radii = np.exp(mu + s * z)
    elif SIZE_DISTRIBUTION == 'gaussian':
        if SIZE_PDI * N_SIGMA >= 1.0:
            raise ValueError("Gaussian size distribution reaches R <= 0: reduce SIZE_PDI or N_SIGMA")
        radii = Ro * (1.0 + SIZE_PDI * z)
    else:
        raise ValueError(f"Unknown SIZE_DISTRIBUTION: {SIZE_DISTRIBUTION}")

    # A discrete radius grid brings the oscillating terms back (aliasing) near q ~ pi/dR.
    # The Gaussian smoothing itself is effective above q ~ 2/sigma_R, so the averaged curve
    # is smooth for 2/sigma_R < q < q_ok with q_ok ~ pi/dR_max - 1.5/sigma_R (dR_max = largest
    # spacing within +/-2 sigma, where nearly all the weight sits).
    sigma_R = SIZE_PDI * Ro
    in_win = np.abs(z) <= 2.0
    d = np.diff(radii)
    d_win = d[in_win[:-1] & in_win[1:]]
    dR_max = np.max(d_win) if d_win.size else np.max(d)
    q_ok = np.pi / dR_max - 1.5 / sigma_R
    print(f"  Radius grid: max spacing {dR_max:.1f} Å (within ±2σ) -> averaged I(q) free of "
          f"aliasing ripples up to q ≈ {max(q_ok, 0):.3f} Å⁻¹")
    if q_ok < qmax:
        print(f"  ⚠ WARNING: q_max = {qmax} is above that limit. Above it the curve shows "
              f"ripples from the discrete radii: increase N_RADII (or reduce N_SIGMA), or ignore that q range.")

    return radii, w


def proteins_for_radius(R):
    """Number of proteins on a droplet of radius R (constant surface coverage)."""
    if SURFACE_DENSITY is None:
        if not POLYDISPERSE:
            return max(0, int(N_proteins))
        gamma = N_proteins / (4 * np.pi * (Ro / 10.0)**2)      # per nm^2
    else:
        gamma = SURFACE_DENSITY
    return max(0, int(round(gamma * 4 * np.pi * (R / 10.0)**2)))


# ================= PROTEIN PLACEMENT =================
def place_proteins_on_surface(protein_template, R_d, N_target,
                              penetration_min, penetration_max, R_protein, elements_template):
    all_positions = []
    all_elements = []

    N_target = int(N_target)
    centers = np.zeros((N_target, 3))
    penetrations = np.zeros(N_target)
    n_placed = 0
    n_fail = 0

    min_separation = 1.4 * R_protein

    if N_target == 0:
        return all_positions, all_elements, centers[:0], penetrations[:0]

    for i in range(N_target):
        max_attempts = 10000
        placed = False
        for attempt in range(max_attempts):
            direction = random_unit_vectors(1)[0]
            penetration = np.random.uniform(penetration_min, penetration_max)
            penetration_depth = penetration * R_protein

            if PERTURB_PENETRATION:
                penetration_noise = np.random.normal(0, PENETRATION_NOISE_AMPLITUDE * R_protein)
                penetration_depth += penetration_noise

            center = direction * (R_d - penetration_depth)

            if n_placed > 0:
                distances = np.linalg.norm(centers[:n_placed] - center, axis=1)
                if np.any(distances < min_separation):
                    continue
            placed = True
            break

        if not placed:
            n_fail += 1
            print(f"  Warning: could not place protein {i+1}/{N_target} (R={R_d:.0f} Å)")
            if n_fail >= 3:
                print(f"  Surface appears jammed; stopping at {n_placed} proteins.")
                break
            continue

        if ALIGNED_PROTEINS:
            rotation_matrix = aligned_rotation_matrix(
                direction,
                alignment_type=ALIGNMENT_AXIS,
                custom_axis=CUSTOM_ALIGNMENT if ALIGNMENT_AXIS == 'custom' else None,
                noise_degrees=ALIGNMENT_NOISE_DEGREES,
                protein_axis=PROTEIN_ALIGNMENT_AXIS,
                control_spin=CONTROL_SPIN,
                spin_angle_deg=SPIN_ANGLE_DEGREES
            )
        else:
            rotation_matrix = random_rotation_matrix()

        protein_pos = (rotation_matrix @ protein_template.T).T + center

        if ADD_THERMAL_NOISE:
            protein_pos = apply_thermal_perturbation(protein_pos, center, THERMAL_NOISE_AMPLITUDE)

        all_positions.append(protein_pos)
        all_elements.append(elements_template)
        centers[n_placed] = center
        penetrations[n_placed] = penetration_depth
        n_placed += 1

        if (i + 1) % 50 == 0:
            vprint(f"  Placed {i+1}/{N_target} proteins")

    return all_positions, all_elements, centers[:n_placed], penetrations[:n_placed]


# ================= CONTRAST =================
def assign_contrasts(all_positions, R_d):
    all_contrasts = []
    n_in_oil = 0
    n_in_water = 0

    for protein_pos in all_positions:
        contrasts = np.zeros(len(protein_pos))
        radii = np.linalg.norm(protein_pos, axis=1)

        mask_oil = radii < R_d
        contrasts[mask_oil] = rho_protein - rho_oil
        n_in_oil += np.sum(mask_oil)

        mask_water = ~mask_oil
        contrasts[mask_water] = rho_protein - rho_water
        n_in_water += np.sum(mask_water)

        all_contrasts.append(contrasts)

    vprint(f"  Contrast assignment: {n_in_oil} atoms in oil (Δρ = {rho_protein - rho_oil:.4f}), "
           f"{n_in_water} atoms in water (Δρ = {rho_protein - rho_water:.4f})")

    return all_contrasts


# ================= FORM FACTORS =================
def sphere_F(q, R):
    x = np.asarray(q) * R
    safe = np.where(x > 0, x, 1.0)
    return np.where(x > 0, 3 * (np.sin(safe) - safe * np.cos(safe)) / safe**3, 1.0)


def get_atomic_volume(element):
    volumes = {'H': 5, 'C': 16, 'N': 14, 'O': 12, 'S': 22, 'P': 18}
    return volumes.get(element, 15)


# ================= HISTOGRAM DEBYE =================
def build_distance_histogram_sampled(positions, contrasts, volumes, dr, rmax, max_pairs):
    N = len(positions)
    n_bins = int(np.ceil(rmax / dr))
    bin_edges = np.arange(0, rmax + dr, dr)
    bin_edges = bin_edges[:n_bins + 1]
    bin_centers = bin_edges[:-1] + dr/2
    weighted_hist = np.zeros(n_bins)

    total_pairs = N * (N - 1) // 2
    vprint(f"  Distance histogram: {N} atoms, {total_pairs:,} possible pairs")

    if total_pairs == 0:
        return bin_centers, weighted_hist

    amplitudes = contrasts * volumes

    if total_pairs <= max_pairs:
        for i in range(N - 1):
            dists = np.linalg.norm(positions[i+1:] - positions[i], axis=1)
            weights = amplitudes[i] * amplitudes[i+1:]
            weighted_hist += np.histogram(dists, bins=bin_edges, weights=weights)[0]
    else:
        n_samples = int(max_pairs)
        vprint(f"  Sampling {n_samples:,} random pairs...")

        n_used = 0
        samples_requested = 0
        batch_size = 1_000_000

        while samples_requested < n_samples:
            batch_n = min(batch_size, n_samples - samples_requested)
            samples_requested += batch_n

            i_samples = np.random.randint(0, N, size=batch_n)
            j_samples = np.random.randint(0, N, size=batch_n)

            valid = i_samples != j_samples          # remove self-pairs
            i_valid = i_samples[valid]
            j_valid = j_samples[valid]

            dists = np.linalg.norm(positions[j_valid] - positions[i_valid], axis=1)
            weights = amplitudes[i_valid] * amplitudes[j_valid]

            weighted_hist += np.histogram(dists, bins=bin_edges, weights=weights)[0]
            n_used += len(dists)

        # Distance and weight are symmetric in (i, j), so each sampled ordered pair is a
        # uniform draw from the unordered pairs; rescale to the full pair count.
        weighted_hist *= total_pairs / n_used
        vprint(f"  Scaled by {total_pairs / n_used:.2f}")

    return bin_centers, weighted_hist


def I_debye_histogram(qs, bin_centers, weighted_hist, amplitudes):
    """Debye sum from a weighted unordered-pair distance histogram (vectorised over q)."""
    I_self = np.sum(amplitudes**2)
    nz = weighted_hist != 0
    if not np.any(nz):
        return np.full_like(qs, I_self, dtype=float)
    sinc_vals = np.sinc(np.outer(qs, bin_centers[nz]) / np.pi)
    return I_self + 2.0 * (sinc_vals @ weighted_hist[nz])


def add_oil_sphere_scattering_with_exclusion(qs, I_protein, R_d, rho_oil_bulk,
                                             all_positions, all_contrasts,
                                             protein_volumes):
    """
    Oil sphere scattering with excluded-volume correction plus the oil-protein cross term.
    Works for zero proteins (bare droplet). The cross term uses a fine radial histogram
    of atom distances from the droplet centre instead of a loop over all atoms.
    """
    V_oil_full = 4/3 * np.pi * R_d**3

    if len(all_positions) > 0:
        positions_flat = np.vstack(all_positions)
        contrasts_flat = np.hstack(all_contrasts)
        volumes_flat = np.hstack(protein_volumes)
        radii = np.linalg.norm(positions_flat, axis=1)
        V_proteins_inside = np.sum(volumes_flat[radii < R_d])
        amplitudes_flat = contrasts_flat * volumes_flat
    else:
        radii = np.zeros(0)
        amplitudes_flat = np.zeros(0)
        V_proteins_inside = 0.0

    V_oil_effective = V_oil_full - V_proteins_inside

    vprint(f"  Excluded volume: full oil {V_oil_full:.2e} Å^3, protein inside {V_proteins_inside:.2e} Å^3 "
           f"({V_proteins_inside/V_oil_full*100:.2f}%)")

    A_oil = rho_oil_bulk * V_oil_effective * sphere_F(qs, R_d)
    I_oil = A_oil**2

    if radii.size > 0:
        edges = np.arange(0.0, radii.max() + 2 * dr_cross, dr_cross)
        H, edges = np.histogram(radii, bins=edges, weights=amplitudes_flat)
        r_centers = 0.5 * (edges[:-1] + edges[1:])
        nz = H != 0
        S = np.sinc(np.outer(qs, r_centers[nz]) / np.pi) @ H[nz]
        I_cross = 2.0 * A_oil * S
    else:
        I_cross = np.zeros_like(qs, dtype=float)

    I_total = I_protein + I_oil + I_cross
    return I_total, I_protein, I_oil, I_cross


# ================= SINGLE DROPLET =================
def compute_droplet(R_d, N_d, protein_template, elements_template, R_protein, qs,
                    keep_geometry=False):
    """Full MC scattering calculation for one droplet of radius R_d with N_d proteins."""
    all_positions, all_elements, centers, penetrations = place_proteins_on_surface(
        protein_template, R_d, N_d, penetration_min, penetration_max,
        R_protein, elements_template)
    n_placed = len(all_positions)

    vol_template = np.array([get_atomic_volume(e) for e in elements_template])

    if n_placed > 0:
        all_contrasts = assign_contrasts(all_positions, R_d)
        positions = np.vstack(all_positions)
        contrasts = np.hstack(all_contrasts)
        volumes = np.tile(vol_template, n_placed)

        rmax = 2 * (R_d + R_protein) + 100
        bin_centers, weighted_hist = build_distance_histogram_sampled(
            positions, contrasts, volumes, dr_histogram, rmax, max_pairs_sample)
        I_protein = I_debye_histogram(qs, bin_centers, weighted_hist, contrasts * volumes)
    else:
        all_contrasts = []
        I_protein = np.zeros_like(qs, dtype=float)

    if USE_OIL_SPHERE:
        I_total, I_prot, I_oil, I_cross = add_oil_sphere_scattering_with_exclusion(
            qs, I_protein, R_d, rho_oil_bulk, all_positions, all_contrasts,
            [vol_template] * n_placed)
    else:
        I_total, I_prot = I_protein, I_protein
        I_oil = np.zeros_like(qs, dtype=float)
        I_cross = np.zeros_like(qs, dtype=float)

    coverage = n_placed * np.pi * R_protein**2 / (4 * np.pi * R_d**2)

    result = dict(R=R_d, N_requested=int(N_d), N=n_placed, coverage=coverage,
                  V=4/3 * np.pi * R_d**3,
                  I_total=I_total, I_prot=I_prot, I_oil=I_oil, I_cross=I_cross)
    if keep_geometry:
        result.update(positions=all_positions, elements=all_elements, centers=centers,
                      penetrations=penetrations)
    return result


# ================= PDB OUTPUT =================
def write_output_pdb(prefix, all_positions, all_elements, centers, R_d,
                     subsample_proteins, cutout, cutout_angle):
    def atom_line(serial, name, res, chain, resseq, x, y, z, elem):
        return (f"ATOM  {serial:5d}  {name:^4s}{res:3s} {chain:1s}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n")

    print(f"Generating {N_oil_vis_points} oil surface points...")
    oil_surface = random_points_on_sphere_surface(N_oil_vis_points, R_d)

    if cutout:
        angle_rad = np.deg2rad(cutout_angle / 2)

        selected_positions = []
        selected_elements = []
        for pos, elem in zip(all_positions, all_elements):
            center = np.mean(pos, axis=0)
            center_norm = center / np.linalg.norm(center)
            if center_norm[0] >= np.cos(angle_rad):
                selected_positions.append(pos)
                selected_elements.append(elem)

        oil_norms = oil_surface / np.linalg.norm(oil_surface, axis=1)[:, None]
        oil_mask = oil_norms[:, 0] >= np.cos(angle_rad)
        oil_surface = oil_surface[oil_mask]

        print(f"  Cutout applied: {len(selected_positions)} proteins, {len(oil_surface)} oil points")
    else:
        selected_positions = all_positions[::subsample_proteins]
        selected_elements = all_elements[::subsample_proteins]

    with open(f"{prefix}_combined.pdb", "w") as f:
        f.write("REMARK Combined visualization: proteins + oil surface\n")
        f.write(f"REMARK Droplet radius: {R_d:.1f} A\n")

        for i, pos in enumerate(oil_surface):
            f.write(atom_line(i + 1, "O", "OIL", "O", i + 1, pos[0], pos[1], pos[2], "O"))
        f.write("TER\n")

        serial = len(oil_surface) + 1
        for prot_idx, (positions, elements) in enumerate(zip(selected_positions, selected_elements)):
            chain = chr(65 + (prot_idx % 26))
            for atom_idx, (pos, elem) in enumerate(zip(positions, elements)):
                elem_clean = elem.strip() if elem.strip() else "C"
                f.write(atom_line(serial, elem_clean, "PRO", chain, atom_idx + 1,
                                  pos[0], pos[1], pos[2], elem_clean))
                serial += 1
            f.write("TER\n")
        f.write("END\n")

    with open(f"{prefix}_proteins.pdb", "w") as f:
        f.write("REMARK Proteins only\n")
        f.write(f"REMARK Droplet radius: {R_d:.1f} A\n")
        serial = 1
        for prot_idx, (positions, elements) in enumerate(zip(selected_positions, selected_elements)):
            chain = chr(65 + (prot_idx % 26))
            for atom_idx, (pos, elem) in enumerate(zip(positions, elements)):
                elem_clean = elem.strip() if elem.strip() else "C"
                f.write(atom_line(serial, elem_clean, "PRO", chain, atom_idx + 1,
                                  pos[0], pos[1], pos[2], elem_clean))
                serial += 1
            f.write("TER\n")
        f.write("END\n")

    with open(f"{prefix}_oil_surface.pdb", "w") as f:
        f.write("REMARK Oil droplet surface\n")
        f.write(f"REMARK Radius: {R_d:.1f} A\n")
        for i, pos in enumerate(oil_surface):
            f.write(atom_line(i + 1, "O", "OIL", "O", i + 1, pos[0], pos[1], pos[2], "O"))
        f.write("TER\nEND\n")

    print(f"  Created: {prefix}_combined.pdb, {prefix}_proteins.pdb, {prefix}_oil_surface.pdb")
    print(f"  Total: {len(selected_positions)} proteins, {len(oil_surface)} oil points")


# ================= MAIN =================
def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("PDB PROTEIN ON OIL DROPLET - HISTOGRAM-BASED SAXS")
    print("POLYDISPERSE DROPLETS, CONSTANT SURFACE COVERAGE")
    print("=" * 70)

    print(f"\nLoading protein from {PDB_FILE}...")
    protein_template, elements_template = load_pdb(PDB_FILE)
    R_protein = get_protein_radius(protein_template)
    print(f"Protein effective radius (max atom distance): {R_protein:.1f} Å")

    radii, weights = make_size_distribution()
    n_radii = len(radii)
    qs = np.logspace(np.log10(qmin), np.log10(qmax), qN)

    print(f"\nMean droplet radius: {Ro:.0f} Å = {Ro/10:.1f} nm   (ratio to protein: {Ro/R_protein:.1f}x)")
    if n_radii > 1:
        print(f"Size distribution: {SIZE_DISTRIBUTION}, sigma_R/<R> = {SIZE_PDI:.2f}, {n_radii} radii "
              f"({radii.min():.0f}-{radii.max():.0f} Å)")
    else:
        print("Size distribution: monodisperse")

    # Representative droplet (closest to mean radius) is kept for PDB visualisation
    rep = int(np.argmin(np.abs(radii - Ro)))

    results = []
    for k, (R_k, w_k) in enumerate(zip(radii, weights)):
        N_k = proteins_for_radius(R_k)
        print(f"\n--- Droplet {k+1}/{n_radii}: R = {R_k:.1f} Å, weight = {w_k:.4f}, "
              f"proteins = {N_k} ---")
        res = compute_droplet(R_k, N_k, protein_template, elements_template,
                              R_protein, qs, keep_geometry=(k == rep))
        res['w'] = w_k
        results.append(res)
        print(f"    placed {res['N']} proteins, footprint coverage {res['coverage']:.1%}")
        if res['N'] == 0:
            print("    (no proteins: bare oil droplet)")
        if res['coverage'] > 1.0:
            print("    ⚠ WARNING: proteins may overlap (coverage > 100%)")

    if not USE_OIL_SPHERE and all(r['N'] == 0 for r in results):
        print("\n⚠ WARNING: no proteins and USE_OIL_SPHERE=False: intensity is identically zero.")

    # ---------- Incoherent average over the size distribution ----------
    def avg(key):
        return sum(r['w'] * r[key] for r in results)

    I_total = avg('I_total')
    I_prot_only = avg('I_prot')
    I_oil_only = avg('I_oil')
    I_cross = avg('I_cross')
    V_mean = sum(r['w'] * r['V'] for r in results)

    print("\n" + "=" * 70)
    print("Size-distribution summary")
    print(f"  <N_proteins> = {sum(r['w'] * r['N'] for r in results):.1f}, "
          f"<R> = {sum(r['w'] * r['R'] for r in results):.1f} Å, <V> = {V_mean:.3e} Å^3")
    print(f"  {'R (Å)':>9s} {'weight':>9s} {'N':>6s} {'coverage':>9s}")
    for r in results:
        print(f"  {r['R']:9.1f} {r['w']:9.4f} {r['N']:6d} {r['coverage']:9.1%}")

    # ---------- Save ----------
    np.savetxt(f"{OUT_PREFIX}_saxs.dat", np.column_stack([qs, I_total]),
               header="q(A^-1)  I(q) [number-averaged, per droplet]", fmt='%.6e')
    np.savetxt(f"{OUT_PREFIX}_saxs_per_volume.dat", np.column_stack([qs, I_total / V_mean]),
               header="q(A^-1)  I(q)/<V_droplet> [per unit oil volume]", fmt='%.6e')
    np.savetxt(f"{OUT_PREFIX}_size_distribution.dat",
               np.array([[r['R'], r['w'], r['N'], r['coverage']] for r in results]),
               header="R(A)  number_weight  N_proteins  footprint_coverage", fmt='%.6e')
    np.savetxt(f"{OUT_PREFIX}_per_radius.dat",
               np.column_stack([qs] + [r['I_total'] for r in results]),
               header="q(A^-1)  I_i(q) for R = " + " ".join(f"{r['R']:.1f}" for r in results),
               fmt='%.6e')
    print(f"\nSaved {OUT_PREFIX}_saxs.dat, _saxs_per_volume.dat, _size_distribution.dat, _per_radius.dat")

    if USE_OIL_SPHERE:
        header_text = (
            f"Number-averaged SAXS components, {n_radii} droplet radii, mean R = {Ro:.1f} A\n"
            f"Columns: q(A^-1)  I_total  I_protein  I_oil  I_cross")
        np.savetxt(f"{OUT_PREFIX}_components.dat",
                   np.column_stack([qs, I_total, I_prot_only, I_oil_only, I_cross]),
                   header=header_text, fmt='%.6e')
        np.savetxt(f"{OUT_PREFIX}_protein_only.dat", np.column_stack([qs, I_prot_only]),
                   header="q(A^-1)  I_protein", fmt='%.6e')
        np.savetxt(f"{OUT_PREFIX}_oil_only.dat", np.column_stack([qs, I_oil_only]),
                   header="q(A^-1)  I_oil", fmt='%.6e')
        np.savetxt(f"{OUT_PREFIX}_cross_term.dat", np.column_stack([qs, I_cross]),
                   header="q(A^-1)  I_cross", fmt='%.6e')
        print(f"Saved {OUT_PREFIX}_components.dat and individual component files")

    print(f"\nIntensity diagnostics:")
    print(f"  I(q) range: [{np.min(I_total):.2e}, {np.max(I_total):.2e}]")
    print(f"  Negative points: {np.sum(I_total < 0)}/{len(I_total)}")
    print(f"  I(q_min) = {I_total[0]:.2e}, I(q_max) = {I_total[-1]:.2e}")

    # ---------- PDB of representative droplet ----------
    rep_res = results[rep]
    print(f"\nWriting PDB files for representative droplet (R = {rep_res['R']:.0f} Å, "
          f"{rep_res['N']} proteins)...")
    write_output_pdb(OUT_PREFIX, rep_res['positions'], rep_res['elements'], rep_res['centers'],
                     rep_res['R'], proteins_vis_subsample, VISUALIZE_CUTOUT, cutout_angle_deg)

    # ---------- Plot ----------
    print("\nGenerating plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    if n_radii > 1:
        for r in results:
            ax1.loglog(qs, np.abs(r['I_total']), color='0.75', linewidth=0.8)
        ax1.loglog([], [], color='0.75', linewidth=0.8, label='Individual radii')
    ax1.loglog(qs, np.abs(I_total), 'b-', linewidth=2,
               label='Size-averaged I(q)' if n_radii > 1 else 'Total I(q)')
    if np.any(I_total < 0):
        ax1.loglog(qs[I_total < 0], np.abs(I_total[I_total < 0]), 'rx',
                   label='Negative values', markersize=6)
    ax1.set_xlabel('q (Å⁻¹)', fontsize=14)
    ax1.set_ylabel('I(q) (arbitrary units)', fontsize=14)
    title_size = (f"{SIZE_DISTRIBUTION}, PDI={SIZE_PDI:.2f}, " if n_radii > 1 else "")
    ax1.set_title(f"SAXS: proteins on oil droplets ({title_size}<R>={Ro/10:.0f} nm, "
                  f"<N>={sum(r['w'] * r['N'] for r in results):.0f})", fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11)

    if USE_OIL_SPHERE:
        ax2.loglog(qs, np.abs(I_prot_only), 'g-', linewidth=2, label='Proteins only', alpha=0.8)
        ax2.loglog(qs, np.abs(I_oil_only), 'orange', linewidth=2, label='Oil sphere only', alpha=0.8)
        ax2.loglog(qs, np.abs(I_cross), 'purple', linewidth=2, label='Oil-Protein cross-terms', alpha=0.8)
        ax2.loglog(qs, np.abs(I_total), 'b--', linewidth=1.5, label='Total', alpha=0.6)
        ax2.set_xlabel('q (Å⁻¹)', fontsize=14)
        ax2.set_ylabel('|I(q)| (arbitrary units)', fontsize=14)
        ax2.set_title('Scattering Components Breakdown (size-averaged)', fontsize=12)
        ax2.legend(fontsize=10)
    else:
        ax2.loglog(qs, np.abs(I_total), 'b-', linewidth=2)
        ax2.set_xlabel('q (Å⁻¹)', fontsize=14)
        ax2.set_ylabel('I(q)', fontsize=14)
        ax2.set_title('SAXS Profile', fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_saxs.png", dpi=600)
    plt.show()

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"  • {OUT_PREFIX}_saxs.dat / _saxs_per_volume.dat - size-averaged I(q)")
    print(f"  • {OUT_PREFIX}_per_radius.dat - I(q) of each droplet radius")
    print(f"  • {OUT_PREFIX}_size_distribution.dat - radii, weights, N, coverage")
    print(f"  • {OUT_PREFIX}_components.dat (+ protein/oil/cross files) - averaged components")
    print(f"  • {OUT_PREFIX}_saxs.png, {OUT_PREFIX}_combined.pdb, _proteins.pdb, _oil_surface.pdb")
    print("=" * 70)


if __name__ == "__main__":
    main()