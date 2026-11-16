# SAXS-simulators
# Raspberry Model Simulator — X-ray Scattering

A Jupyter notebook for simulating small-angle X-ray scattering (SAXS) from raspberry-like particle systems: oil droplets with smaller particles adsorbed at the interface, dispersed in a continuous aqueous phase.

## Overview

The raspberry model describes a composite colloidal system in which small particles (e.g. proteins) adsorb onto the surface of larger emulsion droplets. This notebook computes the theoretical SAXS intensity from such a system, based on the analytical form factor described in the associated manuscript.

The total scattering is decomposed into:
- **I_op** — contribution from droplet–particle complexes (adsorbed particles)
- **I_p** — contribution from free (non-adsorbed) particles in the bulk
- **I_total** — sum of both contributions

## Parameters

All model parameters are defined at the top of the notebook and can be edited directly:

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Core radius | `Ro_å` | 8000 Å | Radius of the oil droplet |
| Particle radius | `Rp_å` | 30 Å | Radius of the small adsorbed particles |
| Penetration depth | `d` | −0.5 | Controls how deeply particles embed in the interface (−1 to 1) |
| Oil electron density contrast | `Dqo_å` | −0.06 e/Å³ | Electron density contrast of oil vs. solvent (triolein: −0.056) |
| Particle electron density contrast | `Dqp_å` | 0.12 e/Å³ | Electron density contrast of particles vs. solvent (protein: ~0.12) |
| Oil volume fraction | `phi_o` | 0.2 | Volume fraction of oil droplets |
| Total particle volume fraction | `phi_p` | 0.0005 | Total volume fraction of small particles |
| Adsorbed particle fraction | `phi_ap` | 1.0 | Fraction of particles adsorbed at the droplet interface (0–1) |

## Workflow

1. **Surface coverage check** — verifies that the number of adsorbed particles per droplet does not exceed the physical monolayer limit.
2. **Scattering calculation** — computes I_op, I_p, and I_total over a q-range of 10⁻⁴ to 1 Å⁻¹ (2500 log-spaced points).
3. **Gaussian smearing** — applies instrument resolution broadening in log(q) space.
4. **Porod analysis** — fits and displays the high-q Porod region (expected slope q⁻⁴ for smooth interfaces).
5. **Export** — saves results to CSV files.

## Output Files

| File | Contents |
|------|----------|
| `scattering_components.csv` | q, I_op, I_p, I_total |
| `smeared_intensity.csv` | q, I_smeared |
| `slope_intensity.csv` | q_fit, I_fitted (Porod fit) |

## Requirements

```
numpy
scipy
matplotlib
pandas
```

Install with:

```bash
pip install numpy scipy matplotlib pandas
```

## Key Equations

The form factor of the raspberry complex (Eq. 10 in the referenced manuscript) is:

$$P(q) = \frac{1}{M^2} \left[ \Delta\rho_o^2 V_o^2 W(q, R_o)^2 + N_p \Delta\rho_p^2 V_p^2 W(q, R_p)^2 + N_p(N_p-1)\Delta\rho_p^2 V_p^2 S_{pp} + 2 N_p \Delta\rho_o \Delta\rho_p V_o V_p S_{op} \right]$$

where $W(q,R) = 3(\sin qR - qR \cos qR)/(qR)^3$ is the sphere form factor amplitude, $M$ is the total scattering length of the complex, and $N_p$ is the mean number of particles per droplet.

The total intensity (Eq. 12) is:

$$I(q) = I_{op}(q) + I_p(q)$$

## Usage

1. Open the notebook in JupyterLab or Jupyter Notebook.
2. Edit the parameters in the first code cell.
3. Run all cells (`Kernel → Restart & Run All`).
4. Inspect the plots and CSV exports.

## Reference

Kjersta Larson-Smith, Andrew Jackson, Danilo C. Pozzo, Small angle scattering model for Pickering emulsions and raspberry particles, Journal of Colloid and Interface Science, Volume 343, Issue 1, 2010, Pages 36-41, https://doi.org/10.1016/j.jcis.2009.11.033.
