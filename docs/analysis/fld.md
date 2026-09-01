# Forming Limit Diagram

A forming limit diagram (FLD) judges every shell element of the formed blank by its
in-plane principal strains: elements that thicken tend to wrinkle, elements that barely
stretch stay elastic, and elements that stretch far approach the material's forming
limit curve (FLC). Classifying all elements of a simulation gives a compact formability
fingerprint — the share of the blank in each zone — that can be compared across the
whole parameter sweep.

The method and zone definitions follow P. Heinzelmann et al.,
[MATEC Web Conf. 408, 01090 (2025)](https://doi.org/10.1051/matecconf/202540801090),
which applied this analysis to the DDACS simulations.

## Zones

With minor strain $\varepsilon_2$ and major strain $\varepsilon_1$ (true strains,
mid-surface, final OP10 state), evaluated left to right:

| zone | rule | meaning |
|---|---|---|
| Wrinkles | $\varepsilon_2 \le -0.01$ and $\varepsilon_1 \le \lvert\varepsilon_2\rvert$ | the sheet thickens |
| Wrinkling tendency | $\varepsilon_2 \le -0.01$ and $\varepsilon_1 \le 1.55\,\lvert\varepsilon_2\rvert$ | strong in-plane compression |
| Inadequate stretch | $\varepsilon_1 + \varepsilon_2 \le 0.02$ | almost no thinning |
| Safe | everything else below the FLC | |
| Risk of cracks / Cracks | within a margin below / above the FLC | requires the material's FLC |

None of the DDACS simulations reach the crack zones; they activate automatically once
an FLC is assigned (`FLC_POINTS` below). All thresholds are plain constants at the top
of the code (`MINOR_LIMIT`, `WRINKLE_SLOPE`, `TENDENCY_SLOPE`, `THINNING_MIN`,
`FLC_MARGIN`) — adjust them there to explore alternative definitions. The shaded
background of the diagram is generated with the same `classify` function as the data
points, so it always reflects the active thresholds.

## The diagram

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/fld_diagram.png" width="700">

??? example "This plot was created with"

    ```python
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import ddacs
    from matplotlib.colors import ListedColormap

    DATA_DIR = Path('./data')      # repository root, or Path('../data') from notebooks/
    SIM_ID = 258864

    # ---- adjustable zone definition (true strains) -------------------------
    MINOR_LIMIT = -0.01      # left of this, the wrinkling zones apply
    WRINKLE_SLOPE = 1.0      # wrinkles: major <= WRINKLE_SLOPE * |minor|
    TENDENCY_SLOPE = 1.5497  # wrinkling tendency: major <= TENDENCY_SLOPE * |minor|
    THINNING_MIN = 0.02      # inadequate stretch: major + minor <= THINNING_MIN
    FLC_POINTS = None        # np.loadtxt('DP600.fld', skiprows=2) once available
    FLC_MARGIN = 0.055       # risk of cracks: within this band below the FLC

    ZONES = ['Wrinkles', 'Wrinkling tendency', 'Inadequate stretch', 'Safe',
             'Risk of cracks', 'Cracks']
    ZONE_COLORS = ['magenta', 'tab:blue', 'tab:gray', 'tab:green', 'gold', 'tab:red']
    ZONE_BACKGROUNDS = ['#fbdcf3', '#dcebfa', '#ececec', '#ddf0dd', '#fdf6d0', '#fbdcdc']


    def principal_strains(strain):
        """In-plane principal strains from the mid-surface strain tensor.

        strain: (m, 2, 6) lower/upper surface tensors (xx, yy, zz, xy, yz, zx).
        Returns (minor, major) per element.
        """
        e = strain.mean(axis=1)  # mid-surface
        xx, yy, xy = e[:, 0], e[:, 1], e[:, 3]
        center = (xx + yy) / 2
        radius = np.sqrt(((xx - yy) / 2) ** 2 + xy**2)
        return center - radius, center + radius


    def classify(minor, major):
        """Zone index per point (see ZONES)."""
        zone = np.full(np.shape(minor), 3)                          # Safe
        zone[major + minor <= THINNING_MIN] = 2                     # Inadequate stretch
        left = minor <= MINOR_LIMIT
        zone[left & (major <= TENDENCY_SLOPE * np.abs(minor))] = 1  # Wrinkling tendency
        zone[left & (major <= WRINKLE_SLOPE * np.abs(minor))] = 0   # Wrinkles
        if FLC_POINTS is not None:
            flc = np.interp(minor, FLC_POINTS[:, 0], FLC_POINTS[:, 1])
            zone[major > flc - FLC_MARGIN] = 4                      # Risk of cracks
            zone[major > flc] = 5                                   # Cracks
        return zone


    with ddacs.open_h5(SIM_ID, data_dir=DATA_DIR) as f:
        strain = f['OP10/blank/element_shell_strain'][-1]

    minor, major = principal_strains(strain)
    zone = classify(minor, major)
    shares = {name: float((zone == k).mean()) for k, name in enumerate(ZONES)}

    XLIM, YLIM = (-0.3, 0.3), (-0.1, 0.7)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    # background: every point of the plane classified with the same function,
    # masked where major < minor (physically impossible)
    gx, gy = np.meshgrid(np.linspace(*XLIM, 600), np.linspace(*YLIM, 600))
    gz = np.ma.masked_where(gy < gx, classify(gx, gy))
    ax.pcolormesh(gx, gy, gz, cmap=ListedColormap(ZONE_BACKGROUNDS),
                  vmin=-0.5, vmax=5.5, shading='auto', zorder=0)
    x = np.linspace(XLIM[0], 0, 50)
    ax.plot(x, WRINKLE_SLOPE * np.abs(x), color='magenta', lw=1.2)
    ax.plot(x, TENDENCY_SLOPE * np.abs(x), color='tab:blue', lw=1.2)
    xr = np.linspace(0, XLIM[1], 50)
    ax.plot(xr, xr, color='black', lw=1.2)   # physical limit: major = minor
    if FLC_POINTS is not None:
        ax.plot(FLC_POINTS[:, 0], FLC_POINTS[:, 1], color='red', lw=1.5)
        ax.plot(FLC_POINTS[:, 0], FLC_POINTS[:, 1] - FLC_MARGIN, color='gold', lw=1.2)
    for k, (name, color) in enumerate(zip(ZONES, ZONE_COLORS)):
        sel = zone == k
        if sel.any():
            ax.scatter(minor[sel], major[sel], s=4, alpha=0.6, color=color,
                       label=f'{name} ({shares[name]:.0%})', edgecolors='none')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_xlabel(r'Minor true strain $\varepsilon_2$')
    ax.set_ylabel(r'Major true strain $\varepsilon_1$')
    ax.set_title(f'Forming Limit Diagram - Simulation {SIM_ID}')
    ax.legend(markerscale=2, loc='upper right')
    ax.grid(alpha=0.3)
    plt.show()
    ```

## The zones on the part

The same labels drawn on the formed blank: wrinkles sit in the drawn wall and the
flange transition, the safe zone on the stretched bottom face.

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/fld_mesh.png" width="700">

??? example "This plot was created with"

    ```python
    with ddacs.open_h5(SIM_ID, data_dir=DATA_DIR) as f:
        nodes = f['OP10/blank/node_displacement'][-1]
        faces = f['OP10/blank/element_shell_node_indexes'][:]

    ax, cbar = ddacs.plot_mesh(
        nodes, faces,
        values=zone.astype(float),
        cmap=ListedColormap(ZONE_COLORS),
        vmin=-0.5, vmax=5.5,
        colorbar_label='FLD zone',
        mirror=True,
    )
    cbar.set_ticks(range(len(ZONES)))
    cbar.set_ticklabels(ZONES)
    ax.set_title(f'FLD zones on the formed blank - Simulation {SIM_ID}')
    plt.show()
    ```

## Notes

- Strains are read from `element_shell_strain` (final OP10 state) and averaged over the
  lower and upper integration surface; the in-plane principal values are computed from
  the $xx$, $yy$, $xy$ components. This is a simple, reproducible definition — for
  strongly bent wall elements a surface-resolved evaluation can differ.
- The zone shares make a per-simulation formability fingerprint: computed over the
  sweep, the safe share rises and the wrinkle share falls monotonically with
  blankholder force.
