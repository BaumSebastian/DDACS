# Forming Limit Diagram

A forming limit diagram (FLD) judges every shell element of the formed blank by its
in-plane principal strains: elements that thicken tend to wrinkle, elements that barely
stretch stay elastic, and elements that stretch far approach the material's forming
limit curve (FLC). Classifying all elements of a simulation gives a compact formability
fingerprint (the share of the blank in each zone) that can be compared across the
whole parameter sweep.

The method and zone definitions follow P. Heinzelmann et al.,
[MATEC Web Conf. 408, 01090 (2025)](https://doi.org/10.1051/matecconf/202540801090),
which applied this analysis to the DDACS simulations.

## Zones

With minor strain $\varepsilon_2$ and major strain $\varepsilon_1$ (true strains,
mid-surface, final OP10 state), evaluated left to right:

| zone | rule |
|---|---|
| Wrinkles | $\varepsilon_2 \le -0.01$ and $\varepsilon_1 \le \lvert\varepsilon_2\rvert$ |
| Wrinkling tendency | $\varepsilon_2 \le -0.01$ and $\varepsilon_1 \le \frac{1+R}{R}\lvert\varepsilon_2\rvert$, $R = 1.82$ |
| Inadequate stretch | $\varepsilon_1 + \varepsilon_2 \le 0.02$ |
| Safe | everything else below the FLC margin |
| Risk of cracks | above the FLC reduced by 10 % ($0.9\,\mathrm{FLC}$) |
| Cracks | above the FLC |

The inadequate-stretch zone is a small wedge at the origin (between the physical limit and the 2 % thinning line); the diagram's zoom inset makes it visible. The zone parameters are the LS-Dyna Formability settings used by Heinzelmann
(R-value 1.82, essential thinning 0.02, wrinkle slope 1, risk margin 10 % of the FLC); the FLC is the
LS-Dyna calculated curve for the sheet (CRLCS, t = 0.8 mm, n = 0.21, true strains),
shipped as [`flc_dp600.csv`](flc_dp600.csv) next to this page. Only a small fraction of the simulations reaches the crack zone at all (see the
note at the figures below). All thresholds are plain constants at the top of
the code (`MINOR_LIMIT`, `WRINKLE_SLOPE`, `R_VALUE`, `THINNING_MIN`, `FLC_MARGIN`);
adjust them there to explore alternative definitions. The shaded background of the
diagram is generated with the same `classify` function as the data points, so it
always reflects the active thresholds.

## The diagram

!!! note "Why this simulation"
    Simulation 309485 is shown for the visual dynamics: it is one of the few
    simulations that reach the crack zone at all (a 1500-simulation sample of the
    published set found 41, about 3 %, typically with only ~1-2 % of their
    elements above the FLC). The typical DDACS simulation stays below the FLC
    everywhere.

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/fld_diagram.png" width="700">

??? example "This plot was created with"

    ```python
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import ddacs
    from matplotlib.colors import ListedColormap

    DATA_DIR = Path('./data')      # repository root, or Path('../data') from notebooks/
    SIM_ID = 309485

    # ---- adjustable zone definition (true strains) -------------------------
    MINOR_LIMIT = -0.01      # left of this, the wrinkling zones apply
    WRINKLE_SLOPE = 1.0      # wrinkles: major <= WRINKLE_SLOPE * |minor| (LS-Dyna wrinkle slope 1)
    R_VALUE = 1.82           # Lankford coefficient of the sheet
    TENDENCY_SLOPE = (1 + R_VALUE) / R_VALUE  # wrinkling tendency: major <= 1.549 * |minor|
    THINNING_MIN = 0.02      # inadequate stretch: major + minor <= THINNING_MIN (essential thinning)
    FLC_POINTS = np.loadtxt('docs/analysis/flc_dp600.csv', delimiter=',', skiprows=3)
    # extend the FLC along its last segment until it meets the physical limit major = minor
    p1, p2 = FLC_POINTS[-2], FLC_POINTS[-1]
    s = (p2[1] - p1[1]) / (p2[0] - p1[0])
    x_red = (p2[1] - s * p2[0]) / (1 - s)
    FLC_POINTS = np.vstack([FLC_POINTS, [x_red, x_red]])
    x_yellow = 0.9 * (p2[1] - s * p2[0]) / (1 - 0.9 * s)   # 0.9 * FLC meets major = minor
    FLC_MARGIN = 0.10        # risk of cracks: FLC reduced by 10 percent (relative, as in the paper)

    ZONES = ['Wrinkles', 'Wrinkling tendency', 'Inadequate stretch', 'Safe',
             'Risk of cracks', 'Cracks']
    ZONE_COLORS = ['magenta', 'tab:blue', 'tab:gray', 'tab:green', 'gold', 'tab:red']
    ZONE_BACKGROUNDS = ['#fbdcf3', '#dcebfa', '#d6d6d6', '#ddf0dd', '#fdf6d0', '#fbdcdc']


    def principal_strains(strain, nodes, faces):
        """In-plane principal strains, evaluated in each element's local frame.

        LS-PrePost recipe: rotate the mid-surface strain tensor into the element
        coordinate system (z = element normal), drop the z components, take the
        2D principal values. strain: (m, 2, 6) lower/upper surface tensors
        (xx, yy, zz, xy, yz, zx) in global coordinates. Returns (minor, major).
        """
        e = strain.mean(axis=1)                    # mid-surface, global frame
        p = nodes[faces - faces.min()]             # (m, 4, 3) corner coordinates
        ez = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])   # element normal
        ez /= np.linalg.norm(ez, axis=1, keepdims=True)
        a = np.where(np.abs(ez[:, [0]]) < 0.9, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        ex = a - (a * ez).sum(1, keepdims=True) * ez
        ex /= np.linalg.norm(ex, axis=1, keepdims=True)
        ey = np.cross(ez, ex)
        R = np.stack([ex, ey, ez], axis=1)         # rows = local axes
        T = np.zeros((len(e), 3, 3))
        T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = e[:, 0], e[:, 1], e[:, 2]
        T[:, 0, 1] = T[:, 1, 0] = e[:, 3]
        T[:, 1, 2] = T[:, 2, 1] = e[:, 4]
        T[:, 0, 2] = T[:, 2, 0] = e[:, 5]
        L = R @ T @ R.transpose(0, 2, 1)           # tensor in the element frame
        xx, yy, xy = L[:, 0, 0], L[:, 1, 1], L[:, 0, 1]
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
            zone[major > (1 - FLC_MARGIN) * flc] = 4                # Risk of cracks
            zone[major > flc] = 5                                   # Cracks
        return zone


    with ddacs.open_h5(SIM_ID, data_dir=DATA_DIR) as f:
        strain = f['OP10/blank/element_shell_strain'][-1]
        nodes = f['OP10/blank/node_displacement'][-1]
        faces = f['OP10/blank/element_shell_node_indexes'][:]

    minor, major = principal_strains(strain, nodes, faces)
    zone = classify(minor, major)
    shares = {name: float((zone == k).mean()) for k, name in enumerate(ZONES)}

    XLIM, YLIM = (-0.3, 0.46), (-0.1, 0.7)  # x reaches the FLC / physical-limit intersection
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
        yy = (1 - FLC_MARGIN) * FLC_POINTS[:, 1]
        keep = yy >= FLC_POINTS[:, 0]              # clip the margin line at major = minor
        ax.plot(np.append(FLC_POINTS[keep, 0], x_yellow), np.append(yy[keep], x_yellow), color='gold', lw=1.2)
    for k, (name, color) in enumerate(zip(ZONES, ZONE_COLORS)):
        sel = zone == k
        if sel.any():
            ax.scatter(minor[sel], major[sel], s=4, alpha=0.6, color=color,
                       label=f'{name} ({shares[name]:.1%})', edgecolors='none')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_xlabel(r'Minor true strain $\varepsilon_2$')
    ax.set_ylabel(r'Major true strain $\varepsilon_1$')
    ax.set_title(f'Forming Limit Diagram - Simulation {SIM_ID}')
    ax.legend(markerscale=2, loc='upper right')
    ax.grid(alpha=0.3)
    # zoom inset at the origin: makes the small inadequate-stretch wedge visible
    axins = ax.inset_axes([0.60, 0.05, 0.38, 0.36])
    gxi, gyi = np.meshgrid(np.linspace(-0.06, 0.04, 400), np.linspace(-0.02, 0.075, 400))
    gzi = np.ma.masked_where(gyi < gxi, classify(gxi, gyi))
    axins.pcolormesh(gxi, gyi, gzi, cmap=ListedColormap(ZONE_BACKGROUNDS),
                     vmin=-0.5, vmax=5.5, shading='auto')
    xi = np.linspace(-0.06, 0, 30)
    axins.plot(xi, WRINKLE_SLOPE * np.abs(xi), color='magenta', lw=1)
    axins.plot(xi, TENDENCY_SLOPE * np.abs(xi), color='tab:blue', lw=1)
    axins.plot(np.linspace(0, 0.04, 10), np.linspace(0, 0.04, 10), color='black', lw=1)
    for k, color in enumerate(ZONE_COLORS):
        sel = zone == k
        if sel.any():
            axins.scatter(minor[sel], major[sel], s=2, alpha=0.5, color=color, edgecolors='none')
    axins.set_xlim(-0.06, 0.04); axins.set_ylim(-0.02, 0.075)
    axins.tick_params(labelsize=7)
    ax.indicate_inset_zoom(axins, edgecolor='black')
    plt.show()
    ```

## The zones on the part

The same labels drawn on the formed blank: wrinkles sit in the drawn wall and the
flange transition, the safe zone on the stretched bottom face.

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/fld_mesh.png" width="700">

??? example "This plot was created with"

    ```python
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

## Zone shares across the parameter space

The per-simulation zone shares are a compact formability fingerprint, the same
aggregation the accompanying paper applies to the full sweep. Each simulation is
reduced to one vector of zone shares, Eq. (1) in the
[paper](https://doi.org/10.1051/matecconf/202540801090):

$$ \mathbf{y} \in \mathbb{R}^6, \qquad y_i = \frac{n_i}{N}, \qquad y_i > 0, \qquad \sum_i y_i = 1 $$

with $n_i$ the number of blank elements in FLD category $i$ and $N$ the total
number of blank elements. This distribution summarizes a whole simulation in a
single vector; the paper uses it as the prediction target of a machine learning
model instead of classifying every element individually. Over all 32,466
published simulations the trends are monotonic and physically plausible: more
blankholder force suppresses wrinkling (40 % to 28 %) and inadequate stretch
(37 % to 20 %) and grows the safe share from 11 % to 37 %. The shaded bands are
the 10th to 90th percentile across the simulations of each force level; the
spread within a level (geometry, thickness, friction) is of the same order as the
trend itself. Crack elements stay rare throughout: 3.1 % of the simulations have
any (at most 1.6 % of their elements), concentrated at the highest forces.

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/fld_zones_vs_bf.png" width="700">

??? example "This plot was created with"

    ```python
    import pandas as pd

    rows = []
    for _, prow in params.iterrows():          # every published simulation
        with ddacs.open_h5(int(prow['index']), data_dir=DATA_DIR) as f:
            strain = f['OP10/blank/element_shell_strain'][-1]
            nodes = f['OP10/blank/node_displacement'][-1]
            faces = f['OP10/blank/element_shell_node_indexes'][:]
        minor, major = principal_strains(strain, nodes, faces)
        zone = classify(minor, major)
        rows.append({'BF': float(prow['blankholder_force']),
                     **{name: float((zone == k).mean()) for k, name in enumerate(ZONES)}})
    df = pd.DataFrame(rows)

    bf = df['BF'] / 1000
    mean = df.groupby(bf)[ZONES].mean() * 100
    lo = df.groupby(bf)[ZONES].quantile(0.10) * 100
    hi = df.groupby(bf)[ZONES].quantile(0.90) * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, color in zip(ZONES, ZONE_COLORS):
        ax.plot(mean.index, mean[name], 'o-', color=color, label=name, ms=4)
        ax.fill_between(mean.index, lo[name], hi[name], color=color, alpha=0.2, linewidth=0)
    ax.set_xlabel('blankholder force in kN')
    ax.set_ylabel('share of elements in %')
    ax.legend(); ax.grid(alpha=0.3)
    plt.show()
    ```

    `params` is `process_parameters.csv`; `principal_strains`, `classify`, `ZONES`
    and `ZONE_COLORS` are the definitions from the diagram example above. The full
    loop takes a couple of hours single-threaded; parallelise over simulations
    (`concurrent.futures`) for a coffee-break runtime.

## Notes

- Strains are read from `element_shell_strain` (final OP10 state) and averaged over the
  lower and upper integration surface; the in-plane principal values are computed from
  the $xx$, $yy$, $xy$ components after rotating the tensor into the element's local
  frame (z = element normal) and dropping the z components (the LS-PrePost recipe).
  A plain global-frame evaluation is only exact where the element plane is
  axis-aligned; on inclined wall elements it underestimates the strains enough to
  hide the crack zone entirely.
- The zone shares make a per-simulation formability fingerprint: computed over the
  sweep, the safe share rises and the wrinkle share falls monotonically with
  blankholder force.
