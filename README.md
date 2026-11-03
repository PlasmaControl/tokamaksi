# Tokamak-SI

## Install

    pip install "tokamaksi @ git+https://github.com/PlasmaControl/tokamaksi.git"

## Get the labels

    hf auth login                    # or: export HF_TOKEN=...   (needs DIII-D and author permission)
    tokamaksi pull v1
    tokamaksi info

## Use

    import tokamaksi as tk

    elm = tk.load_labels("edge_localized_mode", tier="reviewed")
    tk.spans(elm); tk.individuals(elm)          # a span is a stretch of ELMs not counted one by one
    tk.Taxonomy.load().categories("confinement")

    g = tk.load_grid("gold").decode(170659)     # 10 ms frames; -1 means nobody assessed that time
    tk.shotlists.gold().head()

    fig = tk.plot.shot(170659, ["edge_localized_mode", "confinement"])
    fig.savefig("shot.png")

    tokamaksi plot 170659 --events edge_localized_mode --out shot.png

## Shot signals

    export TOKAMAKSI_CORPUS=<folder of *_processed.h5>
    tk.signals.available(185866)                # groups that ran, with (channels, samples)
    tk.plot.shot(185866, ["edge_localized_mode"], signals=["filterscopes:1", "neutron_rate:3"])
    tokamaksi plot 185866 --events edge_localized_mode --signals filterscopes:1,neutron_rate:3 --out shot.png

## Citation

```bibtex
@misc{chen2026tokamaksi,
  title         = {Tokamak-SI: Interpreting Tokamak Discharges with Labeled Catalogs},
  author        = {Chen, Nathaniel and Garcia, Alvin and Shousha, Ricardo and Butt, Jalal and Jalalvand, Azarakhsh and Kolemen, Egemen},
  year          = {2026},
  month         = oct,
  eprint        = {2610.32264},
  archivePrefix = {arXiv},
  primaryClass  = {physics.plasm-ph},
  url           = {https://arxiv.org/abs/2610.32264}
}
```
