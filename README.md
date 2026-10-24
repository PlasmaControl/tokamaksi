# tokamaksi

Read the Tokamak-SI DIII-D event labels and draw them on a shot, with a few of the
shot's signals above them. A small reader for the labels released on Hugging Face;
the labeling tool itself lives elsewhere.

**Code: MIT. Data: not licensed by this repository.** The labels are in a private
Hugging Face dataset (`nc1/tokamak-si`); ask the maintainers for access. They include
third-party tables under their owners' terms. DIII-D data-use rules apply to
everything, signals included.

## Install

    pip install "tokamaksi[plot,h5] @ git+ssh://git@github.com/PlasmaControl/tokamaksi.git"

`plot` adds matplotlib and `h5` adds h5py (reading shot signals). The core needs only
numpy, pandas, pyarrow, pyyaml and huggingface_hub.

## Get the labels (about 20 MB)

    hf auth login                    # or: export HF_TOKEN=...   (needs read access)
    tokamaksi pull v1
    tokamaksi info

A release is a folder of plain files under `$TOKAMAKSI_HOME/releases/<tag>` (default
`~/.cache/tokamaksi`). `pull` downloads into a temporary folder, checks every file
against the sha256 in `release.json` and moves it into place, so a failed pull leaves
nothing half-written. Hugging Face's own cache (`HF_HOME`) is never changed, and no
token is ever read or shown by tokamaksi. Set `TOKAMAKSI_OFFLINE=1` to forbid the
network; a release that is not on disk then says which command to run. A release
folder from anywhere else opens by path: `tk.open_release("/path/to/release")`.

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

Point `TOKAMAKSI_CORPUS` at a folder of processed per-shot files
(`<shot>_processed.h5`: one group per diagnostic, with `xdata` in seconds and `ydata`
as `(channels, samples)`) and stack signals over the labels:

    export TOKAMAKSI_CORPUS=<folder of *_processed.h5>
    tk.signals.available(185866)                # groups that ran, with (channels, samples)
    tk.plot.shot(185866, ["edge_localized_mode"], signals=["filterscopes:0", "neutron_rate:0"])
    tokamaksi plot 185866 --events edge_localized_mode --signals filterscopes:0,neutron_rate:0 --out shot.png

A signal is `group` (every channel, up to eight), `group:3` or `group:0,2,5`. The
folder covers only some shots; a shot or group it lacks raises a clear error. Fetching
signals live from DIII-D (fdp) is planned, not built.

## Read this before using the labels

* Missing is not zero. A row with category 0 says "assessed, not seen"; time with no
  row was never assessed (`-1` on the grid, hatched grey in plots).
* `tier="reviewed"` are tables a person saved on the review page. Many saved ELM
  individuals are detector output a reviewer confirmed, not hand-timed onsets.
  `original` are third-party tables, `derived` model or rule outputs, and `source`
  machine drafts that are not labels.
* The gold shots (split `test`, `tk.shotlists.gold()`) are for evaluation: do not
  train or tune on them.
* `kind`: `span` = iscrowd 1, `individual` = 0, `unspecified` = no flag. It records
  what a table says, not how any benchmark scores it. See `docs/schema.md`.

Not here: a labeling UI, model weights, live DIII-D access, the label data itself.
