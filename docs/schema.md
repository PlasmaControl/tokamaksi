# The release schema

tokamaksi reads a release written by the Tokamak-SI exporter. This page says what it
expects. The schema number is in `release.json` (`"schema"`; absent means 1) and
`tokamaksi.RELEASE_SCHEMA` is the newest one this version reads. A release with a
higher number raises `SchemaError`. Adding columns or files does not change the
number (readers ignore extras); renaming or removing a column, or changing its
meaning or dtype, does.

## Layout

    release.json  taxonomy.yaml  datasets.parquet  signals.yaml  README.md  LICENSE.md
    labels/<tier>/<split>.parquet
    grid/<name>.parquet
    shots/<name>.parquet
    fdp_1ms/<event>/                 (optional: the all-shot 1 ms model labels)

`tables/` (the editable label tables) may exist on the Hub; tokamaksi never fetches
or reads it. `release.json` lists every file with its size and sha256. A listed file
that `pull` does not fetch may be absent.

## Tiers and splits

| tier | meaning |
|---|---|
| `reviewed` | saved by a person on the review page, one label per shot and event |
| `original` | third-party tables, reformatted without changing what they say |
| `derived` | rule or model output |
| `source` | machine drafts that nobody reviewed: not labels |

`load_labels` defaults to `reviewed` and never falls back to another tier.

Splits are per shot: `test` is the gold shots (evaluation only), `holdout` the shots an
event's roster keeps out of training and tuning, `train` the rest. A `cohort_split`
file also has a `val` split.

## Label columns

`shot, event, category, category_name, t_start_ms, t_end_ms, confidence, iscrowd,
attrs, tier, layer, dataset, method, version, reviewer, split`, plus `kind` added by
tokamaksi. Intervals are half-open `[t_start_ms, t_end_ms)` on a 10 ms grid.
`reviewer` is a pseudonym (`R1`, `R2`, ...) and is never to be decoded. `attrs` is
JSON text with a small whitelist of keys.

### `iscrowd` and `kind`

| `iscrowd` | `kind` | meaning |
|---|---|---|
| true | `span` | a stretch of activity whose events are not counted one by one |
| false | `individual` | one countable event |
| null | `unspecified` | the table sets no flag (every `absent` row, and most third-party rows) |

`span` is the value the COCO `iscrowd` flag takes for a group annotation. The words
describe a label table, not a benchmark: how a benchmark scores a missed span is the
benchmark's rule. `unspecified` is not `individual` and not `span`; a consumer that
needs one of the two must choose how to read it.

## Missing is not zero

Category 0 is `absent` for most events: assessed and not seen. Time that no row covers
was not assessed. On the grid it is `-1` (`tokamaksi.schema.UNASSESSED`), and plots draw
it hatched grey, never white and never the colour of category 0. Some events name
something else for 0 (`Taxonomy.zero(event)`): for `confinement` and `detachment` a 0
outside the owner's own save only says nothing was marked, so the grid makes it -1 and
the plot leaves it hatched and out of the legend; the owner's own 0 stays 0 and is
listed by the taxonomy's words ("the owner saved no regime here"). Rows whose category
is `not_observable`, `unassessed` or `unmarked (not assessed)` are not assessed either:
the grid rule makes those frames -1, and the plot leaves the hatch showing instead of
painting them.

## The 10 ms grid

`grid/<name>.parquet` holds runs: `shot, event, frame, n_frames, t_start_ms,
t_end_ms, category, tier, layer, dataset, split`. A run covers frames
`[frame, frame + n_frames)`, each `[10 k, 10 k + 10)` ms. `load_grid(name).decode(shot)`
fills a `(events, frames)` int8 array with `-1` where no run applies. v0.2.1 ships `gold`
(split `test`) and `silver`; a release without a grid raises `MissingPartError`
naming the ones it has.

## Folded events

The review page shows a few events as categories of another: a table for such an event
keeps its own rows, and `Taxonomy.folded_into(event)` gives where the page and the grid
show it and the category mapping (for example `sawtooth_oscillation` is category 1 or 2
of `mhd_n1`). `load_labels("sawtooth_oscillation")` returns the event's own rows; the
folded grid row is under the other event.

## Shot lists

`shots/gold.parquet` (`shot, set, split, run_day`), `silver.parquet` (`shot, set,
split`), `holdout.parquet` (`shot, set, split, events, silver`) and
`cohort_split.parquet` (`shot, split, gold, holdout`). Named review lists are not in
the release.

## All-shot model labels (1 ms)

One model per event, run on every DIII-D plasma shot it could read. These tables are
tier `derived`: model output that nobody reviewed, not labels. Unlike the rest of the
release they are on a 1 ms grid, not 10 ms. A release may ship them as
`fdp_1ms/<event>/`; adding them does not change the schema number. `tokamaksi.fdp`
reads them from `root=`, else the folder `$TOKAMAKSI_FDP_LABELS` names, else the
release's `fdp_1ms/`. Under that folder, `<event>/` is the database when it holds
`meta.json`, else its one subfolder that does; two such subfolders are an error.
`$TOKAMAKSI_FDP_LABELS` names the published databases, or a verbatim copy of them. It
never names a producer's working folder or a development run, and nothing deeper than
`<event>/<name>/` is read.

| file | holds |
|---|---|
| `spans.parquet` | `shot, t_start_ms, t_end_ms` (int32): half-open ms ranges where the event is present. `confinement` adds `category` (1 high, 2 low, 3 qh, 4 wpqh, 5 uncertain), `predicted` (the network's class, kept where `category` is uncertain), `confidence` and `tier` |
| `shots.parquet` | one row per plasma candidate: `status` (`labeled`, or why the shot has none), `plasma_end_ms`, `note`; for labeled shots the assessed window `window_start_ms, window_end_ms` (half-open) and counts. Other columns differ by event |
| `valid.csv` | `shot, valid`: every shot number of the range, `valid` 1 when the shot has labels |
| `meta.json` | the model, method, counts, the `statuses` dictionary and the `caveats`; its keys differ by event |
| `README.md` | a loading snippet |

Inside a labeled shot's window, a ms that no span covers is 0: absent, or for
`confinement` in no segment (beam-off time, say). Outside the window nothing was
assessed: `FdpLabels.decode(shot, t_range)` gives `-1` there, as on the 10 ms grid,
and the plot leaves it hatched. A shot whose status is not `labeled` raises
`NoDataError` naming the status. `alfven_eigenmode` has a `notch` column (true where
pickup lines were notched out of the input before the model ran); it is for display,
never a filter. The producers also write a dense per-shot `grid_1ms/` copy; it holds
nothing the spans and windows do not, a release leaves it out and tokamaksi never
reads it.

## Signals

Not part of the release. `tokamaksi.signals` reads `<shot>_processed.h5` files from a
folder you name (`TOKAMAKSI_CORPUS`): a group per diagnostic with `xdata` (seconds)
and `ydata` `(channels, samples)`. A group whose `ydata` has one sample is a
sentinel meaning the diagnostic did not run. Times are returned in milliseconds from
the start of the shot, the unit of the labels.
