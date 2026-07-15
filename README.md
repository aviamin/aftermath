# AfterMath

**Hurricane building-damage classification from before/after satellite imagery.**

Status: in progress — see `docs/superpowers/specs/2026-07-14-aftermath-design.md` for the full design.

## Dataset setup

This project uses the [xBD dataset](https://xview2.org) (free registration
required). After downloading, place the four hurricane events at:

```
data/raw/hurricane-harvey/
data/raw/hurricane-florence/
data/raw/hurricane-matthew/
data/raw/hurricane-michael/
```

Each event folder should contain xBD's standard `images/` and `labels/`
subfolders as provided by the download.

## Setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Tests

```bash
pytest -v
```
