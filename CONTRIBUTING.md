# Contributing

Pull requests and issues are welcome.

## Good first contributions

- A new model, following the folder layout below
- A data source that broke, and the fix
- Clearer explanations in a README
- An error in the maths, with the correction

## Folder layout

```
<model>/
  model.py        runnable on its own, writes figures to charts/
  notebook.ipynb  optional, the same model step by step
  charts/         figures produced by the script
  data/           small inputs only, under about 1 MB
  README.md       what it does, how to run it, references
```

## Rules

- Everything runs on free data, or no data at all. No paid vendors.
- No credentials in code. Read keys from environment variables.
- Write figures into `charts/`, never to an absolute path.
- Report results honestly, including the ones that do not work. Out-of-sample numbers stay in.
- Papers are cited with a link, never redistributed as PDFs.

## Getting started

Fork the repository, then clone your fork. Pull requests come from your fork's branch.

## Running a model

```bash
pip install -r requirements.txt
cd <model>
py -3 model.py
```
