# Pre-trained checkpoints

This directory is empty in the source repository. Download the published
checkpoint bundle from the link in the top-level README (Hugging Face Hub,
anonymous account during review) and place its contents here:

```bash
pip install huggingface_hub
huggingface-cli download <ANONYMOUS_REPO_ID> \
    --repo-type model --local-dir .
```

After download this directory will contain:

```
checkpoints/
├── MANIFEST.json
├── meccano_comp_atomic/best.pth
├── meccano_comp_shared/best.pth
├── ... one directory per configuration tag ...
```

The tag and file names follow the configuration tags used by the training
and evaluation scripts; no path rewriting is needed.

`MANIFEST.json` lists every checkpoint together with the paper exhibit (table
row, figure) it backs.

To skip downloading and train from scratch instead, see the *Reproducing the
paper exhibits* section of the top-level README.
