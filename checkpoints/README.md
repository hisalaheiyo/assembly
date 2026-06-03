# Pre-trained checkpoints

This directory is empty in the source repository. The published checkpoint
bundle is hosted on the Hugging Face Hub:

🤗 <https://huggingface.co/assembly-verbnoun/assembly-ckpt1>

The bundle is split into six ~9 GB tar shards; download them, then
reassemble and extract:

```bash
pip install huggingface_hub
huggingface-cli download assembly-verbnoun/assembly-ckpt1 --repo-type model --local-dir .

# stream-concat + extract (no 49 GB intermediate file)
cat checkpoints.tar.part* | tar -xf -
rm checkpoints.tar.part*    # optional: clean up the shards
```

After extraction this directory will contain:

```
checkpoints/
├── MANIFEST.json
├── meccano_comp_atomic/best.pth
├── meccano_comp_shared/best.pth
├── ... one directory per configuration tag ...
```

The tag and file names follow the configuration tags used by the training
and evaluation scripts; no path rewriting is needed. `MANIFEST.json` lists
every checkpoint together with the paper exhibit (table row, figure) it
backs.

To skip downloading and train from scratch instead, see the *Reproducing the
paper exhibits* section of the top-level README.
