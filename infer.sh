#!/usr/bin/env bash
python3 -m hred.infer \
    --save_dir=model/nwords \
    --load_checkpoint_path=model/nwords \
    --infer_sample="six thousand seven hundred and fifty four	six thousand seven hundred and fifty five	six thousand seven hundred and fifty six"
