"""Setup check: can Waymax read WOMD on this machine? Run inside WSL (Ubuntu), in the venv.

    python scripts/check_setup.py            # reads from Google Cloud (needs gcloud application-default login)
    python scripts/check_setup.py --local    # reads data/womd_val/*-00000-of-00150 instead

Prints one scene's size and a few numbers. No analysis; it only proves the data path works.
"""
import sys, dataclasses, pathlib
import numpy as np
from waymax import config as _config, dataloader

cfg = _config.WOD_1_1_0_VALIDATION
# normalise the bucket prefix ("gs:///" -> "gs://") in the installed config path
cfg = dataclasses.replace(cfg, path=cfg.path.replace("gs:///", "gs://"))
if "--local" in sys.argv:
    root = pathlib.Path(__file__).resolve().parents[1]
    f = root / "data" / "womd_val" / "validation_tfexample.tfrecord-00000-of-00150"
    if not f.exists():
        sys.exit(f"missing {f} - copy it with gsutil first (see README)")
    cfg = dataclasses.replace(cfg, path=str(f))
if "--all" not in sys.argv and "--local" not in sys.argv:
    # read ONE file (~300 scenes) instead of all 150, so the first scene arrives quickly
    cfg = dataclasses.replace(cfg, path=cfg.path.replace("tfrecord@150", "tfrecord-00000-of-00150"))
cfg = dataclasses.replace(cfg, max_num_objects=32)
print("reading:", cfg.path, flush=True)
import time; t0 = time.time()

state = next(dataloader.simulator_state_generator(config=cfg))
traj = state.log_trajectory
valid = np.asarray(traj.valid)
print("objects x timesteps:", valid.shape)            # expect (32, 91): 9.1 s at 10 Hz
print("valid object-steps:", int(valid.sum()))
sdc = int(np.argmax(np.asarray(state.object_metadata.is_sdc)))
v = np.hypot(np.asarray(traj.vel_x)[sdc], np.asarray(traj.vel_y)[sdc])[valid[sdc]]
print(f"self-driving car index {sdc}; speed range {v.min():.1f}-{v.max():.1f} m/s")
print(f"first scene took {time.time() - t0:.0f} s")
print("OK - Waymax can read WOMD on this machine.")
