# av-sim-credibility

How far can simulated road users be trusted? This repository compares closed-loop traffic simulation with recorded driving, one scenario type at a time, and turns the comparison into a decision about what the simulation may be used for.

Each recorded scene is run twice: as recorded, and with the other vehicles handed to a simulated driver. For every vehicle the code computes the minimum time-to-collision (TTC) and any collision in both versions, groups vehicles into five scenario families, and reports per family:

- **bias δ**: typical ratio of simulated to recorded minimum TTC (1 = no bias),
- **scatter ω**: spread of that ratio between vehicles (0 = every vehicle matches),
- **tail ratio R**: how many times as often simulated vehicles come within 1.5 s of a collision (1 = same frequency),
- the area between the two distributions, and who collided with whom.

Those numbers are compared with the credibility a stated decision requires (following the risk-informed logic of ASME V&V 40), and, if the simulation qualifies, a power prior says how much simulated exposure is worth in a Bayesian event-rate estimate.

**Technical report:** [`docs/technical_report.pdf`](docs/technical_report.pdf) ([Word](docs/technical_report.docx), [Markdown](docs/technical_report.md))

## Results on the Waymo Open Motion Dataset

2,000 scenes of WOMD v1.1 (validation split), 13,195 moving vehicles, simulated in closed loop with [Waymax](https://github.com/waymo-research/waymax). The simulated drivers are simple baselines: Waymax's `IDMRoutePolicy` with default and with hand-picked cautious parameters, and a constant-speed model. Log replay is the check.

| Setting (pooled over families) | δ | ω | R (TTC < 1.5 s) | Ran into something ahead |
|---|---|---|---|---|
| Log replay (check) | 1.000 | 0.000 | 1.00 | 0.36% (same as the recording) |
| IDM, cautious parameters | 0.94 | 1.10 | 1.04 | 4.8% |
| IDM, default parameters | 0.45 | 1.62 | 3.52 | 7.1% |
| Constant speed | 0.07 | 2.48 | 3.97 | 22.3% |

- Log replay gives δ = 1, ω = 0 and R = 1 in every family, so the pipeline adds no error of its own.
- With default parameters, close calls were 1.5 to 4.4 times as frequent as in the recordings, depending on the family.
- The cautious parameters matched the pooled close-call rate but not individual vehicles (ω = 1.10), and produced about half the recorded close calls in lane changes (R = 0.51).
- In every setting the pooled numbers looked better than the worst family.

These are rule-based baseline agents. The results are not an assessment of Waymax as a platform, of learned simulation agents, or of any company's internal simulation, none of which were tested.

## Reproduce

```bash
pip install -r requirements.txt
python tests/test_core.py            # 10 tests against hand-worked answers
python scripts/synthetic_pilot.py    # the method on synthetic data with known errors
python scripts/make_figures.py       # report figures from the aggregate results in results/
python scripts/build_report.py       # Word and PDF report (needs pandoc and Playwright)
```

The real-data runs need Waymax and registered access to the Waymo Open Dataset:

```bash
python scripts/check_setup.py --local                      # confirms Waymax can read the data
for a in log idm idm_cautious constant_speed; do
  python scripts/womd_pilot.py --scenes 2000 --agent $a --local ~/womd
done
python scripts/compare_agents.py
```

`--local` points at a folder holding the `validation_tfexample.tfrecord-XXXXX-of-00150` files; without it the scripts read from the dataset's cloud bucket.

## Layout

```
src/avcred/      measures, WOMD helpers, scenario rules, validation metrics, credibility, evidence
scripts/         synthetic check, real-data runs, comparison, figures, report build
tests/           unit tests
results/         aggregate results (per-vehicle files stay local, see .gitignore)
docs/            technical report and figures
```

## Data and licence

Results are derived from the [Waymo Open Dataset](https://waymo.com/open), used under its licence for non-commercial research. No scene data or per-vehicle values are redistributed; `results/womd_agents_*.csv` is excluded from the repository. Code: MIT (see `LICENSE`).

Parts of the code were written with AI assistance; the study design, the runs, the checks and the interpretation are the author's.

Related: [av-sim-vnv-evidence](https://github.com/mohalbati/av-sim-vnv-evidence) uses the per-vehicle results of this study to measure how much real data simulation saves when the two are combined in one estimate.

Mohammad Albati, Ph.D.
