# Held-out plasticity replicate

This small synthetic study asks whether a search-assisted value learner keeps
more capacity to learn after a task shift than a value-only learner does.

The two arms were first matched on the trajectory of their return and policy
KL during an initial shift. `held_out_trajectory.csv` then records how each arm
learned across five updates on a second, unseen shift. There are three fixed
synthetic seeds per arm. The values are public challenge data, not measurements
from a private project and not evidence about a deployed model.

The bounded Experiment in RCP should:

1. verify all 30 rows and all six seed-arm trajectories are present;
2. verify the arm-level first-shift return and KL paths stay within the declared
   0.02 absolute matching tolerance at every update;
3. calculate a least-squares second-shift learning slope for every seed;
4. compare the mean slope between arms without changing the broader Hypothesis
   standing; and
5. save a visual result artifact that shows the six second-shift curves and the
   matching diagnostics.

Run the deterministic reference analysis from this repository:

```bash
python3 study/analyze_held_out.py study/held_out_trajectory.csv
```

The script is intentionally standard-library-only and prints one JSON object.
It never edits RCP state; the Experiment agent remains responsible for a valid
Patch, its scoped interpretation, and any visual artifact.
