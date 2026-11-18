"""Training curriculum shared by CarlaGymEnv (applies a stage) and the curriculum callback (promotes).

Six stages, from empty roads to dense traffic. A stage takes effect at the next env.reset().
`num_npc` / `num_walkers` are the stage's counts; the env caps them by the counts it was constructed
with (the experiment's --num-npc / --num-pedestrians) and, for vehicles, by the current map's spawn
points (small towns cannot hold 300 cars). Traffic lights and stop signs are enforced in every stage.

Promotion (commons/curriculum.py): a stage is left as soon as the success gate is met
(>= PROMOTE_SUCCESS_RATE over the last PROMOTE_WINDOW episodes, at least MIN_EPISODES_PER_STAGE in
the stage), and at the latest when the run reaches PROMOTE_BY_STEP[stage] total steps. The step
schedule spreads the whole curriculum over CURRICULUM_TOTAL_STEPS: the final stage gets the last
60k steps of a 200k-step run.

crash_penalty (the reward's C) grows with the traffic; stages S0-S2 keep their earlier values so
checkpoints trained on them continue on the same reward scale.
"""

STAGES = [
    dict(name="S0_empty_short", num_npc=0, num_walkers=0, route_min_m=80.0, route_max_m=150.0, crash_penalty=50.0),
    dict(name="S1_empty_full", num_npc=0, num_walkers=0, route_min_m=200.0, route_max_m=400.0, crash_penalty=100.0),
    dict(name="S2_light_traffic", num_npc=15, num_walkers=15, route_min_m=200.0, route_max_m=400.0, crash_penalty=150.0),
    dict(name="S3_medium_traffic", num_npc=60, num_walkers=60, route_min_m=200.0, route_max_m=400.0, crash_penalty=200.0),
    dict(name="S4_heavy_traffic", num_npc=150, num_walkers=150, route_min_m=200.0, route_max_m=400.0, crash_penalty=250.0),
    dict(name="S5_dense_traffic", num_npc=300, num_walkers=300, route_min_m=200.0, route_max_m=400.0, crash_penalty=300.0),
]

FINAL_STAGE = len(STAGES) - 1
PROMOTE_SUCCESS_RATE = 0.8     # fraction of episodes that reach the route end
PROMOTE_WINDOW = 50            # over the last N episodes of the current stage
MIN_EPISODES_PER_STAGE = 50

CURRICULUM_TOTAL_STEPS = 200_000
# Total-step marks by which S0..S4 are left even if the success gate was not met.
PROMOTE_BY_STEP = [15_000, 35_000, 65_000, 100_000, 140_000]
assert len(PROMOTE_BY_STEP) == FINAL_STAGE and PROMOTE_BY_STEP == sorted(PROMOTE_BY_STEP)


def scheduled_stage(num_timesteps: int) -> int:
    """The lowest stage the step schedule allows at `num_timesteps` (number of marks passed)."""
    return sum(int(num_timesteps) >= mark for mark in PROMOTE_BY_STEP)
