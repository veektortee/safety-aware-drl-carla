"""Training curriculum shared by CarlaGymEnv (applies a stage) and the curriculum callback (promotes).

A stage takes effect at the next env.reset(). `num_npc` / `num_walkers` = None means
"use the counts the env was constructed with" (the experiment's --num-npc / --num-pedestrians).
Traffic lights and stop signs are enforced in every stage.
"""

STAGES = [
    dict(name="S0_empty_short", num_npc=0, num_walkers=0, route_min_m=80.0, route_max_m=150.0, crash_penalty=50.0),
    dict(name="S1_empty_full", num_npc=0, num_walkers=0, route_min_m=200.0, route_max_m=400.0, crash_penalty=100.0),
    dict(name="S2_light_traffic", num_npc=15, num_walkers=15, route_min_m=200.0, route_max_m=400.0, crash_penalty=150.0),
    dict(name="S3_full_traffic", num_npc=None, num_walkers=None, route_min_m=200.0, route_max_m=400.0, crash_penalty=300.0),
]

FINAL_STAGE = len(STAGES) - 1
PROMOTE_SUCCESS_RATE = 0.8     # fraction of episodes that reach the route end
PROMOTE_WINDOW = 50            # over the last N episodes of the current stage
MIN_EPISODES_PER_STAGE = 50
