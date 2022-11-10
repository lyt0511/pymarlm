from functools import partial
import sys
import os
from .multiagentenv import MultiAgentEnv
from .starcraft2.starcraft2 import StarCraft2Env
from .starcraft2.starcraft2multi import StarCraft2EnvMulti


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2_multi"] = partial(env_fn, env=StarCraft2EnvMulti)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))




