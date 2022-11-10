from functools import partial
from .smac.env import MultiAgentEnv, StarCraft2Env, StarCraft2EnvMulti
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2_multi"] = partial(env_fn, env=StarCraft2EnvMulti)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
