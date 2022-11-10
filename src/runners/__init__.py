REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .episode_runner_multi import EpisodeRunner_Multi
REGISTRY["episode_multi"] = EpisodeRunner_Multi

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner
