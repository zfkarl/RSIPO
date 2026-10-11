"""Keep engine RNG configuration separate from per-request sampling options."""


def rollout_sampling_overrides(config, sampling_params_cls):
    """Copy sampling controls without resetting every request to config.seed.

    A fixed engine seed keeps the run reproducible while its RNG advances across
    requests. SamplingParams.seed must retain its default None so equal prompts
    in a GRPO rollout group do not all use an identical random stream.
    """
    defaults = sampling_params_cls()
    return {
        str(key): config.get(key)
        for key in config.keys()
        if str(key) != "seed" and hasattr(defaults, str(key))
    }
