"""Metrics for audio trajectories without fabricating text-token lengths."""

from miles.utils.tracking_utils import tracking


def log_rollout(rollout_id, args, samples, extra_metrics, rollout_time):
    metrics = {"rollout/step": rollout_id, "rollout/time": rollout_time}
    rewards = [sample.get_reward_value(args) for sample in samples]
    metrics["rollout/rewards"] = sum(rewards) / len(rewards)
    metrics["rollout/moss_samples"] = len(samples)
    metrics.update({f"rollout/{key}": value for key, value in (extra_metrics or {}).items()})
    tracking.log(args, metrics, step_key="rollout/step")
    return True
