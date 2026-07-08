#!/usr/bin/env python3
"""Run a SMAC scenario with random agents and save rendered PNG frames."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a StarCraft Multi-Agent Challenge scenario and save each "
            "rendered frame as a PNG."
        )
    )
    parser.add_argument("--map", default="8m", help="SMAC map name.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="frames/smac_8m")
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Save one frame every N environment steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap per episode, useful for quick smoke tests.",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--human",
        action="store_true",
        help="Open a pygame window instead of headless rendering.",
    )
    return parser.parse_args()


def choose_random_actions(env, n_agents: int, rng: np.random.Generator) -> list[int]:
    actions = []
    for agent_id in range(n_agents):
        avail_actions = env.get_avail_agent_actions(agent_id)
        avail_actions_ind = np.flatnonzero(avail_actions)
        actions.append(int(rng.choice(avail_actions_ind)))
    return actions


def save_frame(frame: np.ndarray, path: Path) -> None:
    Image.fromarray(frame).save(path)


def main() -> None:
    args = parse_args()

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
    if not args.human:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    from smac.env import StarCraft2Env

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    env = StarCraft2Env(
        map_name=args.map,
        seed=args.seed,
        window_size_x=args.width,
        window_size_y=args.height,
    )

    frame_idx = 0
    try:
        env_info = env.get_env_info()
        n_agents = env_info["n_agents"]
        print(
            f"Map={args.map} agents={n_agents} "
            f"actions={env_info['n_actions']} limit={env_info['episode_limit']}"
        )

        for episode in range(args.episodes):
            env.reset()
            terminated = False
            episode_reward = 0.0
            step = 0

            frame = env.render(mode="rgb_array")
            save_frame(frame, out_dir / f"episode_{episode:03d}_frame_{frame_idx:06d}.png")
            frame_idx += 1

            while not terminated:
                actions = choose_random_actions(env, n_agents, rng)
                reward, terminated, _ = env.step(actions)
                episode_reward += reward
                step += 1

                if step % args.every == 0:
                    frame = env.render(mode="rgb_array")
                    save_frame(
                        frame,
                        out_dir / f"episode_{episode:03d}_frame_{frame_idx:06d}.png",
                    )
                    frame_idx += 1

                if args.max_steps is not None and step >= args.max_steps:
                    break

            print(
                f"Episode {episode}: reward={episode_reward:.3f} "
                f"steps={step} frames={frame_idx}"
            )
    finally:
        env.close()

    print(f"Wrote {frame_idx} PNG frames to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
