#!/usr/bin/env python3
"""Interactive cartpole control via OpenEnv.

This example demonstrates using the dm_control OpenEnv client with
the cartpole environment. Use arrow keys to control the cart.

Controls:
    LEFT/RIGHT arrows: Apply force to move cart
    R: Reset environment
    ESC or Q: Quit

Requirements:
    pip install pygame

Usage:
    1. Start the server: uvicorn server.app:app --host 0.0.0.0 --port 8000
    2. Run this script: python examples/cartpole_control.py

    For visual mode (requires working MuJoCo rendering):
        python examples/cartpole_control.py --visual
"""

import argparse
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import DMControlEnv
from models import DMControlAction


def run_headless(env: DMControlEnv, task: str = "balance", max_steps: int = 500):
    """Run cartpole control in headless mode."""
    print("\n=== Headless Mode (OpenEnv Step/Observation Pattern) ===")
    print("This mode demonstrates the OpenEnv API with the cartpole.\n")

    # Reset environment using OpenEnv pattern
    result = env.reset(domain_name="cartpole", task_name=task)
    print(f"Initial observations: {list(result.observation.observations.keys())}")
    print(f"  position: {result.observation.observations.get('position', [])}")
    print(f"  velocity: {result.observation.observations.get('velocity', [])}")

    total_reward = 0.0
    step_count = 0

    print("\nRunning with random actions to demonstrate step/observation pattern...\n")

    while not result.done and step_count < max_steps:
        # Random action in [-1, 1]
        action_value = random.uniform(-1.0, 1.0)

        # Step the environment using OpenEnv pattern
        action = DMControlAction(values=[action_value])
        result = env.step(action)

        # Access observation and reward from result
        total_reward += result.reward or 0.0
        step_count += 1

        # Print progress periodically
        if step_count % 50 == 0:
            pos = result.observation.observations.get("position", [])
            vel = result.observation.observations.get("velocity", [])
            print(
                f"Step {step_count}: reward={result.reward:.3f}, "
                f"total={total_reward:.2f}, done={result.done}"
            )
            print(f"  position={pos}, velocity={vel}")

    print(f"\nEpisode finished: {step_count} steps, total reward: {total_reward:.2f}")


def run_interactive(env: DMControlEnv, task: str = "balance"):
    """Run interactive control with keyboard input via pygame."""
    import pygame

    print("\n=== Interactive Mode (OpenEnv Step/Observation Pattern) ===")
    print("Use LEFT/RIGHT arrows to control cart, R to reset, ESC to quit.\n")

    # Reset environment using OpenEnv pattern
    result = env.reset(domain_name="cartpole", task_name=task)
    print(f"Initial observations: {list(result.observation.observations.keys())}")

    # Initialize pygame for keyboard input (minimal window)
    pygame.init()
    screen = pygame.display.set_mode((400, 100))
    pygame.display.set_caption("Cartpole Control - Arrow keys to move, R to reset")
    clock = pygame.time.Clock()

    # Font for display
    font = pygame.font.Font(None, 24)

    running = True
    total_reward = 0.0
    step_count = 0

    print("\nControls:")
    print("  LEFT/RIGHT arrows: Move cart")
    print("  R: Reset environment")
    print("  ESC or Q: Quit\n")

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    result = env.reset(domain_name="cartpole", task_name=task)
                    total_reward = 0.0
                    step_count = 0
                    print("Environment reset")

        # Check for held keys (for continuous control)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action_value = -1.0
        elif keys[pygame.K_RIGHT]:
            action_value = 1.0
        else:
            action_value = 0.0

        # Step the environment using OpenEnv pattern
        action = DMControlAction(values=[action_value])
        result = env.step(action)

        # Track reward from result
        total_reward += result.reward or 0.0
        step_count += 1

        # Check if episode is done
        if result.done:
            print(
                f"Episode finished! Steps: {step_count}, "
                f"Total reward: {total_reward:.2f}"
            )
            # Auto-reset on done
            result = env.reset(domain_name="cartpole", task_name=task)
            total_reward = 0.0
            step_count = 0

        # Update display
        direction = (
            "<--" if action_value < 0 else ("-->" if action_value > 0 else "---")
        )
        screen.fill((30, 30, 30))
        text = font.render(
            f"Step: {step_count} | Reward: {total_reward:.1f} | {direction}",
            True,
            (255, 255, 255),
        )
        screen.blit(text, (10, 40))
        pygame.display.flip()

        # Print progress periodically
        if step_count % 200 == 0 and step_count > 0:
            print(f"Step {step_count}: Total reward: {total_reward:.2f}")

        # Cap at 30 FPS
        clock.tick(30)

    pygame.quit()
    print(f"Session ended. Final reward: {total_reward:.2f}")


def run_visual(env: DMControlEnv, task: str = "balance"):
    """Run with pygame visualization showing rendered frames."""
    import base64
    import io

    import pygame

    print("\n=== Visual Mode (OpenEnv Step/Observation Pattern) ===")

    # Reset environment with rendering enabled
    result = env.reset(domain_name="cartpole", task_name=task, render=True)
    print(f"Initial observations: {list(result.observation.observations.keys())}")

    # Get first frame to determine window size
    if result.observation.pixels is None:
        print("Error: Server did not return rendered pixels.")
        print("Make sure the server supports render=True")
        print("\nTry running in interactive mode (default) instead.")
        sys.exit(1)

    # Decode base64 PNG to pygame surface
    png_data = base64.b64decode(result.observation.pixels)
    frame = pygame.image.load(io.BytesIO(png_data))
    frame_size = frame.get_size()

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode(frame_size)
    pygame.display.set_caption(
        "Cartpole (OpenEnv) - Arrow Keys to Move, R to Reset, ESC to Quit"
    )
    clock = pygame.time.Clock()

    print("Controls:")
    print("  LEFT/RIGHT arrows: Move cart")
    print("  R: Reset environment")
    print("  ESC or Q: Quit")

    running = True
    total_reward = 0.0
    step_count = 0

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    result = env.reset(
                        domain_name="cartpole", task_name=task, render=True
                    )
                    total_reward = 0.0
                    step_count = 0
                    print("Environment reset")

        # Check for held keys (for continuous control)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action_value = -1.0
        elif keys[pygame.K_RIGHT]:
            action_value = 1.0
        else:
            action_value = 0.0

        # Step the environment using OpenEnv pattern
        action = DMControlAction(values=[action_value])
        result = env.step(action, render=True)

        # Track reward from result
        total_reward += result.reward or 0.0
        step_count += 1

        # Check if episode is done
        if result.done:
            print(
                f"Episode finished! Steps: {step_count}, "
                f"Total reward: {total_reward:.2f}"
            )
            result = env.reset(domain_name="cartpole", task_name=task, render=True)
            total_reward = 0.0
            step_count = 0

        # Render the frame from observation pixels
        if result.observation.pixels:
            png_data = base64.b64decode(result.observation.pixels)
            frame = pygame.image.load(io.BytesIO(png_data))
            screen.blit(frame, (0, 0))
            pygame.display.flip()

        # Print progress periodically
        if step_count % 200 == 0 and step_count > 0:
            print(f"Step {step_count}: Total reward: {total_reward:.2f}")

        # Cap at 30 FPS
        clock.tick(30)

    pygame.quit()
    print(f"Session ended. Final reward: {total_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive cartpole control via OpenEnv"
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Enable pygame visualization with rendered frames",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no pygame, automated control)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps for headless mode (default: 500)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="balance",
        choices=["balance", "balance_sparse", "swingup", "swingup_sparse"],
        help="Cartpole task (default: balance)",
    )
    args = parser.parse_args()

    server_url = "http://localhost:8000"
    print(f"Connecting to {server_url}...")

    try:
        with DMControlEnv(base_url=server_url) as env:
            print("Connected!")

            # Get environment state
            state = env.state()
            print(f"Domain: {state.domain_name}, Task: {state.task_name}")
            print(f"Action spec: {state.action_spec}")

            if args.headless:
                run_headless(env, task=args.task, max_steps=args.max_steps)
            elif args.visual:
                run_visual(env, task=args.task)
            else:
                run_interactive(env, task=args.task)

    except ConnectionError as e:
        print(f"Failed to connect: {e}")
        print("\nMake sure the server is running:")
        print("  cd OpenEnv")
        print(
            "  PYTHONPATH=src:envs uvicorn envs.dm_control_env.server.app:app --port 8000"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
