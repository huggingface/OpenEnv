#!/usr/bin/env python3
"""Interactive quadruped control via OpenEnv.

This example demonstrates using the dm_control OpenEnv client with
the quadruped environment. Press SPACE to apply random forces to the joints.

Controls:
    SPACE: Apply random force to all joints
    R: Reset environment
    ESC or Q: Quit

Requirements:
    pip install pygame

Usage:
    1. Start the server: uvicorn server.app:app --host 0.0.0.0 --port 8000
    2. Run this script: python examples/quadruped_control.py

    For visual mode (requires working MuJoCo rendering):
        python examples/quadruped_control.py --visual
"""

import argparse
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import DMControlEnv
from models import DMControlAction


def get_action_dim(env: DMControlEnv) -> int:
    """Get the action dimension from the environment state."""
    state = env.state()
    action_spec = state.action_spec
    if action_spec and "shape" in action_spec:
        shape = action_spec["shape"]
        if isinstance(shape, list) and len(shape) > 0:
            return shape[0]
    # Quadruped default: 12 actuators (3 per leg x 4 legs)
    return 12


def generate_random_action(action_dim: int, magnitude: float = 1.0) -> DMControlAction:
    """Generate a random action with values in [-magnitude, magnitude]."""
    values = [random.uniform(-magnitude, magnitude) for _ in range(action_dim)]
    return DMControlAction(values=values)


def generate_zero_action(action_dim: int) -> DMControlAction:
    """Generate a zero action (no force applied)."""
    return DMControlAction(values=[0.0] * action_dim)


def run_headless(env: DMControlEnv, max_steps: int = 1000):
    """Run quadruped control in headless mode."""
    print("\n=== Headless Mode (OpenEnv Step/Observation Pattern) ===")
    print("This mode demonstrates the OpenEnv API with the quadruped.\n")

    # Reset environment using OpenEnv pattern
    result = env.reset(domain_name="quadruped", task_name="walk")
    print(f"Initial observations: {list(result.observation.observations.keys())}")

    # Get action dimension
    action_dim = get_action_dim(env)
    print(f"Action dimension: {action_dim}")

    total_reward = 0.0
    step_count = 0

    print("\nRunning with periodic random forces...")
    print("Every 50 steps, a random force burst is applied.\n")

    while not result.done and step_count < max_steps:
        # Apply random force every 50 steps, otherwise zero action
        if step_count % 50 < 10:
            # Random force burst for 10 steps
            action = generate_random_action(action_dim, magnitude=0.5)
        else:
            # No force
            action = generate_zero_action(action_dim)

        # Step the environment using OpenEnv pattern
        result = env.step(action)

        # Access observation and reward from result
        total_reward += result.reward or 0.0
        step_count += 1

        # Print progress periodically
        if step_count % 100 == 0:
            # Get some observation values
            egocentric_state = result.observation.observations.get(
                "egocentric_state", []
            )
            print(
                f"Step {step_count}: reward={result.reward:.3f}, "
                f"total={total_reward:.2f}, done={result.done}"
            )
            if egocentric_state:
                print(f"  egocentric_state (first 5): {egocentric_state[:5]}")

    print(f"\nEpisode finished: {step_count} steps, total reward: {total_reward:.2f}")


def run_interactive(env: DMControlEnv):
    """Run interactive control with keyboard input via pygame."""
    import pygame

    print("\n=== Interactive Mode (OpenEnv Step/Observation Pattern) ===")
    print("Press SPACE to apply random force, R to reset, ESC to quit.\n")

    # Reset environment using OpenEnv pattern
    result = env.reset(domain_name="quadruped", task_name="walk")
    print(f"Initial observations: {list(result.observation.observations.keys())}")

    # Get action dimension
    action_dim = get_action_dim(env)
    print(f"Action dimension: {action_dim}")

    # Initialize pygame for keyboard input (minimal window)
    pygame.init()
    screen = pygame.display.set_mode((400, 100))
    pygame.display.set_caption("Quadruped Control - SPACE for random force, R to reset")
    clock = pygame.time.Clock()

    # Draw instructions on the window
    font = pygame.font.Font(None, 24)

    running = True
    total_reward = 0.0
    step_count = 0
    apply_random_force = False

    print("\nControls:")
    print("  SPACE: Apply random force to joints")
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
                    result = env.reset(domain_name="quadruped", task_name="walk")
                    total_reward = 0.0
                    step_count = 0
                    print("Environment reset")

        # Check for held keys
        keys = pygame.key.get_pressed()
        apply_random_force = keys[pygame.K_SPACE]

        # Generate action based on input
        if apply_random_force:
            action = generate_random_action(action_dim, magnitude=2.0)
        else:
            action = generate_zero_action(action_dim)

        # Step the environment using OpenEnv pattern
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
            result = env.reset(domain_name="quadruped", task_name="walk")
            total_reward = 0.0
            step_count = 0

        # Update display
        screen.fill((30, 30, 30))
        status = "FORCE!" if apply_random_force else "idle"
        text = font.render(
            f"Step: {step_count} | Reward: {total_reward:.1f} | {status}",
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


def run_visual(env: DMControlEnv):
    """Run with pygame visualization showing rendered frames."""
    import base64
    import io

    import pygame

    print("\n=== Visual Mode (OpenEnv Step/Observation Pattern) ===")

    # Reset environment with rendering enabled
    result = env.reset(domain_name="quadruped", task_name="walk", render=True)
    print(f"Initial observations: {list(result.observation.observations.keys())}")

    # Get action dimension
    action_dim = get_action_dim(env)
    print(f"Action dimension: {action_dim}")

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
        "Quadruped (OpenEnv) - SPACE for random force, R to Reset, ESC to Quit"
    )
    clock = pygame.time.Clock()

    print("Controls:")
    print("  SPACE: Apply random force to joints")
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
                        domain_name="quadruped", task_name="walk", render=True
                    )
                    total_reward = 0.0
                    step_count = 0
                    print("Environment reset")

        # Check for held keys
        keys = pygame.key.get_pressed()
        apply_random_force = keys[pygame.K_SPACE]

        # Generate action based on input
        if apply_random_force:
            action = generate_random_action(action_dim, magnitude=2.0)
        else:
            action = generate_zero_action(action_dim)

        # Step the environment using OpenEnv pattern
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
            result = env.reset(domain_name="quadruped", task_name="walk", render=True)
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
        description="Interactive quadruped control via OpenEnv"
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
        default=1000,
        help="Maximum steps for headless mode (default: 1000)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="walk",
        choices=["walk", "run", "escape", "fetch"],
        help="Quadruped task (default: walk)",
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
                run_headless(env, max_steps=args.max_steps)
            elif args.visual:
                run_visual(env)
            else:
                run_interactive(env)

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
