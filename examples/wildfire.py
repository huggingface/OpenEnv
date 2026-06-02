#!/usr/bin/env python3
"""
Simple example demonstrating Wildfire Environment usage.

This example shows how to:
1. Connect to a Wildfire environment
2. Reset the environment
3. Take strategic actions (water, firebreak, wait)
4. Monitor fire spread and containment
5. Visualize the grid state

Usage:
    # First, start the server:
    python -m envs.wildfire_env.server.app

    # Then run this script:
    python examples/wildfire.py
"""

import sys
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.wildfire_env import WildfireEnv, WildfireAction
from envs.wildfire_env.client import render_grid


def simple_agent_strategy(obs):
    """
    Simple firefighting strategy:
    - Target burning cells with water if available
    - Build firebreaks near fires if water is depleted
    - Otherwise wait
    """
    # Find burning cells
    burning_cells = []
    for y in range(obs.height):
        for x in range(obs.width):
            idx = y * obs.width + x
            if obs.grid[idx] == 2:  # burning
                burning_cells.append((x, y))

    if not burning_cells:
        return WildfireAction(action="wait")

    # Pick a random burning cell to target
    target_x, target_y = random.choice(burning_cells)

    # Use water if available, otherwise use firebreak
    if obs.remaining_water > 0:
        return WildfireAction(action="water", x=target_x, y=target_y)
    elif obs.remaining_breaks > 0:
        # Build firebreak adjacent to fire
        return WildfireAction(action="break", x=target_x, y=target_y)
    else:
        return WildfireAction(action="wait")


def main():
    """Run a wildfire containment episode."""
    # Connect to the Wildfire environment server
    print("Connecting to Wildfire environment...")
    print("Note: Make sure the server is running with: python -m envs.wildfire_env.server.app")

    # Connect to local server
    env = WildfireEnv(base_url="http://localhost:8000")

    try:
        # Reset the environment
        print("\nResetting environment...")
        result = env.reset()
        obs = result.observation

        print(f"\n🌲 Wildfire Containment Mission Started!")
        print(f"Grid size: {obs.width}x{obs.height}")
        print(f"Initial fires: {obs.burning_count}")
        print(f"Wind direction: {obs.wind_dir}")
        print(f"Humidity: {obs.humidity:.2f}")
        print(f"Water capacity: {obs.remaining_water}")
        print(f"Firebreak materials: {obs.remaining_breaks}")

        # Print initial grid
        print("\nInitial state:")
        print(render_grid(obs))
        print("\nLegend: ⬛=ash 🟩=fuel 🟥=fire 🟫=firebreak 🟦=water")

        # Run episode
        print("\n" + "="*60)
        print("Starting containment operations...")
        print("="*60)

        episode_reward = 0
        step_count = 0
        max_steps = 50  # Limit steps for demo

        while not result.done and step_count < max_steps:
            # Choose action using simple strategy
            action = simple_agent_strategy(obs)

            # Take action
            result = env.step(action)
            obs = result.observation
            episode_reward += result.reward or 0
            step_count += 1

            # Print progress every 5 steps
            if step_count % 5 == 0 or result.done:
                print(f"\n--- Step {step_count} ---")
                print(f"Action: {action.action}" +
                      (f" at ({action.x}, {action.y})" if action.x is not None else ""))
                print(f"Reward: {result.reward:.3f} | Total: {episode_reward:.2f}")
                print(f"Fires: {obs.burning_count} | Burned: {obs.burned_count}")
                print(f"Water left: {obs.remaining_water} | Breaks left: {obs.remaining_breaks}")
                print(render_grid(obs))

            if result.done:
                break

        # Episode summary
        print("\n" + "="*60)
        print("🏁 EPISODE COMPLETE")
        print("="*60)

        if obs.burning_count == 0:
            print("✅ SUCCESS! All fires have been extinguished!")
        else:
            print(f"⚠️  Episode ended with {obs.burning_count} fires still burning")

        print(f"\nFinal Statistics:")
        print(f"  Steps taken: {step_count}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Cells burned: {obs.burned_count}")
        print(f"  Cells saved: {obs.width * obs.height - obs.burned_count}")
        print(f"  Water used: {result.observation.remaining_water} remaining (started with more)")
        print(f"  Firebreaks used: {result.observation.remaining_breaks} remaining")

        # Get environment state
        state = env.state()
        print(f"\n📊 Environment State:")
        print(f"  Episode ID: {state.episode_id}")
        print(f"  Total burned: {state.total_burned}")
        print(f"  Total extinguished: {state.total_extinguished}")
        print(f"  Final wind: {state.wind_dir}")
        print(f"  Final humidity: {state.humidity:.2f}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the Wildfire server is running:")
        print("  python -m envs.wildfire_env.server.app")

    finally:
        # Cleanup
        print("\nClosing environment...")
        env.close()
        print("Done!")


if __name__ == "__main__":
    main()
