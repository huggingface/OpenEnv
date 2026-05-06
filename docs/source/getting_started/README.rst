Interactive Tutorials
=====================

This section provides a hands-on introduction to reinforcement learning (RL) and OpenEnv through a series of interactive tutorials. Whether you're new to RL or looking to learn how OpenEnv simplifies building and deploying environments, these tutorials will guide you through the fundamentals.

**What is OpenEnv?**

OpenEnv is a collaborative effort between **Meta, Hugging Face, Unsloth, GPU Mode, Reflection**, and other industry leaders to standardize reinforcement learning environments. Our goal is to make environment creation as easy and standardized as model sharing on Hugging Face.

Learning Path
-------------

The tutorials are designed to be followed in sequence, building upon concepts from previous lessons:

1. **Introduction & Quick Start** - Understand what OpenEnv is, why it exists, and run your first environment. Includes a comparison with traditional solutions like OpenAI Gym.

2. **Using Environments** - Learn how to connect to environments (Hub, Docker, URL), create AI policies, and run evaluations. Work with different games and multi-player scenarios.

3. **Building & Sharing Environments** - Create your own custom environment from scratch, package it with Docker, and share it on Hugging Face Hub.

4. **Packaging & Deploying** - The complete reference guide for creating, packaging, and deploying custom environments with the ``openenv`` CLI.

5. **Contributing to Hugging Face** - Publish, fork, and contribute to environments hosted as Hugging Face Spaces.

**No GPU Required!** All five tutorials run without a GPU.

For GPU-intensive training workflows, see the :doc:`RL Training Tutorial </tutorials/rl-training-2048>` in the Tutorials section.

Prerequisites
-------------

Before starting, ensure you have:

- Basic Python programming knowledge
- Python 3.11+ installed
- Docker (optional, for container-based deployment)

Running the Tutorials
---------------------

You can run these tutorials locally:

.. code-block:: bash

    # Install OpenEnv
    pip install openenv-core

    # Run the Python scripts
    python plot_01_introduction_quickstart.py

Or view them directly in the documentation with full code output below.

.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   plot_01_introduction_quickstart
   plot_02_using_environments
   plot_03_building_environments
   environment-builder
   contributing-envs
