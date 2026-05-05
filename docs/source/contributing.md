# Contributing to OpenEnv

We welcome contributions from the community! OpenEnv is an open-source project and we're excited to have you join us.

## Ways to Contribute

### 🐛 Report Bugs

Found a bug? Please [open an issue](https://github.com/meta-pytorch/OpenEnv/issues/new) with:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)

### 💡 Suggest Features

Have an idea? Open a [feature request](https://github.com/meta-pytorch/OpenEnv/issues/new) describing:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

### 🌍 Add an Environment

One of the best ways to contribute is to add a new environment! See the [Building Environments](guides/first-environment.md) guide to get started.

### 📝 Improve Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples
- Write tutorials
- Translate content

### 🔧 Submit Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/OpenEnv.git
cd OpenEnv

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
```

## Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Write docstrings for public functions
- Add type hints
- Write tests for new functionality

## Community

- [Discord](https://discord.gg/YsTYBh6PD9) - Chat with the community
- [GitHub Discussions](https://github.com/meta-pytorch/OpenEnv/discussions) - Ask questions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing! 🙏
