## Release PR: v<!-- VERSION -->

## Release Checklist

### Before opening this PR
- [ ] `pyproject.toml` version changed from `X.Y.Z.dev0` → `X.Y.Z` (no `.dev0` suffix)
- [ ] `hf-staging/` is NOT in this PR's diff
- [ ] No `print()`, `breakpoint()`, or `TODO` in release-critical paths

### CI gates (must be green before merge)
- [ ] `test` passes on Python 3.11
- [ ] `test` passes on Python 3.12
- [ ] `lint` passes (usort + ruff)

### TestPyPI validation (before merging)
- [ ] Manual dispatch of `publish-pypi-core.yml` with `use_test_pypi=true` from this branch
- [ ] `pip install --index-url https://test.pypi.org/simple/ openenv-core==X.Y.Z` installs cleanly
- [ ] `python -c "import openenv; print(openenv.__version__)"` prints `X.Y.Z`

### Post-merge steps (author only)
- [ ] Tag `vX.Y.Z` pushed: `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push origin vX.Y.Z`
- [ ] GitHub Release published (triggers real PyPI publish via `release: published`)
- [ ] PyPI publish Actions job completed successfully
- [ ] `pip install openenv-core==X.Y.Z` from production PyPI verified
- [ ] `auto-bump-version.yml` created `bump/X.Y.(Z+1).dev0` PR
