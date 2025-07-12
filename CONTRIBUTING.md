# Contributing to Scikit-NeuroMSI

Thank you for contributing to `scikit-neuromsi`, a Python framework for modeling multisensory integration in computational neuroscience. Follow these guidelines to ensure smooth collaboration.

## Setup
1. **Fork and Clone**:
   ```bash
   git clone https://github.com/<your-username>/scikit-neuromsi.git
   cd scikit-neuromsi
   ```
2. **Install Dependencies**:
   Use a virtual environment and install:
   ```bash
   pip install -e .
   pip install tox
   ```

## Running Tests
All pull requests must pass the test suite via `tox`:
```bash
tox
```
Run specific environments if needed (e.g., `tox -e py310`). Ensure tests pass locally to align with GitHub Actions CI.

## Contribution Process
1. **Report Issues**: Open an issue on [GitHub Issues](https://github.com/renatoparedes/scikit-neuromsi/issues) with clear details (steps, expected vs. actual behavior).
2. **Submit Pull Requests**:
   - Create a branch: `git checkout -b feature/<name>` or `fix/<name>`.
   - Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).
   - Update tests in `tests/` and documentation in `docs/` if needed.
   - Run `tox` to confirm tests pass.
   - Commit with [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat: add new model`, `fix: docstring issue, closes #123`).
   - Push and open a PR to the `main` branch with a summary and issue reference.
3. **Code Review**: Address feedback; PRs must pass GitHub Actions CI to merge.

## Documentation and Testing
- Update docstrings and Sphinx docs (`docs/`) for API changes. Preview locally:
  ```bash
  cd docs
  make html
  ```
- Add `pytest` tests in `tests/` for new features or fixes.
- Check coverage: `tox -e coverage`.

## Getting Help
Open an issue or contact Renato Paredes at paredesrenato92@gmail.com.

Thank you for advancing multisensory integration modeling!