# Contributing to NS-ARC

Thank you for your interest in contributing to NS-ARC! 🎉

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. Create a new issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Rust/Python versions)

### Suggesting Features

1. Open a GitHub Discussion or Issue
2. Describe the feature and its use case
3. Explain how it fits with the project goals

### Code Contributions

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `cargo test`
5. Commit with clear messages: `git commit -m "Add feature X"`
6. Push and open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ns-arc.git
cd ns-arc

# Build
cd ns-arc-demon
cargo build

# Run tests
cargo test

# Check formatting
cargo fmt --check
cargo clippy
```

## Code Style

- **Rust**: Follow `rustfmt` defaults
- **Python**: Follow PEP 8
- **Commits**: Use conventional commits (`feat:`, `fix:`, `docs:`)

## Pull Request Guidelines

- Keep PRs focused on a single feature/fix
- Include tests for new functionality
- Update documentation as needed
- Ensure CI passes

## Questions?

Open a Discussion or reach out to maintainers.

Thank you for contributing! 🙏
