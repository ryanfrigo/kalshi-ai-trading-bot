# Contributing to Kalshi AI Trading Bot

Thank you for your interest in contributing to the Kalshi AI Trading Bot! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug Reports**: Report bugs and issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests with code changes
- **Documentation**: Improve or add documentation
- **Testing**: Add tests or improve test coverage
- **Performance**: Optimize code or improve performance

### Before You Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up the development environment** (see Installation section)
4. **Create a feature branch** for your changes

## 🛠️ Development Setup

### Prerequisites

- Python 3.12+
- Git
- Kalshi API account (for testing)
- xAI API key (for AI features)

### Local Development Setup

1. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/kalshi-ai-trading-bot.git
   cd kalshi-ai-trading-bot
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Configure environment**
   ```bash
   cp env.template .env
   # Edit .env with your API keys
   ```

5. **Initialize database**
   ```bash
   python init_database.py
   ```

## 📝 Code Style and Standards

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black formatter default)
- **Import sorting**: Use `isort`
- **Type hints**: Required for all functions and methods
- **Docstrings**: Use Google-style docstrings

### Code Formatting

We use automated tools for code formatting:

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Type checking with mypy
mypy src/
```

### Pre-commit Hooks

Set up pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_decide.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run integration tests
pytest tests/integration/
```

### Writing Tests

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **Mock external APIs**: Don't make real API calls in tests
- **Test coverage**: Aim for >80% coverage

### Test Structure

```python
def test_function_name():
    """Test description."""
    # Arrange
    input_data = "test"
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result == "expected"
```

## 🔧 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write your code following the style guidelines
- Add tests for new functionality
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run tests
pytest

# Check code style
black --check src/
isort --check-only src/

# Type checking
mypy src/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

Use conventional commit messages:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 📋 Pull Request Guidelines

### Before Submitting

1. **Ensure tests pass**: All tests should pass locally
2. **Check code style**: Run formatting tools
3. **Update documentation**: Update README, docstrings, etc.
4. **Add tests**: Include tests for new functionality
5. **Check for sensitive data**: Ensure no API keys or secrets are included

### Pull Request Template

Use this template when creating a pull request:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No sensitive data included
```

## 🐛 Bug Reports

### Before Reporting

1. **Check existing issues**: Search for similar issues
2. **Reproduce the bug**: Ensure you can reproduce it consistently
3. **Check logs**: Look at error logs and stack traces

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., macOS 14.0]
- Python: [e.g., 3.12.0]
- Bot Version: [e.g., commit hash]

## Logs
Relevant error logs or stack traces
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the requested feature

## Use Case
Why this feature would be useful

## Proposed Implementation
Optional: How you think it could be implemented

## Alternatives Considered
Other approaches you've considered
```

## 🔒 Security

### Security Guidelines

- **Never commit API keys or secrets**
- **Use environment variables** for sensitive data
- **Validate all inputs** to prevent injection attacks
- **Follow secure coding practices**
- **Report security issues privately**

### Reporting Security Issues

If you find a security vulnerability:

1. **Don't create a public issue**
2. **Email the maintainers privately**
3. **Provide detailed information**
4. **Allow time for response**

## 📚 Documentation

### Documentation Standards

- **Clear and concise**: Write for clarity
- **Include examples**: Provide usage examples
- **Keep updated**: Update docs with code changes
- **Use proper formatting**: Follow markdown guidelines

### Documentation Areas

- **README.md**: Project overview and quick start
- **Code docstrings**: Function and class documentation
- **API documentation**: External API documentation
- **Configuration docs**: Settings and configuration
- **Tutorials**: Step-by-step guides

### Project Docs

Use these repo docs as the source of truth for operation, testing, and AI-assisted development:

- [Runbook](docs/RUNBOOK.md) — how to start, stop, and verify the bot
- [Testing Guide](docs/TESTING.md) — what to test and what to report in a PR
- [AI Collaboration Guide](docs/AI_COLLABORATION.md) — branch, PR, and agent workflow
- [AGENTS.md](AGENTS.md) — repo-wide operating rules for humans and AI agents

## 🏷️ Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## 📞 Getting Help

### Questions and Support

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check existing docs first

### Community Guidelines

- **Be respectful**: Treat others with respect
- **Be helpful**: Help other contributors
- **Be patient**: Maintainers are volunteers
- **Follow the code of conduct**

## 🎉 Recognition

Contributors will be recognized in:

- **README.md**: List of contributors
- **Release notes**: Credit for significant contributions
- **GitHub contributors page**: Automatic recognition

---

Thank you for contributing to the Kalshi AI Trading Bot! Your contributions help make this project better for everyone. 
