# Contributing to Fractal-Attention Analysis

Thank you for your interest in contributing to the Fractal-Attention Analysis (FAA) framework! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## 🤝 Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

## 🚀 Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your changes
5. Make your changes
6. Test your changes
7. Submit a pull request

## 💻 Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) CUDA-capable GPU for testing GPU features

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fractal-attention-analysis.git
cd fractal-attention-analysis

# Add upstream remote
git remote add upstream https://github.com/ross-sec/fractal_attention_analysis.git

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Install package in development mode with all dependencies
pip install -e ".[dev,docs,gpu,all]"

# Install pre-commit hooks
pre-commit install
```

## 🔧 Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-metric` - for new features
- `fix/attention-extraction-bug` - for bug fixes
- `docs/update-readme` - for documentation
- `refactor/improve-performance` - for refactoring

### Commit Messages

Follow conventional commits format:

```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat(metrics): add self-similarity metric

Implement self-similarity calculation for attention patterns
using correlation-based approach.

Closes #123
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fractal_attention_analysis --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::test_analyzer_initialization
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Include docstrings explaining what is being tested

Example:

```python
def test_fractal_dimension_calculation():
    """Test that fractal dimension is computed correctly for known patterns."""
    transforms = FractalTransforms()
    # Create test attention matrix
    attention = np.random.rand(10, 10)
    dimension = transforms.compute_fractal_dimension(attention)
    
    # Verify dimension is in expected range
    assert 0 < dimension < 3, "Fractal dimension should be between 0 and 3"
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Test with different model architectures
- Test GPU and CPU code paths

## 🎨 Code Style

We follow PEP 8 with some modifications:

### Formatting

- Line length: 100 characters
- Use Black for automatic formatting
- Use isort for import sorting

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/
```

### Linting

```bash
# Run flake8
flake8 src/ tests/

# Run mypy for type checking
mypy src/
```

### Documentation

- Use Google-style docstrings
- Include type hints for all function parameters and return values
- Document all public classes and methods

Example:

```python
def compute_fractal_dimension(
    self,
    attention_matrix: np.ndarray,
    scales: Optional[List[int]] = None
) -> float:
    """
    Compute fractal dimension using box-counting method.
    
    Args:
        attention_matrix: Input attention matrix [seq_len, seq_len]
        scales: List of box sizes for counting (defaults to powers of 2)
        
    Returns:
        Estimated fractal dimension
        
    Raises:
        ValueError: If attention_matrix is empty or invalid
        
    Example:
        >>> transforms = FractalTransforms()
        >>> dimension = transforms.compute_fractal_dimension(attention)
        >>> print(f"Dimension: {dimension:.4f}")
    """
```

## 📝 Submitting Changes

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Format
   black src/ tests/
   isort src/ tests/
   
   # Lint
   flake8 src/ tests/
   
   # Type check
   mypy src/
   
   # Test
   pytest --cov=fractal_attention_analysis
   ```

3. **Push to your fork**:
   ```bash
   git push origin your-branch-name
   ```

4. **Create Pull Request**:
   - Go to GitHub and create a pull request
   - Fill in the PR template
   - Link related issues
   - Request review from maintainers

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for changes
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

## 🐛 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Try latest version
3. Verify it's not a configuration issue

### Bug Report Template

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Initialize analyzer with '...'
2. Call method '...'
3. See error

**Expected behavior**
What you expected to happen

**Environment:**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Package version: [e.g., 0.1.0]
- GPU: [e.g., NVIDIA RTX 3090, None]

**Additional context**
Any other relevant information
```

## 💡 Feature Requests

We welcome feature requests! Please:

1. Check if feature already exists or is planned
2. Clearly describe the feature and use case
3. Provide examples of how it would be used
4. Explain why it would be valuable

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Explain the problem this feature would solve

**Proposed Solution**
How you envision the feature working

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Any other relevant information
```

## 📚 Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings in code
2. **User Guide**: README.md and examples
3. **API Reference**: Auto-generated from docstrings
4. **Tutorials**: Step-by-step guides

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# View documentation
# Open docs/_build/html/index.html in browser
```

## 🎯 Areas for Contribution

We especially welcome contributions in:

- **New Metrics**: Additional attention analysis metrics
- **Visualizations**: Enhanced plotting capabilities
- **Performance**: Optimization and GPU acceleration
- **Model Support**: Testing with new architectures
- **Documentation**: Tutorials, examples, and guides
- **Testing**: Expanding test coverage
- **Bug Fixes**: Fixing reported issues

## 📞 Questions?

- Open a [GitHub Discussion](https://github.com/ross-sec/fractal_attention_analysis/discussions)
- Email: devops.ross@gmail.com

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

---

**Happy Contributing! 🚀**

