# Contributing to 3D Reconstruction with Bundle Adjustment

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs or request features
- Check existing issues before creating new ones
- Provide clear, detailed descriptions with steps to reproduce
- Include system information (OS, Python version, dependency versions)

### Suggesting Features
- Open a GitHub Issue with the "enhancement" label
- Describe the feature and its use case
- Explain why it would be valuable to users
- Consider implementation complexity and maintenance burden

### Submitting Code Changes

#### 1. Fork and Clone
```bash
git clone https://github.com/yourusername/3d-reconstruction.git
cd 3d-reconstruction
```

#### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

#### 3. Set Up Development Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

#### 4. Make Changes
- Follow the coding standards below
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

#### 5. Test Your Changes
```bash
# Run existing tests
python -m pytest tests/

# Test web interface
cd web && python app.py

# Test CLI
python main.py --help
```

#### 6. Submit Pull Request
- Push your branch to your fork
- Create a pull request against the main branch
- Provide a clear description of changes
- Reference any related issues

## 📝 Coding Standards

### Python Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write clear, descriptive variable and function names
- Add docstrings for all public functions and classes

### Code Organization
- Keep functions focused and single-purpose
- Organize imports: standard library, third-party, local imports
- Use meaningful module and package names
- Maintain consistent file structure

### Documentation
- Update README.md for user-facing changes
- Add inline comments for complex algorithms
- Document API changes
- Include examples for new features

### Testing
- Write unit tests for new functions
- Include integration tests for major features
- Test edge cases and error conditions
- Maintain test coverage above 80%

## 🔍 Code Review Process

### What We Look For
- **Functionality**: Does the code work as intended?
- **Code Quality**: Is it readable, maintainable, and well-structured?
- **Testing**: Are there adequate tests?
- **Documentation**: Is it properly documented?
- **Security**: Are there any security concerns?
- **Performance**: Does it impact system performance?

### Review Timeline
- Initial review within 3-5 business days
- Follow-up reviews within 1-2 business days
- Merging after approval and passing CI checks

## 🏗️ Development Guidelines

### Setting Up Your Environment
1. **Python Version**: Use Python 3.8 or higher
2. **Virtual Environment**: Always use a virtual environment
3. **Dependencies**: Install from requirements.txt
4. **IDE Setup**: Configure your IDE for Python development

### Project Structure
```
src/
├── core/              # Core algorithms
├── visualization/     # Visualization components
└── utils/            # Utility functions

web/
├── app.py            # Flask application
├── templates/        # HTML templates
└── static/          # CSS, JS, images

tests/
├── test_core/        # Core algorithm tests
├── test_web/         # Web interface tests
└── fixtures/         # Test data
```

### Adding New Features

#### New Algorithm Implementation
1. Add core logic to appropriate module in `src/core/`
2. Create comprehensive unit tests
3. Add visualization if applicable
4. Update CLI and web interfaces
5. Document in README.md

#### Web Interface Changes
1. Update Flask routes in `web/app.py`
2. Modify HTML templates if needed
3. Test in multiple browsers
4. Ensure mobile responsiveness
5. Add error handling

#### CLI Enhancements
1. Update argument parsing in `main.py`
2. Maintain backward compatibility
3. Add help text and examples
4. Test with various input formats

## 🧪 Testing Guidelines

### Test Types
- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test resource usage

### Test Data
- Use small, synthetic test images
- Include edge cases (empty files, large files, etc.)
- Test with various image formats
- Create reproducible test scenarios

### Running Tests
```bash
# All tests
python -m pytest

# Specific test file
python -m pytest tests/test_core/test_features.py

# With coverage
python -m pytest --cov=src tests/
```

## 📋 Checklist Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New functionality is tested
- [ ] Documentation is updated
- [ ] No sensitive information in commits
- [ ] Git history is clean (consider squashing commits)
- [ ] Pull request description is clear

## 🚫 What Not to Contribute

### Avoid These Changes
- Breaking changes without discussion
- Code that introduces security vulnerabilities
- Large refactoring without prior discussion
- Features that significantly increase complexity
- Changes that reduce performance without justification

### Restricted Areas
- Don't include personal information or credentials
- Don't add unnecessary dependencies
- Don't remove existing functionality without deprecation
- Don't include large binary files or sample images

## 🎯 Areas Where Help Is Needed

### High Priority
- Performance optimization
- Additional camera models
- GPU acceleration support
- Better error handling
- Cross-platform testing

### Medium Priority
- UI/UX improvements
- Additional export formats
- Documentation improvements
- Example tutorials
- Docker containerization

### Low Priority
- Code refactoring
- Additional visualizations
- Experimental features
- Research integrations

## 💬 Communication

### Getting Help
- GitHub Issues for bugs and feature requests
- GitHub Discussions for general questions
- Code comments for implementation questions

### Staying Updated
- Watch the repository for notifications
- Check the changelog for updates
- Follow project milestones

## 📜 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment
- Respect different viewpoints and experiences

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or inflammatory comments
- Personal attacks
- Publishing private information

### Enforcement
- Issues will be reviewed by maintainers
- Violations may result in temporary or permanent bans
- Contact maintainers for serious issues

---

Thank you for contributing to the 3D Reconstruction project! 🚀