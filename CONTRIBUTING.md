# Contributing

Thank you for your interest in contributing to the Physical AI & Robotics Textbook!

## Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes following the code standards below
4. Run tests: `npm run typecheck && npm run lint`
5. Commit with descriptive messages
6. Open a Pull Request

## Code Standards

### Frontend (TypeScript/React)
- Use TypeScript strict mode
- Follow ESLint configuration
- Use functional components with hooks
- Add JSDoc comments for public APIs

### Backend (Python/FastAPI)
- Follow PEP 8 style guide
- Use type hints throughout
- Format with Ruff
- Add docstrings for all functions

### Documentation (Markdown)
- Follow chapter template structure
- Include code examples with expected output
- Keep chapters within word count limits
- Add Mermaid diagrams for architecture

## Content Guidelines

### Chapters
- Regular: 1700-2500 words
- Capstone: 3000-3500 words
- Include learning objectives, code examples, lab references

### Labs
- Clear prerequisites
- Step-by-step instructions
- Verification commands
- Troubleshooting table

## Pull Request Process

1. Update relevant documentation
2. Add tests for new features
3. Ensure CI passes
4. Request review from maintainers

## Questions?

Open an issue for discussion before major changes.
