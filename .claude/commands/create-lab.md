# Create Lab

Generate lab exercise files for a chapter.

## Usage
```
/create-lab <module> <chapter> <topic>
```

## Output Structure
```
labs/module-X/chXX-name/
├── README.md
├── INSTRUCTIONS.md
├── package.xml (if ROS 2)
├── setup.py (if ROS 2)
└── src/
    └── main_code.py
```

## INSTRUCTIONS.md Template
```markdown
# Lab: {{Title}}

## Objectives
- [ ] Objective 1
- [ ] Objective 2

## Prerequisites
- Software X installed
- Chapter Y completed

## Setup
\`\`\`bash
# Setup commands
\`\`\`

## Part 1: [Task]
### Step 1.1
[Instructions]

### Expected Output
\`\`\`
[Sample output]
\`\`\`

## Verification
- [ ] Check 1
- [ ] Check 2

## Troubleshooting
| Issue | Solution |
|-------|----------|
| Error X | Fix Y |

## Submission
Upload to: [location]
```

## Time Estimates
- Basic lab: 45-60 min
- Intermediate: 60-90 min
- Advanced: 90-120 min
