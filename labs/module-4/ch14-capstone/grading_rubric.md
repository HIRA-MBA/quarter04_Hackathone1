# Capstone Grading Rubric

## Physical AI & Humanoid Robotics - Final Capstone

**Total Points: 100**

---

## 1. System Integration (25 points)

| Criteria | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (2) | Incomplete (0-1) |
|----------|--------------|----------|------------------|----------------|------------------|
| **Subsystem Communication** | All systems communicate flawlessly via ROS 2 | Minor latency issues | Occasional message drops | Frequent communication failures | Systems don't communicate |
| **State Management** | Blackboard/state properly shared | Minor sync issues | Some state inconsistencies | Frequent state conflicts | No shared state |
| **Initialization** | Clean startup, all systems ready | Minor init delays | Some systems need retry | Requires manual intervention | Fails to initialize |
| **Shutdown** | Graceful shutdown, resources freed | Minor cleanup issues | Some resources leaked | Requires force kill | Crashes on shutdown |
| **Architecture** | Clean separation of concerns | Mostly modular | Some tight coupling | Heavy coupling | Monolithic mess |

**Scoring Guide:**
- 23-25: All subsystems work together seamlessly
- 20-22: Minor integration issues, overall functional
- 15-19: Core integration works, edge cases fail
- 10-14: Basic communication, many integration bugs
- 0-9: Systems largely isolated or non-functional

---

## 2. Task Completion (25 points)

| Criteria | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (2) | Incomplete (0-1) |
|----------|--------------|----------|------------------|----------------|------------------|
| **Perception** | >95% detection rate, <5cm accuracy | >85% detection, <10cm | >70% detection, <20cm | >50% detection | No working detection |
| **Locomotion** | Stable walking, reaches all targets | Minor balance issues | Occasional stumbles | Frequent balance loss | Cannot walk |
| **Manipulation** | >90% grasp success, precise placement | >75% grasp success | >60% grasp success | >40% grasp success | Cannot grasp |
| **Language** | All commands parsed correctly | >90% parse accuracy | >75% parse accuracy | >50% parse accuracy | Parsing fails |
| **Demo Completion** | Full scenario completed smoothly | Completed with minor issues | Partially completed | Minimal completion | Demo fails |

**Scoring Guide:**
- 23-25: Demo runs flawlessly, all objectives achieved
- 20-22: Demo completes with recoverable errors
- 15-19: Core tasks work, some objectives missed
- 10-14: Basic functionality, many failures
- 0-9: Demo largely non-functional

---

## 3. Safety & Robustness (20 points)

| Criteria | Excellent (4) | Good (3) | Satisfactory (2) | Needs Work (1) | Incomplete (0) |
|----------|--------------|----------|------------------|----------------|------------------|
| **Emergency Stop** | <50ms response, tested | <100ms response | <500ms response | >1s response | No e-stop |
| **Fault Detection** | Detects all faults, clear alerts | Detects major faults | Detects some faults | Misses many faults | No detection |
| **Error Recovery** | Automatic recovery, continues task | Manual recovery possible | Partial recovery | Requires restart | No recovery |
| **Edge Cases** | Handles all edge cases | Handles most edge cases | Handles common cases | Fails on edge cases | No edge case handling |
| **Graceful Degradation** | Continues with reduced capability | Continues in safe mode | Stops safely | Stops unsafely | Crashes |

**Scoring Guide:**
- 18-20: Handles errors gracefully, robust in all scenarios
- 15-17: Good error handling, minor gaps
- 11-14: Basic safety, some vulnerabilities
- 6-10: Minimal safety features
- 0-5: Unsafe operation

---

## 4. Code Quality (15 points)

| Criteria | Excellent (3) | Good (2) | Needs Work (1) | Incomplete (0) |
|----------|--------------|----------|----------------|------------------|
| **Organization** | Clear module structure, good separation | Mostly organized | Some disorganization | Chaotic structure |
| **Documentation** | Comprehensive docstrings, comments | Good documentation | Minimal comments | No documentation |
| **Naming** | Clear, consistent naming conventions | Mostly consistent | Inconsistent naming | Poor naming |
| **Error Handling** | Proper try/except, informative messages | Basic error handling | Limited handling | No error handling |
| **Code Style** | Follows PEP 8 / style guide | Minor style issues | Style inconsistencies | No style consistency |

**Scoring Guide:**
- 14-15: Professional quality code, well documented
- 11-13: Good code quality, minor issues
- 7-10: Functional but messy code
- 4-6: Difficult to read/maintain
- 0-3: Unacceptable code quality

---

## 5. Documentation & Demo (15 points)

| Criteria | Excellent (3) | Good (2) | Needs Work (1) | Incomplete (0) |
|----------|--------------|----------|----------------|------------------|
| **README** | Complete setup, all deps listed | Good instructions | Basic instructions | Missing or unclear |
| **Architecture Docs** | Clear diagrams, design rationale | Good diagrams | Basic overview | No architecture docs |
| **Video Demo** | Professional, shows all features | Good demonstration | Basic walkthrough | Poor quality/missing |
| **Written Report** | Thorough analysis, insights | Good coverage | Minimal analysis | Missing or incomplete |
| **Presentation** | Clear explanation, Q&A ready | Good presentation | Basic explanation | Unclear presentation |

**Scoring Guide:**
- 14-15: Excellent documentation and compelling demo
- 11-13: Good documentation, clear demo
- 7-10: Adequate documentation, functional demo
- 4-6: Minimal documentation, basic demo
- 0-3: Missing or poor documentation

---

## Bonus Points (Up to 10)

| Bonus | Points | Description |
|-------|--------|-------------|
| **Multi-robot coordination** | +5 | Two or more robots working together |
| **Real hardware deployment** | +5 | Successfully deployed to physical robot |
| **Novel feature** | +3 | Creative addition beyond requirements |
| **Performance optimization** | +2 | Demonstrable performance improvements |
| **Comprehensive testing** | +2 | Unit tests, integration tests included |

---

## Deductions

| Deduction | Points | Reason |
|-----------|--------|--------|
| **Late submission** | -10/day | Up to 3 days, then 0 |
| **Missing video** | -10 | No video demonstration |
| **Plagiarism** | -100 | Automatic failure |
| **Unsafe operation** | -20 | Robot could harm itself/environment |
| **Missing requirements** | -5 each | Per missing core requirement |

---

## Grade Mapping

| Points | Grade | Description |
|--------|-------|-------------|
| 90-100+ | A | Excellent - exceeds expectations |
| 85-89 | A- | Very good with minor issues |
| 80-84 | B+ | Good work, meets expectations |
| 75-79 | B | Solid work with some gaps |
| 70-74 | B- | Acceptable, noticeable issues |
| 65-69 | C+ | Below expectations |
| 60-64 | C | Minimum passing |
| 55-59 | C- | Marginally passing |
| 50-54 | D | Poor performance |
| 0-49 | F | Failing |

---

## Submission Checklist

Before submitting, verify:

- [ ] All source code in repository
- [ ] README with setup instructions
- [ ] `requirements.txt` or `package.xml` with dependencies
- [ ] Architecture diagram included
- [ ] Video demonstration uploaded (5-10 min)
- [ ] Written report (4-6 pages)
- [ ] Code runs without errors on clean setup
- [ ] Demo scenario clearly identified

---

## Evaluation Process

1. **Code Review** (Day 1)
   - Clone repository
   - Check code structure and quality
   - Review documentation

2. **Setup & Run** (Day 1-2)
   - Follow README instructions
   - Note any setup issues
   - Run system in simulation

3. **Demo Evaluation** (Day 2)
   - Watch video demonstration
   - Note task completion
   - Evaluate robustness

4. **Interview** (Day 3, if needed)
   - Clarify design decisions
   - Discuss challenges
   - Answer technical questions

5. **Final Scoring** (Day 3)
   - Apply rubric
   - Calculate total
   - Provide feedback

---

## Feedback Template

```
## Capstone Evaluation: [Student Name]

### Scores
- System Integration: __/25
- Task Completion: __/25
- Safety & Robustness: __/20
- Code Quality: __/15
- Documentation & Demo: __/15
- Bonus: +__
- Deductions: -__

**Total: __/100**
**Grade: __**

### Strengths
-

### Areas for Improvement
-

### Comments
-
```
