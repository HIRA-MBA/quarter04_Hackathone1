# Generate Diagram

Generate Mermaid diagrams for the textbook.

## Usage
```
/generate-diagram <type> <description>
```

## Types

### Architecture
```mermaid
graph TB
    A[Component A] --> B[Component B]
    B --> C[Component C]
```

### Flowchart
```mermaid
flowchart TD
    Start --> Process
    Process --> Decision{Check}
    Decision -->|Yes| Action1
    Decision -->|No| Action2
```

### Sequence
```mermaid
sequenceDiagram
    participant A as Node A
    participant B as Node B
    A->>B: Request
    B-->>A: Response
```

### State
```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Running: start
    Running --> Idle: stop
```

### Class
```mermaid
classDiagram
    class Robot {
        +move()
        +sense()
    }
```

## Guidelines
- Keep diagrams simple
- Use clear labels
- Max 10-15 nodes
- Add title comments
