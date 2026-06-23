# Geometric Solver and Visualizer for 2D Generic Belt Systems

This module provides the `Belt` and `Circle` classes to define, validate, and solve the geometry of a closed-loop belt-like system with no restrictions on the number of pulleys. It also supports custom routing topologies (front/back) via the `BeltFace` class, and dynamic length optimization via SciPy. Visualization is provided through Matplotlib. Axes and Figures are returned.

## Examples

### Standard 2-Pulley System
```python
c1 = Circle(40.0, (0, 0), name="Drive Motor")
c2 = Circle(40.0, (250, 0), name="Driven Wheel")

pulleys = [c1, c2]
routing = [BeltFace.FRONT, BeltFace.FRONT]

belt = Belt(circles=pulleys, topology=routing, allow_crossing=False)
print(f"Total Belt Length: {belt.total_length:.2f}")
belt.plot()
```

### Three-Pulley System with Mixed Routing and Optimization
```python
import numpy as np

c1 = Circle(52 * 2 / np.pi / 2, (0, 0), name="c1")
c2 = Circle(23 * 2 / np.pi / 2, (10, 50), name="c2")
c3 = Circle(20 * 2 / np.pi / 2, (0, 78.87), name="c3")

pulleys = [c1, c2, c3]
routing = [BeltFace.FRONT, BeltFace.BACK, BeltFace.FRONT]

belt = Belt(circles=pulleys, topology=routing, allow_crossing=True)
print(f"Original Length: {belt.total_length:.2f}")

# Slide the tensioner (c2) along the X-axis to reach exactly 240 units
belt.find_movable_circle_position(
    target_length=240, movable_circle_idx=1, slide_vector=np.array([1, 0])
)
print(f"New Length: {belt.total_length:.2f}")
belt.plot()
```

### 4-Pulley System (Default Topology)
```python
c1 = Circle(40.0, (0, 0), name="c1")
c2 = Circle(40.0, (250, 0), name="c2")
c3 = Circle(20.0, (240, -75), name="c3")
c4 = Circle(40.0, (100, -160), name="c4")

pulleys = [c1, c2, c3, c4]

# Topology defaults to all BeltFace.FRONT if not specified
belt = Belt(circles=pulleys, allow_crossing=False)
print(f"Total Belt Length: {belt.total_length:.2f}")
belt.plot()
```

---
**Author:** Vincent Wallsten  
**Date:** 2026-06-23