from pybeltsolver import Circle, BeltFace, Belt
import numpy as np

if __name__ == "__main__":
    print("--- Test 1: Standard 2-Pulley System ---")
    # Test 1: Two identical pulleys spaced 250 units apart
    c1 = Circle(40.0, (0, 0), name="Drive Motor")
    c2 = Circle(40.0, (250, 0), name="Driven Wheel")

    pulleys = [c1, c2]

    routing = [BeltFace.FRONT, BeltFace.FRONT]
    belt = Belt(circles=pulleys, topology=routing, allow_crossing=False)
    print(f"Total Belt Length: {belt.total_length:.2f}")
    belt.plot()

    # Test 2: Three-Pulley System with Mixed Routing and Optimization
    print("\n--- Test 2: Three-Pulley System with Mixed Routing and Optimization ---")
    c1 = Circle(52 * 2 / np.pi / 2, (0, 0), name="c1")
    c2 = Circle(23 * 2 / np.pi / 2, (10, 50), name="c2")
    c3 = Circle(20 * 2 / np.pi / 2, (0, 78.87), name="c3")

    pulleys = [c1, c2, c3]
    routing = [BeltFace.FRONT, BeltFace.BACK, BeltFace.FRONT]

    belt = Belt(circles=pulleys, topology=routing, allow_crossing=True)
    print(f"Original Length: {belt.total_length:.2f}")

    belt.find_movable_circle_position(
        target_length=240, movable_circle_idx=1, slide_vector=np.array([1, 0])
    )

    print(f"New Length: {belt.total_length:.2f}")

    # Plot the optimized result
    belt.plot()

    print("--- Test 3: 4-Pulley System ---")
    # Test 3:
    c1 = Circle(40.0, (0, 0), name="c1")
    c2 = Circle(40.0, (250, 0), name="c2")
    c3 = Circle(20, (240, -75), name="c3")
    c4 = Circle(40.0, (100, -160), name="c4")

    pulleys = [c1, c2, c3, c4]

    belt = Belt(circles=pulleys, allow_crossing=False)
    print(f"Total Belt Length: {belt.total_length:.2f}")
    belt.plot()
