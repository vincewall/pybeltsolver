"""
Geometric solver and visualizer for 2D generic belt systems.

This module provides the `Belt` and `Circle` classes to define, validate,
and solve the geometry of a closed-loop belt-like system with no restrictions on the
number of pulleys. It also supports custom routing topologies (front/back), dynamic
length optimization via SciPy Visualization is provided through Matplotlib. Axes and
Figures are returned.

Examples:
    Test 1: Standard 2-Pulley System
    ```python
    c1 = Circle(40.0, (0, 0), name="Drive Motor")
    c2 = Circle(40.0, (250, 0), name="Driven Wheel")

    pulleys = [c1, c2]
    routing = [BeltFace.FRONT, BeltFace.FRONT]

    belt = Belt(circles=pulleys, topology=routing, allow_crossing=False)
    print(f"Total Belt Length: {belt.total_length:.2f}")
    belt.plot()
    ```

    Test 2: Three-Pulley System with Mixed Routing and Optimization
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

    Test 3: 4-Pulley System (Default Topology)
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

Author: Vincent Wallsten
Date: 2026-06-23
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import scipy.optimize as opt
from enum import Enum


class BeltSolverError(Exception):
    """Base class for all errors raised by the Belt kinematic solver."""

    pass


class TopologyMismatchError(BeltSolverError):
    """Raised when the number of faces does not match the number of pulleys."""

    pass


class OverlappingPulleyError(BeltSolverError):
    """
    Raised when the distance between two pulleys is less than their combined radii.
    """

    pass


class CrossingBeltError(BeltSolverError):
    """Raised when a crossing belt is detected and crossing is not allowed."""

    pass


class BeltFace(Enum):
    """
    Defines the face of the belt for routing purposes.
    """

    FRONT = 1
    BACK = 2


class Circle:
    """
    Represents the Belt Pulleys.
    """

    def __init__(self, r: float, coords: np.ndarray, name: str = None) -> None:
        """
        Initializes the Circle class.

        Args:
            r (float): Radius of the circle.
            coords (np.ndarray): Circle center coordinates as a 2D array [x, y].
            name (str, optional): Name of the circle. Defaults to None.
        """

        self.x = coords[0]
        self.y = coords[1]
        self.coords = np.array(coords)
        self.r = r
        self.name = name

    def get_matplotlib_circle(self) -> plt.Circle:
        """
        Returns a matplotlib Circle patch for visualization.

        Returns:
            matplotlib.patches.Circle: A matplotlib Circle patch for visualization.
        """
        return plt.Circle((self.x, self.y), self.r, fill=False, color="black")

    def dist(self, c: "Circle") -> float:
        """
        Calculates the distance between two circles (self, other)

        Args:
            c (Circle): The other circle to calculate the distance to.

        Returns:
            float: The distance between the centers of the two circles.
        """
        diff = self.coords - c.coords
        return np.linalg.norm(diff)

    def __str__(self) -> str:
        """
        String representation of the circle object.

        Returns:
            str: A string representation of the circle object.
        """
        return f"Circle '{self.name}' at ({self.x}, {self.y}) with radius {self.r}"


class Belt:
    """
    Complete representation of a generic belt. Includes geometry calculations,
    validation, optimization methods and visualization.
    """

    def __init__(
        self,
        circles: list[Circle],
        topology: list[BeltFace] = None,
        do_validate: bool = True,
        allow_crossing: bool = False,
    ) -> None:
        """
        Validates the user input, computes the geometry and checks for crossing belt
        configurations.

        Args:
            circles (list[Circle]): List of Circle objects representing the pulleys in
                the belt system. Must be ordered sequentially (preferably clockwise) in
                the direction of the belt routing.
            topology (list[BeltFace], optional): List of instructions for how the belt
                is to be routed. Defaults to None, which assumes the front (inside) of
                the belt touches all pulleys.
            do_validate (bool, optional): Runs geometric validation checks upon
                initialization. Defaults to True.
            allow_crossing (bool, optional): If False, throws an error when a crossing
                belt (Figure-8) is detected. Set to True to allow intentional crossings.
                Defaults to False.
        """

        self.circles = circles
        self.topology = topology if topology else [BeltFace.FRONT] * len(circles)
        self.do_validate = do_validate
        self.allow_crossing = allow_crossing
        self.lines = []
        self.arcs = []

        self.line_lengths = []
        self.arcs_lengths = []

        self.total_length = 0.0
        self.wrap_points = {c: {"entry": None, "exit": None} for c in circles}
        self.wrap_angles = {c: {"entry": None, "exit": None} for c in circles}
        self.wrap_lines = {c: {"entry": None, "exit": None} for c in circles}

        self.debug = False

        if self.do_validate:
            self._check_validity()

        self._calculate_geometry()

        self._check_crossing_lines()

    def _check_validity(self) -> None:
        """
        Validates the user input for the belt configuration.

        Raises:
            BeltSolverError: Raised for insufficient number of circles.
            TopologyMismatchError: Raised for topology list and circle list length
                mismatch.
            OverlappingPulleyError: Raised for overlapping pulleys.
        """
        # Basic input error checks
        if len(self.circles) < 2:
            raise BeltSolverError("At least two circles are required to form a belt.")

        if len(self.topology) != len(self.circles):
            raise TopologyMismatchError(
                f"Topology list ({len(self.topology)}) must match the number of "
                f"circles ({len(self.circles)})."
            )
        # Check for overlapping circles
        for i, c1 in enumerate(self.circles):
            for j in range(len(self.circles)):
                if i != j:
                    c2 = self.circles[j]
                    if c1.dist(c2) < (c1.r + c2.r):
                        raise OverlappingPulleyError(
                            f"Circles '{c1.name}' and '{c2.name}' are overlapping."
                        )

    def _check_crossing_lines(self) -> None:
        """
        Checks for crossing belts and raises an error if crossing is not allowed.

        Raises:
            CrossingBeltError: Raised when a crossing belt is detected and crossing is
            not allowed.
        """

        for c in self.circles:
            line_in = self.wrap_lines[c]["entry"]
            line_out = self.wrap_lines[c]["exit"]

            P_in = line_in[0]
            P_out = line_out[0]

            V_in = line_in[1] - line_in[0]
            V_out = line_out[1] - line_out[0]

            V = np.vstack((V_in, -V_out)).T
            P = P_out - P_in

            # Determinant will be close to zero for parallel lines
            det = np.linalg.det(V)

            if abs(det) > 1e-9:
                lambd_in, lambd_out = np.linalg.solve(V, P)

                if 0 < lambd_in < 1 and 0 < lambd_out < 1:
                    if not self.allow_crossing:
                        raise CrossingBeltError(
                            f"Fatal Geometry: Belt crosses itself at '{c.name}'."
                            f"If intentional, enable 'Allow Crossing'."
                        )
                    else:
                        if self.debug:
                            print(f"Notice: crossed belt detected at '{c.name}'.")
                            print(lambd_in, lambd_out)

    def _calculate_geometry(self) -> None:
        """
        Generates the geometry of the belt by computing the tangencies and arcs
        appropriate to the specified topology.
        """
        # Clear previous geometry (for solvers)
        self.lines.clear()
        self.arcs.clear()
        self.line_lengths.clear()
        self.arcs_lengths.clear()

        # Tangent lines
        for i in range(len(self.circles)):
            j = (i + 1) % len(self.circles)
            c1 = self.circles[i]
            c2 = self.circles[j]
            c1_routing = self.topology[i]
            c2_routing = self.topology[j]

            # Routing assumes clockwise topology
            if c1_routing == c2_routing:
                which = "R" if c1_routing == BeltFace.FRONT else "L"
                line = self._create_outer_tangent_lines(c1, c2, which=which)
            else:
                which = "RL" if c1_routing == BeltFace.FRONT else "LR"
                line = self._create_inner_tangent_lines(c1, c2, which=which)
            self.lines.append(line)

            # Add entry and exit points and lines to the circles
            self.wrap_points[c1]["exit"] = line[0]
            self.wrap_points[c2]["entry"] = line[1]

            self.wrap_lines[c1]["exit"] = line
            self.wrap_lines[c2]["entry"] = line

        for i, c in enumerate(self.circles):
            entry_point = self.wrap_points[c]["entry"]
            exit_point = self.wrap_points[c]["exit"]

            entry_line = self.wrap_lines[c]["entry"]
            exit_line = self.wrap_lines[c]["exit"]

            arc = self._create_arcs(c, entry_point, exit_point, entry_line, exit_line)
            self.arcs.append(arc)

        self._calculate_total_length()

    def _calculate_total_length(self) -> None:
        """
        Calculates the total belt length.
        """
        self.total_length = sum(self.line_lengths) + sum(self.arcs_lengths)

    def find_movable_circle_position(
        self, target_length: float, movable_circle_idx: int, slide_vector: np.ndarray
    ) -> bool:
        """
        Find the new position of a specified circle along the specified slide vector to
        achieve the target length.

        Args:
            target_length (float): Target length of the belt.
            movable_circle_idx (int): Index of the circle to move.
            slide_vector (np.ndarray): Vector along which to slide the circle.

        Returns:
            bool: True if a valid position is found, False otherwise.
        """
        movable_circle = self.circles[movable_circle_idx]
        original_coords = movable_circle.coords.copy()

        slide_vector = slide_vector / np.linalg.norm(slide_vector)

        def length_error(t):
            new_pos = original_coords + (t * slide_vector)
            movable_circle.coords = new_pos
            movable_circle.x, movable_circle.y = new_pos

            self._calculate_geometry()

            return self.total_length - target_length

        try:
            result = opt.root_scalar(length_error, x0=0.0, x1=5.0, method="secant")
            print(f"Position found! Shifted by {result.root:.2f} units.")
            if self.do_validate:
                self._check_crossing_lines()

            return True

        except ValueError as e:
            print(
                f"Target length is physically impossible on this slide vector."
                f"Error: {e}"
            )
            movable_circle.coords = original_coords
            movable_circle.x, movable_circle.y = original_coords
            self._calculate_geometry()

            return False

    def find_editable_circle_radius(
        self, target_length: float, editable_circle_idx: int
    ) -> bool:
        """
        Find the new radius of a specified circle to achieve the target length.

        Args:
            target_length (float): Target length of the belt.
            editable_circle_idx (int): Index of the circle to adjust.

        Returns:
            bool: True if a valid radius is found, False otherwise.
        """

        editable_circle = self.circles[editable_circle_idx]
        original_radius = editable_circle.r

        def length_error(r):
            editable_circle.r = r
            self._calculate_geometry()
            return self.total_length - target_length

        try:
            result = opt.root_scalar(
                length_error,
                x0=original_radius,
                x1=original_radius * 1.1,
                method="secant",
            )
            print(f"Radius found! Adjusted by {result.root:.2f} units.")
            if self.do_validate:
                self._check_crossing_lines()

            return True

        except ValueError as e:
            print(
                f"Target length is physically impossible with this radius adjustment."
                f"Error: {e}"
            )
            editable_circle.r = original_radius
            self._calculate_geometry()
            return False

    def plot(self, show: bool = True, plot_circles: bool = True) -> tuple[Figure, Axes]:
        """
        Plots the complete Belt in matplotlib.

        Args:
            show (bool, optional): Show the plot before returning. Defaults to True.
            plot_circles (bool, optional): Whether to plot circles. Defaults to True.

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and Axes objects of the plot.
        """
        fig, ax = plt.subplots()

        # Plot circles
        if plot_circles:
            for c in self.circles:
                ax.add_patch(c.get_matplotlib_circle())

        # Plot belt lines and arcs
        segs = self.lines + self.arcs
        line_segments = LineCollection(segs, linestyle="solid", color="red")
        ax.add_collection(line_segments)

        ax.autoscale()
        ax.set_aspect("equal")
        plt.show() if show else None

        return fig, ax

    def _create_outer_tangent_lines(
        self, c1: Circle, c2: Circle, which="L"
    ) -> np.ndarray:
        """
        Computes the outer tangents between two circles on the specified side.

        Args:
            c1 (Circle): First circle.
            c2 (Circle): Second circle.
            which (str, optional): Which side the tangent should be draw on. Side is
                defined by the left and right side of the line between the circle
                centers. Defaults to "L".

        Returns:
            np.ndarray: Array containing the coordinates of the tangent line.
        """
        hypotenuse = c1.dist(c2)
        short = c1.r - c2.r

        ratio = np.clip(short / hypotenuse, -1, 1)

        if which == "R":
            phi = np.atan2(c2.y - c1.y, c2.x - c1.x) + np.arccos(ratio)
        elif which == "L":
            phi = np.atan2(c2.y - c1.y, c2.x - c1.x) - np.arccos(ratio)
        else:
            raise ValueError(
                f"Invalid input: 'which' must be 'L' or 'R', got '{which}'."
            )

        t1x = c1.x + c1.r * np.cos(phi)
        t1y = c1.y + c1.r * np.sin(phi)

        t2x = c2.x + c2.r * np.cos(phi)
        t2y = c2.y + c2.r * np.sin(phi)

        line = np.array([[t1x, t1y], [t2x, t2y]])

        self.line_lengths.append(np.linalg.norm(line[1] - line[0]))

        return line

    def _create_inner_tangent_lines(
        self, c1: Circle, c2: Circle, which="RL"
    ) -> np.ndarray:
        """
        Creates the inner (or crossing) tangent line between two circles.

        Args:
            c1 (Circle): First circle.
            c2 (Circle): Second circle.
            which (str, optional): Which side the tangent should be drawn on. See
            outer_tangent_lines for more details on side definition. Defaults to "RL".

        Returns:
            np.ndarray: Array containing the coordinates of the tangent line.
        """

        hypotenuse = c1.dist(c2)
        short = c1.r + c2.r

        ratio = np.clip(short / hypotenuse, -1, 1)
        if which == "RL":
            phi = np.atan2(c2.y - c1.y, c2.x - c1.x) - np.arcsin(ratio) + np.pi / 2
        elif which == "LR":
            phi = np.atan2(c2.y - c1.y, c2.x - c1.x) + np.arcsin(ratio) - np.pi / 2
        else:
            exit(f"Which: {which} is not a valid input.")

        t1x = c1.x + c1.r * np.cos(phi)
        t1y = c1.y + c1.r * np.sin(phi)

        t2x = c2.x + c2.r * np.cos(phi + np.pi)
        t2y = c2.y + c2.r * np.sin(phi + np.pi)

        line = np.array([[t1x, t1y], [t2x, t2y]])

        self.line_lengths.append(np.linalg.norm(line[1] - line[0]))

        return line

    def _create_arcs(
        self,
        c: Circle,
        entry_point: np.ndarray,
        exit_point: np.ndarray,
        entry_line: np.ndarray,
        exit_line: np.ndarray,
    ) -> np.ndarray:
        """
        This function uses a ray approach to determine which arc should be drawn.

        Args:
            c (Circle): Circle for which the arc is being calculated.
            entry_point (np.ndarray): Point where the arc starts.
            exit_point (np.ndarray): Point where the arc ends.
            entry_line (np.ndarray): Line from which the arc starts.
            exit_line (np.ndarray): Line from which the arc ends.

        Returns:
            np.ndarray: Array containing the coordinates of the arc.
        """

        # Internal functions
        def calculate_arc_length(phi_1, phi_2, c: Circle):
            sweep_angle = abs(phi_2 - phi_1)
            arc_length = c.r * sweep_angle
            return arc_length

        def get_arc_points(p1, p2, n, c: Circle):
            theta = np.linspace(p1, p2, n)
            return np.column_stack(
                [c.r * np.cos(theta) + c.x, c.r * np.sin(theta) + c.y]
            )

        def swap_angle(phi_1, phi_2):
            if phi_2 > phi_1:
                phi_2 -= 2 * np.pi
            else:
                phi_2 += 2 * np.pi
            return phi_1, phi_2

        arcs = np.array([[c.x, c.y], [c.x, c.y]])

        phi_1 = np.atan2(*(entry_point - c.coords)[::-1])
        phi_2 = np.atan2(*(exit_point - c.coords)[::-1])

        self.wrap_angles[c]["entry"] = phi_1
        self.wrap_angles[c]["exit"] = phi_2

        sweep_angle = abs(phi_2 - phi_1)
        arc_length = calculate_arc_length(phi_1, phi_2, c)

        P = exit_point - entry_point

        # Points on circle
        p1 = entry_point
        p2 = exit_point

        # Intersection lines
        line1_coords = entry_line
        line2_coords = exit_line

        # Determine which point sits opposite on the line
        l1 = (
            line1_coords[0]
            if np.linalg.norm(line1_coords[0] - p1)
            > np.linalg.norm(line1_coords[1] - p1)
            else line1_coords[1]
        )
        l2 = (
            line2_coords[0]
            if np.linalg.norm(line2_coords[0] - p2)
            > np.linalg.norm(line2_coords[1] - p2)
            else line2_coords[1]
        )

        # Compute normals
        v1 = l1 - p1
        v2 = l2 - p2
        n1 = v1 / np.linalg.norm(v1)
        n2 = v2 / np.linalg.norm(v2)
        N = np.vstack([n1, -n2]).T

        det = np.linalg.det(N)
        if abs(det) > 1e-6:
            lambd_1, lambd_2 = np.linalg.solve(N, P)
            # If both lambdas are positive, the intersect is enclosing the small arc,
            # else the intersect is enclosing the large arc
            if lambd_1 > 0 and lambd_2 > 0:
                if sweep_angle < np.pi:
                    phi_1, phi_2 = swap_angle(phi_1, phi_2)
                    arc_length = calculate_arc_length(phi_1, phi_2, c)
            else:
                if sweep_angle > np.pi:
                    phi_1, phi_2 = swap_angle(phi_1, phi_2)
                    arc_length = calculate_arc_length(phi_1, phi_2, c)
        else:
            # Do a test with the mid angle to determine which arc is correct
            mid_phi = (phi_1 + phi_2) / 2.0
            test_vec = np.array([np.cos(mid_phi), np.sin(mid_phi)])
            outgoing_vec = line2_coords[1] - line2_coords[0]

            if np.dot(test_vec, outgoing_vec) > 0:
                phi_1, phi_2 = swap_angle(phi_1, phi_2)

            arc_length = calculate_arc_length(phi_1, phi_2, c)

        # Finally, compute arc points
        n = max(5, int(arc_length * 180 / np.pi))
        arcs = get_arc_points(phi_1, phi_2, n, c)
        self.arcs_lengths.append(arc_length)

        return arcs