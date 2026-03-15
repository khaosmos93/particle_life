"""Core scene object definitions for the initial Sun/Earth visualization."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CelestialBody:
    """Minimal shared representation for scene objects."""

    name: str
    radius: float
    position: tuple[float, float, float]


SUN = CelestialBody(name="sun", radius=1.5, position=(-6.0, 0.0, 0.0))
EARTH = CelestialBody(name="earth", radius=1.0, position=(2.0, 0.0, 0.0))
