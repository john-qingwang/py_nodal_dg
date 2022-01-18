"""A library for boundary conditions."""
import enum


class BoundaryCondition(enum.Enum):
    """Defines the type of boundary condition."""
    IN = 1
    OUT = 2
    WALL = 3
    FAR = 4
    CYL = 5
    DIRICHLET = 6
    NEUMANN = 7
    SLIP = 8

