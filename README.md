# Helmholtz-Coils-Simulation
Simulating the magnetic field of Helmholtz Coils

The single_coil.py is based on 3 classes, the Wire, Coil and Space classes. 
It plots the vector field of the x and y components of the magnetic field
in the x, y plane. It also compares with a plot the theoretical magnetic
field on the x-axis - which is the axis of the coil - to the magnetic field
from the simulation. The mean squared error was then found for different n 
segment wires and a plot is made that shows the error being exponentially
dependent on n.

The helmholtz_coils.py is almost the same as the single coil with a modification 
in the space class to find the field caused by both coils and add them to get the
final magnetic field solution. This program plots the vector field in the x,y plane
and makes a contour plot of the strength of the magnetic field near the center of the
system.
