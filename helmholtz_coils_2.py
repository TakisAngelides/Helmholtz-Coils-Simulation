import numpy as np
from scipy.constants import mu_0, pi
import pandas as pd
import matplotlib.pyplot as plt

class Wire:

    current = 1/mu_0

    def __init__(self, x, y, z, vx, vy, vz):
        """
        :param x: (float) x position in space for the segment wire
        :param y: (float) y position in space for the segment wire
        :param z: (float) z position in space for the segment wire
        :param vx: (float) x component of the direction of the segment wire
        :param vy: (float) y component of the direction of the segment wire
        :param vz: (float) z component of the direction of the segment wire
        """
        # Position vector
        self.x = x
        self.y = y
        self.z = z
        # dl vector
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.dl = [vx, vy, vz]

    def get_space_position(self):
        """
        :return: (nparray) Returns the position in space of a segment wire in the form (x, y, z)
        """
        return np.array([self.x, self.y, self.z])

def subtract(list1, list2):
    """
    Subtracts 2 vectors element-wise
    :param list1: (list) vector 1
    :param list2: (list) vector 2
    :return: (list) vector 2 - vector 1
    """
    if len(list1) != len(list2):
        print('You used the function subtract with 2 vectors of different length')
        exit()
    else:
        return [list2[i] - list1[i] for i in range(len(list1))]

def get_mid_point(x1, y1, x2, y2):
    """
    :param x1: (float) x1 point from the coil's perimeter
    :param y1: (float) x1 point from the coil's perimeter
    :param x2: (float) x1 point from the coil's perimeter
    :param y2: (float) x1 point from the coil's perimeter
    :return: (float) Mid point between (x1, y1) and (x2, y2) which
    is the position for the dl vector going from (x1, y1) to (x2, y2)
    """
    return [(x2 + x1)/2, (y2 + y1)/2]

class Coil:

    n = 30 # Number of segment wires that form the coil
    radius = 1 # Radius of the coil

    def __init__(self, px, py, pz):
        """
        :param px: (float) x component in space for the center of the coil
        :param py: (float) y component in space for the center of the coil
        :param pz: (float) z component in space for the center of the coil
        """
        # Position of the coil's center in space - forms origin for coils frame
        self.px = px
        self.py = py
        self.pz = pz
        # In this list we store Wire objects that form the coil, each Wire object carries with it position and dl vectors
        self.coil_list = []

        # First list contains y,z positions for all segments that
        # form the coil and second list is their associated dl vectors
        position_in_x_plane_list, wire_vector_list = Coil.create_coil(self)

        # For every Wire object we want to create, we need to give it an appropriate position and dl vector
        # and append it to the coil_list.
        for i in range(len(position_in_x_plane_list)):
            self.coil_list.append(Wire(x = px, y = position_in_x_plane_list[i][0], z = position_in_x_plane_list[i][1],
                                       vx = 0.0, vy = wire_vector_list[i][0], vz = wire_vector_list[i][1]))

    def create_coil(self):
        """
        :return: (2-tuple) First list contains y,z positions for all segments that
        form the coil and second list is their associated dl vectors
        """
        radius = Coil.radius # Radius of coil
        n = Coil.n # Number of segments that form the whole coil
        p = np.linspace(0, 2 * pi, num=n + 1) # Points from 0 to 2pi
        origin_y = self.py # y component of the center of the coil
        origin_z = self.pz # z component of the center of the coil
        # Set of points that form a cirle, we use these to create dl vectors and mid-points for dl positions in space
        yp = radius * np.cos(p) + origin_y
        zp = radius * np.sin(p) + origin_z

        # Zip the y and z points into tuples that form (y, z) points - we use y, z because we want the x-axis to be the
        # axis of the coil
        pos_list_in_x_plane = [[yp[i], zp[i]] for i in range(len(yp))]

        # In this list we create and store the dl vectors
        vec_list = [subtract(pos_list_in_x_plane[i], pos_list_in_x_plane[i + 1]) for i in
                    range(len(pos_list_in_x_plane) - 1)]

        # In this list we create and store the mid-points that are the positions for the dl vectors in space
        mid_pt_list_in_x_plane = [get_mid_point(pos_list_in_x_plane[i][0], pos_list_in_x_plane[i][1],
                                                pos_list_in_x_plane[i+1][0], pos_list_in_x_plane[i+1][1])
                                                for i in range(len(pos_list_in_x_plane)-1)]

        return  mid_pt_list_in_x_plane, vec_list

def get_magnitude(list_vector):
    """
    Calculates the magnitude of any vector
    :param list_vector: (list) A vector
    :return: (float) The magnitude of the vector
    """
    vec = np.array(list_vector)
    return np.sqrt(vec.dot(vec))

def add_vectors(list_1, list_2):
    """
    Adds 2 vectors element-wise
    :param list_1: (list) vector 1
    :param list_2: (list) vector 2
    :return: (list) vector 1 + vector 2
    """
    return list(np.array(list_1) + np.array(list_2))

class Space:

    def __init__(self, length, diameter):
        """
        :param length: (float) Length of cylinder
        :param diameter: (float) Diameter of cylinder
        """
        self.length = length
        self.diameter = diameter
        # Create the grid that forms the space, for each of these points we will find the B field
        phi = np.linspace(0, 2 * pi, num = 6)  # Points from 0 to 2pi
        r_vec = np.linspace(0, diameter/2, num = 6) # Points from 0 to radius of cylinder
        self.x_list = np.linspace(-length/2, length/2, num = 36) # Points along the cylinder's length
        self.y_list = []
        self.z_list = []

        for r_mag in r_vec:
            for angle in phi:
                self.y_list.append(r_mag * np.cos(angle))
                self.z_list.append(r_mag * np.sin(angle))

    def get_data_at_point(self, x, y, z, coil_list):
        """
        :param x: (float) x component of point in space we want to get B field for
        :param y: (float) y component of point in space we want to get B field for
        :param z: (float) z component of point in space we want to get B field for
        :param coil_list: (Coil list) Holds Wire objects that form the coil
        :return: (7-tuple) x, y, z, Bx, By, Bz, |B| lists for all points in space
        """
        B_vec = [0.0, 0.0, 0.0]  # Initialize a B vector for the specific point in space
        wires = coil_list  # This list represents the whole coil
        point_pos = [x, y, z]  # Position in space we want to get data for

        # Go through every segment in the coil to calculate its contribution to that point in space
        for wire in wires:
            wire_pos = wire.get_space_position()  # Position vector of segment dl
            dl = wire.dl  # dl vector
            # If the point in space happens to have a segment wire then return a 0 B field
            if wire_pos[0] == x and wire_pos[1] == y and wire_pos[2] == z:
                return point_pos, [0.0, 0.0, 0.0], 0.0
            else:
                r = subtract(wire_pos, point_pos) # r vector in Biot-Savart law
                r_mag = get_magnitude(r) # Magnitude of r vector
                db = ((mu_0 * Wire.current) / (4 * pi * r_mag ** 3)) * np.cross(dl, r)  # Biot-Savart Law
                B_vec = B_vec + db  # Add contribution from current segment and move for loop to next wire segment

        B_mag = get_magnitude(B_vec)  # Get B field magnitude from final B vector
        # Components of B vector
        Bx = B_vec[0]
        By = B_vec[1]
        Bz = B_vec[2]
        return x, y, z, Bx, By, Bz, B_mag

    def get_data_for_all_points(self, coil_object_1, coil_object_2):
        """
        :param coil_object_1: (Coil) First coil
        :param coil_object_2: (Coil) Second coil
        :return: (7-tuple) x, y, z, Bx, By, Bz, |B|
        """
        # Lists to store value of position and B field component for all points in space
        px_list = []
        py_list = []
        pz_list = []
        # B field due to coil 1
        Bx_1_list = []
        By_1_list = []
        Bz_1_list = []
        B_1_mag_list = []
        # B field due to coil 2
        Bx_2_list = []
        By_2_list = []
        Bz_2_list = []
        B_2_mag_list = []
        # Final B field which is the addition of field 1 and 2
        Bx_total = []
        By_total = []
        Bz_total = []
        B_mag_total = []
        # Lists that represent the 2 coils
        coil_list_1 = coil_object_1.coil_list
        coil_list_2 = coil_object_2.coil_list

        # Traverse all space and evaluate the B field
        for x in self.x_list:
            for y in self.y_list:
                for z in self.z_list:
                    # Get data for each of the 2 coils, the coils share the same space hence px, py, pz are the same
                    px_1, py_1, pz_1, Bx_1, By_1, Bz_1, B_mag_1 = self.get_data_at_point(x, y, z, coil_list_1)
                    px_2, py_2, pz_2, Bx_2, By_2, Bz_2, B_mag_2 = self.get_data_at_point(x, y, z, coil_list_2)

                    px_list.append(px_1)
                    py_list.append(py_1)
                    pz_list.append(pz_1)

                    Bx_1_list.append(Bx_1)
                    By_1_list.append(By_1)
                    Bz_1_list.append(Bz_1)
                    B_1_mag_list.append(B_mag_1)

                    Bx_2_list.append(Bx_2)
                    By_2_list.append(By_2)
                    Bz_2_list.append(Bz_2)
                    B_2_mag_list.append(B_mag_2)

                    Bx_total = add_vectors(Bx_1_list, Bx_2_list)
                    By_total = add_vectors(By_1_list, By_2_list)
                    Bz_total = add_vectors(Bz_1_list, Bz_2_list)
                    B_mag_total = add_vectors(B_1_mag_list, B_2_mag_list)

        return px_list, py_list, pz_list, Bx_total, By_total, Bz_total, B_mag_total

# Create the cylindrical space with length and diameter specified here
space = Space(length = 0.1, diameter = 0.1)

# Create the coils with specified centers
coil_1 = Coil(px = -0.5, py = 0, pz = 0)
coil_2 = Coil(px = 0.5, py = 0, pz = 0)

# Generate data for all points in space
px_data, py_data, pz_data, Bx_data, By_data, Bz_data, B_mag_data = space.get_data_for_all_points(coil_1, coil_2)

# Put all data into a pandas dataframe
px_data_df = pd.DataFrame(px_data, columns = ['x'])
py_data_df = pd.DataFrame(py_data, columns = ['y'])
pz_data_df = pd.DataFrame(pz_data, columns = ['z'])
Bx_data_df = pd.DataFrame(Bx_data, columns = ['Bx'])
By_data_df = pd.DataFrame(By_data, columns = ['By'])
Bz_data_df = pd.DataFrame(Bz_data, columns = ['Bz'])
B_mag_data_df = pd.DataFrame(B_mag_data, columns = ['|B|'])
df = pd.concat([px_data_df, py_data_df, pz_data_df, Bx_data_df, By_data_df, Bz_data_df, B_mag_data_df], axis = 1)

def plot_Bx_By_field():
    # Quiver generates a vector field plot using the first 2 lists as the axis of the plot and the other 2 lists as
    # the vector components. The space object must have the same x and y num when initialized to make this plot.
    plt.quiver(px_data, py_data, Bx_data, By_data, scale = 100)
    plt.xlabel('x')
    plt.ylabel('y')
    # Center depends on px, py, pz when initializing the Coil object
    plt.title(f'Vector field for Bx, By\n(Helmholtz coils with radius 1, center (+/-0.5, 0, 0))')
    plt.savefig('helmholtz_field_2.pdf', bbox = 'tight')
    plt.show()

def plot_field_strength():
    # Plot the contour of the magnetic field strength in the x-y plane
    cp = plt.tricontourf(np.array(py_data), np.array(pz_data), np.array(B_mag_data))
    plt.colorbar(cp)
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('Contour plot for the magnetic field strength in the y-z plane\n(Helmholtz coils with radius 1, center (+/-0.5, 0, 0))')
    plt.savefig('contour_strength_2.pdf', bbox = 'tight')
    plt.show()

def get_percentage_difference(mag_at_0, mag_at_point):
    """
    :param mag_at_0: (float) Theoretical magnitude at (0, 0, 0)
    :param mag_at_point: (float) Magnitude of the magnetic field at a point in space
    :return: (float) The percentage difference between the 2 passed arguments
    """
    return (abs(mag_at_0-mag_at_point)/abs(mag_at_0))*100

def get_theoretical_center_field():
    """
    :return: (float) Theoretical values of the magnetic field strength at the origin of the system (0, 0, 0)
    """
    return ((4/5)**(3/2))*((mu_0*Wire.current)/Coil.radius)

def get_per_dif_list():
    """
    :return: (list) Percentage difference for all points in space between their field magnitude and magnitude at origin
    """
    field_strength = np.array(B_mag_data)
    center_field = get_theoretical_center_field()
    per_dif_array = (abs(field_strength-center_field)/abs(center_field))*100
    return list(per_dif_array)

def get_max_deviation():
    """
    :return: (float) Maximum percentage difference found between the field strength at a point and the origin
    """
    difference_list = get_per_dif_list()
    return max(difference_list)

print(f'Maximum percentage deviation found is: {get_max_deviation():.3f}%')

plot_Bx_By_field()
plot_field_strength()