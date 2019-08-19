import numpy as np
from scipy.constants import mu_0, pi
import pandas as pd
import matplotlib.pyplot as plt

N = 63 # Number of coils to create

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

    n = 20 # Number of segment wires that form the coil
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

    def __init__(self, x_lower, x_upper, x_num, y_lower, y_upper, y_num, z_lower, z_upper, z_num):
        """
        :param x_lower: (int or float) Lower limit to the x dimension that forms the space
        :param x_upper: (int or float) Upper limit to the x dimension that forms the space
        :param x_num: (int) Number of points in the x dimension of space
        :param y_lower: (int or float) Lower limit to the y dimension that forms the space
        :param y_upper: (int or float) Upper limit to the y dimension that forms the space
        :param y_num: (int) Number of points in the y dimension of space
        :param z_lower: (int or float) Lower limit to the z dimension that forms the space
        :param z_upper: (int or float) Upper limit to the z dimension that forms the space
        :param z_num: (int) Number of points in the z dimension of space
        """
        # The Space object holds attributes that define the lower and upper limit of each dimension in 3D space
        # along with how many points to create in each dimension
        # Lower limit of dimension
        self.x_lower = x_lower
        self.y_lower = y_lower
        self.z_lower = z_lower
        # Upper limit of dimension
        self.x_upper = x_upper
        self.y_upper = y_upper
        self.z_upper = z_upper
        # How many points to create in each dimension
        self.x_num = x_num
        self.y_num = y_num
        self.z_num = z_num
        # Create the grid that forms the space, for each of these points we will find the B field
        self.x_list = np.linspace(x_lower, x_upper, x_num)
        self.y_list = np.linspace(y_lower, y_upper, y_num)
        self.z_list = np.linspace(z_lower, z_upper, z_num)

    def get_data_at_point(self, x, y, z, coil_list):
        """
        :param x: (float) x component of point in space we want to get B field for
        :param y: (float) y component of point in space we want to get B field for
        :param z: (float) z component of point in space we want to get B field for
        :param coil_list: (Coil list) Holds Wire objects that form the coil
        :return: (7-tuple) x, y, z, Bx, By, Bz, |B| for a point in space
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

    def get_data_for_all_points(self, list_of_all_coils):
        """
        :param list_of_all_coils: (Coils list) Holds all the Coil objects to be created
        :return: (7-tuple) x, y, z, Bx, By, Bz, |B| lists for all points in space
        """
        # Lists to put the x, y, z values that the B field has been evaluated at
        px_list = []
        py_list = []
        pz_list = []
        # Total number of points to evaluate the B field at
        total_num = self.x_num*self.y_num*self.z_num
        # Arrays to accumulate the final B field from all the coils (initialised to 0)
        total_Bx = np.zeros(total_num)
        total_By = np.zeros(total_num)
        total_Bz = np.zeros(total_num)
        total_B_mag = np.zeros(total_num)
        # After iteration 0 we don't need to store x, y, z values as they are the same space for all coils
        iteration = 0

        for coil_obj in list_of_all_coils:
            # Lists to hold the B field solution for 1 coil temporarily
            tmp_Bx = []
            tmp_By = []
            tmp_Bz = []
            tmp_B_mag = []
            # List of wires that define the specific coil in the loop
            coil_wires = coil_obj.coil_list
            # Traverse throughout all the space and evaluate the B field
            for x in self.x_list:
                for y in self.y_list:
                    for z in self.z_list:
                        # Gets data for a point in space (x, y, z)
                        px, py, pz, Bx, By, Bz, B_mag = self.get_data_at_point(x, y, z, coil_wires)
                        # After iteration 0, we don't store position values
                        if iteration == 0:
                            px_list.append(px)
                            py_list.append(py)
                            pz_list.append(pz)
                        # Append the temporary data that form the solution for 1 coil
                        tmp_Bx.append(Bx)
                        tmp_By.append(By)
                        tmp_Bz.append(Bz)
                        tmp_B_mag.append(B_mag)
            # Add the contribution of this 1 coil to the total magnetic field solution
            total_Bx = total_Bx + np.array(tmp_Bx)
            total_By = total_By + np.array(tmp_By)
            total_Bz = total_Bz + np.array(tmp_Bz)
            total_B_mag = total_B_mag + np.array(tmp_B_mag)
            # Makes sure the position data are only appended once to the initial position lists
            iteration += 1

        return px_list, py_list, pz_list, total_Bx, total_By, total_Bz, total_B_mag

# The length of the x dimension is set to D = 10R which forms the fixed distance between the outermost coils
x_dimension_length = 10*Coil.radius
# Separation distance between the coils
dist = x_dimension_length/(N-1)
# Generate the x values to place the coils uniformly spaced at
x_placements = np.linspace(-x_dimension_length/2, x_dimension_length/2, N)

# Initialise the space object that we place the coils in
space = Space(-(x_dimension_length+3)/2, (x_dimension_length+3)/2,50, -1.6,1.6,50, 0,1,1)

# This list will hold N Coil objects
all_coils = []
for x_position in x_placements:
    all_coils.append(Coil(px = x_position, py = 0, pz = 0))

# Generate data for all points in space
px_data, py_data, pz_data, Bx_data, By_data, Bz_data, B_mag_data = space.get_data_for_all_points(all_coils)

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
    # Give colour to the vectors
    c = B_mag_data.copy()
    c = (np.ravel(c) - np.min(c)) / np.ptp(c)
    c = np.concatenate((c, np.repeat(c, 2)))
    c = plt.cm.inferno(c)
    fig = plt.figure()
    ax = fig.gca()
    ax.quiver(px_data, py_data, Bx_data, By_data, scale = 60, pivot = 'middle', color = c)
    plt.xlabel('x')
    plt.ylabel('y')
    # Center depends on px, py, pz when initializing the Coil object
    plt.title(f'Vector field for Bx, By\n{N} coils with radius 1')
    plt.savefig(f'vector_field_{N}_coils.pdf', bbox = 'tight')
    plt.show()

def plot_field_strength():
    # Plot the contour of the magnetic field strength in the x-y plane
    cp = plt.tricontourf(np.array(px_data), np.array(py_data), np.array(B_mag_data))
    plt.colorbar(cp)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Contour plot for the magnetic field strength in the x-y plane\n{N} coils with radius 1')
    plt.savefig(f'contour_strength_{N}_coils.pdf', bbox = 'tight')
    plt.show()

plot_Bx_By_field()
plot_field_strength()
