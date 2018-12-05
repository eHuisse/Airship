import numpy as np
from matplotlib import pyplot as plt

'''
File wrote mostly based on Development of an Aerodynamic Model and Control

Law Design for a High Altitude Airship

Joseph B. Mueller∗ and Michael A. Paluszek†
Princeton Satellite Systems, Princeton, NJ 08542

Yiyuan Zhao‡

University of Minnesota, Minneapolis, MN 55455
'''

class airship(object):
    def __init__(self, n_discret = 50, isEllipsoid=True):
        self.half_gondola_height = 0.1
        self.half_fins_height = 0.15
        self.CoMpos = np.array([0, 0, 0]) #OUTPUT Position of the total center of mass
        self.CoBpos = np.array([0, 0, 0]) #OUTPUT Position of the center of buyoancy
        self.CoVpos = np.array([0, 0, 0]) #OUTPUT Position of the center of volume
        self.CoMpos_sheath = np.array([0, 0, 0]) #OUTPUT position of the center of mass of sheath
        self.sheath_mass = 0
        self.CoMpos_gimbal = np.array([0, 0, 0]) #OUTPUT position of the center of mass of the gimbal
        self.gimbal_mass = 0.3
        self.CoMpos_fins = np.array([2.2, 0, 0]) #INPUT position of the center of mass of fins
        self.fins_mass = 0.15

        self.a = 0 #Semi major axis
        self.b = 0 #Semi minor axis
        self.g = 9.81 #gravity constant
        self.helium_mass = 0 #Mass of helium
        self.rho_air = 1.225 #Density of air
        self.rho_helium = 0.169 #Density of helium
        self.sheath_density = 0.05 #Sheath density kg/m2
        self.hull_surface = 0
        self.fins_surface = 0.05
        self.gondola_surface = 0.001
        self.liftforce = 0 #Lift force of the balloon
        self.ballonShape = np.array([]) #containing the balloon shape
        self.goreShape = np.array([]) #containing the gore shape
        self.balloon_abscisse = np.array([]) #Balloon abscisse
        self.gore_abscisse = np.array([])
        self.shape_poly_coef = {"a1": 1, "a2": 3.462153, "a3": -26.960996,
                                "a4": 59.357210, "a5": -56.480030, "a6": 19.62167}
        self.hull_inertia = {}


        if isEllipsoid:
            #Defining all the corners of fins in meter x axis of polygone is longitudinal
            self.fins_polygone = np.array([[], []])
            self.balloon_shape_ellipsoid()
            self.addedmass_compute()
            self.sheath_mass_center()
            self.lift_force()
            self.inertia_compute_ellipsoid()
            self.place_fins_zmass()
            self.gimbal_placement()

            # Aerodynamic
            self.CDh0 = 0.025  # Hull drag coefficient 0 incidence
            self.CDf0 = 0.006  # Fins drag coefficient 0 incidence
            self.CDg0 = 1  # Gondola drag coefficient 0 incidence
            self.CDhc = 0.5  # Hull drag coefficient cross flow
            self.CDfc = 1  # Fins drag coefficient cross flow
            self.CDgc = 1  # Gondola drag coefficient cross flow
            self.DLCaa = 5.73  # Derivative of fin lift-coefficient with respect to the angle-of-attack at zero incidence
            self.DLCfd = 1.24  # Derivative of fin lift-coefficient with respect to the flap deflection angle
            self.fin_efficiency = 0.29  # Fin efficiency factor accounting for the effect of the hull on the fins
            self.hull_efficiency = 1.19  # Hull efficiency factor accounting for the effect of the fins on the hull
            self.drag_coefficient = {}

            self.compute_dragcoef()

        else:
            self.balloon_shape_shpere()
            self.addedmass_compute()
            self.sheath_mass_center()
            self.lift_force()
            self.inertia_compute_sphere()

        self.print_airship()
        self.plot_balloon()

    def place_fins_zmass(self):
        idx = np.argsort(abs(self.balloon_abscisse - self.CoMpos_fins[0]))[0]
        print('idx' +str(self.ballonShape))
        self.CoMpos_fins[2] = - self.ballonShape[0, idx] - self.half_fins_height

    def compute_dragcoef(self):
        X1 = -(self.CDh0*self.hull_surface+self.CDf0*self.fins_surface+self.CDg0*self.gondola_surface)
        X2 = (self.addedmass_coef['k_trans'] - self.addedmass_coef['k_long'])\
             *self.hull_efficiency*self.hull_inertia['Ixx']*self.hull_surface
        Y1 = X2
        Y2 = -0.5*self.DLCaa*self.fins_surface*self.fin_efficiency
        Y3 = -(self.CDhc)


    def print_airship(self):
        print('CoM : ' + str(self.CoMpos))
        print('CoB : ' + str(self.CoBpos))
        print('CoV : ' + str(self.CoVpos))
        print('CoM_sheath : ' + str(self.CoMpos_sheath))
        print('CoM_gimbal : ' + str(self.CoMpos_gimbal))
        print('CoM_fins : ' + str(self.CoMpos_fins))
        print('a : ' + str(self.a))
        print('b : ' + str(self.b))
        print('g : ' + str(self.g))
        print('helium_mass : ' + str(self.helium_mass))
        print('rho air : ' + str(self.rho_air))
        print('rho helium : ' + str(self.rho_helium))
        print('sheath density : ' + str(self.sheath_density))
        print('sheath mass : ' + str(self.sheath_mass))
        print('lift force : ' + str(self.liftforce))
        #print('balloon shape : ' + str(self.ballonShape))
        #print('gore shape : ' + str(self.goreShape))
        #print('Balloon absc : ' + str(self.balloon_abscisse))
        print('shape poly coef : ' + str(self.shape_poly_coef))
        print('added mass coef' + str(self.addedmass_coef))
        print('hull inertia' + str(self.hull_inertia))
#        print('' + str())


    def plot_balloon(self):
        '''
        This function plot balloon and all mass center to look if everything fine
        :return:
        '''
        plt.figure()
        plt.plot(self.balloon_abscisse, self.ballonShape[0,:], 'b-', self.balloon_abscisse, self.ballonShape[1,:], 'b-')
        plt.plot(self.CoBpos[0], self.CoBpos[2], 'ro')
        plt.text(self.CoBpos[0], self.CoBpos[2] + 0.01, 'buoyancy', color='red')
        plt.plot(self.CoMpos_sheath[0], self.CoMpos_sheath[2], 'go')
        plt.text(self.CoMpos_sheath[0], self.CoMpos_sheath[2] - 0.02, 'Sheath', color='green')
        plt.plot(self.CoMpos_fins[0], self.CoMpos_fins[2], 'co')
        plt.text(self.CoMpos_fins[0], self.CoMpos_fins[2] + 0.01, 'Empennage', color='cyan')
        plt.plot(self.CoMpos_gimbal[0], self.CoMpos_gimbal[2], 'mo')
        plt.text(self.CoMpos_gimbal[0], self.CoMpos_gimbal[2], 'Gimbal', color='magenta')
        plt.plot(self.CoMpos[0], self.CoMpos[2], 'k*')
        plt.text(self.CoMpos[0], self.CoMpos[2] - 0.05, 'Total mass', color='black')
        plt.show()
        plt.figure()
        plt.plot(self.gore_abscisse, self.goreShape[0, :], 'b-', self.gore_abscisse, self.goreShape[1, :], 'b-')
        plt.show()

    def balloon_shape_shpere(self, n=50, volume = 1., nG = 2):
        '''
        This function return the shape of a Spherical balloon.
        :param n: Number of points for shape discretisation
        :param volume: Volume of the hull
        :param nG: Number of gore
        :return:
        '''

        Radius = ((3 / 4) * volume / np.pi) ** (1 / 3)

        # creation vecteur lineaire 1,2,...,n
        i = [j for j in range(n + 1)]

        phi = [j * np.pi / n for j in range(n + 1)]

        x2L = [(1 - np.cos(phi[j])) / 2 for j in range(n + 1)]

        x = [x2L[j] * 2 * Radius for j in range(n + 1)]

        r = [np.sqrt(Radius ** 2 - (x[j] - Radius) ** 2) for j in range(n + 1)]

        minus_r = [-r[j] for j in range(n + 1)]

        x_prime = [(x[j - 1] + x[j]) / 2 for j in range(1, n + 1)]
        x_prime.insert(0, 0)

        dx = [(x[j] - x[j - 1]) for j in range(1, n + 1)]
        dx.insert(0, 0)

        dr = [(r[j] - r[j - 1]) for j in range(1, n + 1)]
        dr.insert(0, 0)

        ds = [np.sqrt(dx[j] ** 2 + dr[j] ** 2) for j in range(n + 1)]

        s = [np.float64(0.)]
        for j in range(1, n + 1):
            s.append(s[j - 1] + ds[j])

        Bshape_pos = [np.pi * r[j] / nG for j in range(n + 1)]
        Bshape_min = [-Bshape_pos[j] for j in range(n + 1)]

        self.balloon_abscisse = np.array(x)
        self.ballonShape = np.vstack((np.array(r),np.array(minus_r)))
        self.gore_abscisse = np.array(s)
        self.goreShape = np.vstack((np.array(Bshape_pos),np.array(Bshape_min)))
        self.a = Radius
        self.b = Radius

        return x, r, minus_r, s, Bshape_pos, Bshape_min

    def balloon_shape_ellipsoid(self, n=50, volume=1, nG=2, L2D=3):
        '''
        This function return the shape of a Spherical balloon.
        :param n: Number of points for shape discretisation
        :param volume: Volume of the hull
        :param nG: Number of gore
        :return:
        '''
        # Prism
        Cp = 0.7
        Cb = Cp * np.pi / 4
        Vb = volume/Cp
        L = (volume * (L2D ** 2) * 4 / (Cp * np.pi)) ** (1 / 3)
        D = L / L2D
        # creation vecteur lineaire 1,2,...,n
        i = [j for j in range(n + 1)]

        phi = [j * np.pi / n for j in range(n + 1)]

        x2L = [(1 - np.cos(phi[j])) / 2 for j in range(n + 1)]

        r2D2 = [self.shape_poly_coef['a1'] * x2L[j] + self.shape_poly_coef['a2'] * x2L[j] ** 2 + self.shape_poly_coef['a3'] * x2L[j] ** 3 +
                self.shape_poly_coef['a4'] * x2L[j] ** 4 + self.shape_poly_coef['a5'] * x2L[j] ** 5 + self.shape_poly_coef['a6'] * x2L[j] ** 6
                for j in range(n + 1)]

        r2D = [np.sqrt(r2D2[j]) for j in range(n + 1)]

        x = [x2L[j] * L for j in range(n + 1)]

        r = [r2D[j] * D for j in range(n + 1)]

        minus_r = [-r[j] for j in range(n + 1)]

        x_prime = [(x[j - 1] + x[j]) / 2 for j in range(1, n + 1)]
        x_prime.insert(0, 0)

        dx = [(x[j] - x[j - 1]) for j in range(1, n + 1)]
        dx.insert(0, 0)

        dr = [(r[j] - r[j - 1]) for j in range(1, n + 1)]
        dr.insert(0, 0)

        ds = [np.sqrt(dx[j] ** 2 + dr[j] ** 2) for j in range(n + 1)]

        s = [np.float64(0.)]
        for j in range(1, n + 1):
            s.append(s[j - 1] + ds[j])

        Bshape_pos = [np.pi * r[j] / nG for j in range(n + 1)]
        Bshape_min = [-Bshape_pos[j] for j in range(n + 1)]

        self.balloon_abscisse = np.array(x)
        self.ballonShape = np.vstack((np.array(r),np.array(minus_r)))
        self.gore_abscisse = np.array(s)
        self.goreShape = np.vstack((np.array(Bshape_pos),np.array(Bshape_min)))
        self.b = max(r)
        self.a = L/2

        return x, r, minus_r, s, Bshape_pos, Bshape_min

    def addedmass_compute(self):
        '''
        express in Center of volume. 
        :param a: length
        :param b: diameter
        :return:
        '''
        if self.b == self.a:
            k1 = 0.5
            k2 = 0.5
            k3 = 0
            self.addedmass_coef = {'k_longit': k1, 'k_trans': k2, 'k_rot': k3}
            print(self.addedmass_coef)
            return {'k_longit': k1, 'k_trans': k2, 'k_rot': k3}


        e = np.sqrt(1 - (self.b / self.a) ** 2)
        f = np.log((1 + e) / (1 - e))
        gamma = (2 * (1 - e ** 2) / e ** 3) * (0.5 * f - e)
        alpha = (1 / e ** 2) - ((1 - e ** 2) / (2 * e ** 3)) * f
        k1 = gamma / (2 - gamma)
        k2 = alpha / (2 - alpha)
        k3 = (e ** 4 * (alpha - gamma)) / ((2 - e ** 2) * (2 * e ** 2 - (2 - e ** 2) * (alpha - gamma)))

        self.addedmass_coef = {'k_longit': k1, 'k_trans': k2, 'k_rot': k3}
        #print(self.addedmass_coef)
        return {'k_longit': k1, 'k_trans': k2, 'k_rot': k3}

    def lift_force(self):
        '''
        Compute the lift force and its application point
        :return:
        '''
        volume = 0
        x_B = 0

        for i in range(len(self.balloon_abscisse) - 1):
            h = self.balloon_abscisse[i + 1] - self.balloon_abscisse[i]
            volume_tmp = (h * np.pi / 3) * (self.ballonShape[0, i + 1] ** 2 + self.ballonShape[0, i] ** 2
                                            + self.ballonShape[0, i + 1] * self.ballonShape[0, i])
            x_buyo = self.balloon_abscisse[i] + (self.balloon_abscisse[i + 1] - self.balloon_abscisse[i]) / 2

            x_B = x_B + volume_tmp * x_buyo

            volume = volume + volume_tmp

        x_B = x_B / volume

        self.CoVpos = np.array([x_B, 0, 0])
        self.CoBpos = np.array([x_B, 0, 0])
        self.CoMpos = np.array([x_B, 0, 0])
        self.liftforce = volume * (self.rho_air - self.rho_helium) * self.g
        self.helium_mass = volume * self.rho_helium

    def sheath_mass_center(self):
        '''
        Compute the center of mass of sheath and the the mass of sheath
        :return:
        '''
        surface = 0
        x_G = 0.
        for i in range(len(self.balloon_abscisse) - 1):
            x_tmp = self.balloon_abscisse[i] + (self.balloon_abscisse[i + 1] - self.balloon_abscisse[i]) / 2
            h = self.balloon_abscisse[i + 1] - self.balloon_abscisse[i]
            apotheme = np.sqrt(h ** 2 + abs((self.ballonShape[0, i] - self.ballonShape[0, i + 1])) ** 2)
            surface_tmp = np.pi * (self.ballonShape[0, i] + self.ballonShape[0, i + 1]) * apotheme
            x_G = x_G + surface_tmp * self.sheath_density * x_tmp
            surface = surface + surface_tmp
        print(x_G)
        self.hull_surface = surface
        self.sheath_mass = surface * self.sheath_density
        self.CoMpos_sheath[0] = x_G / self.sheath_mass

    def inertia_compute_ellipsoid(self):
        '''
        express on the center of volume.
        :return:
        '''
        Ixx_he = 0.2 * self.helium_mass * 2 * self.b**2
        Iyy_he = 0.2 * self.helium_mass * (self.b**2+self.a**2)
        Izz_he = 0.2 * self.helium_mass * (self.b**2+self.a**2)
        Ixx_hu = (1/3) * self.sheath_mass * 2 * self.b**2
        Iyy_hu = (1/3) * self.sheath_mass * (self.b**2+self.a**2)
        Izz_hu = (1/3) * self.sheath_mass * (self.b**2+self.a**2)

        self.hull_inertia = {'Ixx': Ixx_he+Ixx_hu, 'Iyy': Iyy_he+Iyy_hu, 'Izz': Izz_he+Izz_hu}

    def inertia_compute_sphere(self):
        '''
        express on the center of volume.
        :return:
        '''
        Ixx_he = 0.2 * self.helium_mass * 2 * self.b**2
        Iyy_he = 0.2 * self.helium_mass * (self.b**2+self.a**2)
        Izz_he = 0.2 * self.helium_mass * (self.b**2+self.a**2)
        Ixx_hu = (2/3) * self.sheath_mass * self.b**2
        Iyy_hu = (2/3) * self.sheath_mass * self.b**2
        Izz_hu = (2/3) * self.sheath_mass * self.b**2

        self.hull_inertia = {'Ixx': Ixx_he+Ixx_hu, 'Iyy': Iyy_he+Iyy_hu, 'Izz': Izz_he+Izz_hu}

    def center_of_mass(self, mass, coordinates):
        tmp_mass = 0
        xtmp = 0
        ytmp = 0
        ztmp = 0
        for i in range(mass):
            xtmp = xtmp + coordinates[i][0] * mass[i]
            ytmp = ytmp + coordinates[i][1] * mass[i]
            ztmp = ztmp + coordinates[i][2] * mass[i]
            tmp_mass = tmp_mass + mass[i]

        self.CoMpos = np.array([xtmp/tmp_mass, ytmp/tmp_mass, ztmp/tmp_mass])

    def gimbal_placement(self):
        self.CoMpos_gimbal[0] = self.CoBpos[0] - ((self.CoMpos_fins[0] - self.CoBpos[0]) * self.fins_mass +(self.CoMpos_sheath[0] - self.CoBpos[0]) * self.sheath_mass) / self.gimbal_mass
        print('idx    ' +str(self.CoMpos_gimbal[0]))
        idx = np.argsort(abs(self.balloon_abscisse - self.CoMpos_gimbal[0]))[0]
        self.CoMpos_gimbal[2] = - self.ballonShape[0, idx] - self.half_gondola_height


if __name__ == "__main__":
    bouboule = airship(isEllipsoid=True)

def addedmass_coef(a, b):
    '''

    :param a: length
    :param b: diameter
    :return:
    '''
    e = np.sqrt(1 - (b/a)**2)
    f = np.log((1+e)/(1-e))
    gamma = (2 * (1 - e**2) / e**3) * ( 0.5 * f - e)
    alpha = (1 / e**2) - ((1 - e**2) / (2 * e**3)) * f
    k1 = gamma / (2 - gamma)
    k2 = alpha / (2 - alpha)
    k3 = (e**4 * (alpha - gamma))/((2 - e**2)*(2 * e**2 - (2 - e**2)*(alpha-gamma)))

    return k1, k2, k3

dia = 1
long = np.linspace(1.1, 10, 100)
result = np.zeros((3, 100))

for i in range(100):
    k1, k2, k3 = addedmass_coef(long[i], dia)
    result[0, i] = k1
    result[1, i] = k2
    result[2, i] = k3

plt.plot(long, result[0, :], 'ro',long, result[1, :], 'bo',long, result[2, :], 'go')
plt.title('Added mass coefficient')
plt.xlabel('Fineness Ration')
plt.ylabel('Coefficient value')
plt.show()