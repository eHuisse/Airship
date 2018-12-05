import numpy as np
from matplotlib import pyplot as plt

a_1 = 1.000000
a_2 = 3.462153
a_3 = -26.960996
a_4 = 59.357210
a_5 = -56.480030
a_6 = 19.62167

# Inputs BaloonShape
#nb of point
n = 50
#nb of gore
nG = 2
#length 2 diam ration
L2D = 3
#Prism
Cp = 0.7
Cb = Cp * np.pi / 4
#block volume
Vb = 1.9
V = Vb * Cb
L = (V*(L2D ** 2)*4/(Cp*np.pi)) ** (1/3)
D = L / L2D

Vsphere = 1.0
Radius = ((3/4)*Vsphere/np.pi)**(1/3)

# Input Gimbal/Empennage
Gimbal_weight = 0.350
Empennage_weight = 0.150
x_emp = 1
Sheath_density = 0.05


rho_air = 1.225
rho_helium = 0.169
g = 9.81

def baloon_shape(n):
    #creation vecteur lineaire 1,2,...,n
    i = [j for j in range(n+1)]

    phi = [j*np.pi/n for j in range(n+1)]

    x2L = [(1-np.cos(phi[j]))/2 for j in range(n+1)]

    r2D2 = [a_1 * x2L[j] + a_2 * x2L[j] ** 2 + a_3 * x2L[j] ** 3 +
            a_4 * x2L[j] ** 4 + a_5 * x2L[j] ** 5 + a_6 * x2L[j] ** 6 for j in range(n+1)]

    r2D = [np.sqrt(r2D2[j]) for j in range(n+1)]

    x = [x2L[j] * L for j in range(n+1)]

    r = [r2D[j] * D for j in range(n+1)]

    minus_r = [-r[j] for j in range(n+1)]

    x_prime = [(x[j-1] + x[j])/2 for j in range(1, n+1)]
    x_prime.insert(0, 0)

    dx = [(x[j] - x[j-1]) for j in range(1, n+1)]
    dx.insert(0, 0)

    dr = [(r[j] - r[j-1]) for j in range(1, n+1)]
    dr.insert(0, 0)

    ds = [np.sqrt(dx[j] ** 2 + dr[j] ** 2) for j in range (n+1)]

    s = [np.float64(0.)]
    for j in range(1, n+1):
        s.append(s[j-1] + ds[j])

    Bshape_pos = [np.pi * r[j] / nG for j in range(n+1)]
    Bshape_min = [-Bshape_pos[j] for j in range(n+1)]

    return x, r, minus_r, s, Bshape_pos, Bshape_min

def baloon_shape_shpere(n):
    #creation vecteur lineaire 1,2,...,n
    i = [j for j in range(n+1)]

    phi = [j*np.pi/n for j in range(n+1)]

    x2L = [(1-np.cos(phi[j]))/2 for j in range(n+1)]

    x = [x2L[j] * 2 * Radius for j in range(n+1)]
    print(x)

    #r = [r2D[j] * D for j in range(n+1)]

    #minus_r = [-r[j] for j in range(n+1)]

    r = [np.sqrt(Radius**2 - (x[j]-Radius) ** 2) for j in range(n+1)]
    print(r)
    minus_r = [-r[j] for j in range(n + 1)]

    x_prime = [(x[j-1] + x[j])/2 for j in range(1, n+1)]
    x_prime.insert(0, 0)

    dx = [(x[j] - x[j-1]) for j in range(1, n+1)]
    dx.insert(0, 0)

    dr = [(r[j] - r[j-1]) for j in range(1, n+1)]
    dr.insert(0, 0)

    ds = [np.sqrt(dx[j] ** 2 + dr[j] ** 2) for j in range (n+1)]

    s = [np.float64(0.)]
    for j in range(1, n+1):
        s.append(s[j-1] + ds[j])

    Bshape_pos = [np.pi * r[j] / nG for j in range(n+1)]
    Bshape_min = [-Bshape_pos[j] for j in range(n+1)]

    return x, r, minus_r, s, Bshape_pos, Bshape_min

def lift_force(baloon_shape_pos, x):
    if len(baloon_shape_pos) != len(x):
        raise('baloon_shape_min, baloon_shape_pos, x different in shape')

    volume = 0
    x_B = 0

    for i in range(len(x)-1):
        h = x[i+1] - x[i]
        volume_tmp = (h * np.pi / 3) * (baloon_shape_pos[i+1] ** 2 + baloon_shape_pos[i] ** 2
                                             + baloon_shape_pos[i+1] * baloon_shape_pos[i])
        x_buyo = x[i] + (x[i+1] - x[i])/2

        x_B = x_B + volume_tmp * x_buyo

        volume = volume + volume_tmp

    x_B = x_B / volume
    lift = volume * (rho_air - rho_helium) * g

    return lift, x_B, volume * rho_helium

def sheath_mass_center(baloon_shape_pos, x):
    surface = 0
    x_G = 0
    for i in range(len(x)-1):
        x_tmp = x[i] + (x[i + 1] - x[i]) / 2
        h = x[i+1] - x[i]
        apotheme = np.sqrt(h ** 2 + abs((baloon_shape_pos[i] - baloon_shape_pos[i+1])) ** 2)
        surface_tmp = np.pi * (baloon_shape_pos[i] + baloon_shape_pos[i+1]) * apotheme
        x_G = x_G + surface_tmp * Sheath_density * x_tmp
        surface = surface + surface_tmp

    sheath_mass = surface * Sheath_density
    x_G = x_G / sheath_mass
    return sheath_mass, x_G


def gimbal_placement(x_buyo, x_emp, x_sheath, M_gimbal, M_emp, M_sheath):
    return x_buyo - ((x_emp - x_buyo) * M_emp + (x_sheath - x_buyo) * M_sheath) / M_gimbal


def center_of_mass(center_of_mass, mass):
    if len(center_of_mass) != len(mass):
        raise('len of mass have to be the same as center of mass')

    x = 0
    total_mass = 0
    for j in range(len(mass)):
        x = x + mass[j] * center_of_mass[j]
        total_mass = total_mass + mass[j]

    return x / total_mass


if __name__ == "__main__":
    x, r, minus_r, s, Bshape_pos, Bshape_min = baloon_shape(n)

    liftF, x_B, helium_mass = lift_force(r, x)
    sheath_mass, x_G_sheath = sheath_mass_center(r, x)
    x_gimbal = gimbal_placement(x_B, x_emp, x_G_sheath, Gimbal_weight, Empennage_weight, sheath_mass)

    x_center_of_mass = center_of_mass([x_B, x_G_sheath, x_emp, x_gimbal], [helium_mass, sheath_mass, Empennage_weight, Gimbal_weight])

    print("liftF : " + str(liftF))
    print("sheath mass : " + str(sheath_mass))
    print("X gimbal : " + str(x_gimbal))
    print('X sheath : ' + str(x_G_sheath))
    print('X CoM : ' + str(x_center_of_mass))
    print('X buoyancy : ' + str(x_B))

    plt.figure()
    plt.plot(x, r, 'b-', x, minus_r, 'b-')
    plt.plot(x_B, 0, 'ro')
    plt.text(x_B, 0.01, 'buoyancy', color='red')
    plt.plot(x_G_sheath, 0, 'go')
    plt.text(x_G_sheath, -0.02, 'Sheath', color='green')
    plt.plot(x_emp, 0, 'co')
    plt.text(x_emp, 0.01, 'Empennage', color='cyan')
    plt.plot(x_gimbal, 0, 'mo')
    plt.text(x_gimbal, 0, 'Gimbal', color='magenta')
    plt.plot(x_center_of_mass, 0, 'k*')
    plt.text(x_center_of_mass, -0.05, 'Total mass', color='black')
    plt.show()
    plt.figure()
    plt.plot(s, Bshape_pos, 'b-', s, Bshape_min, 'b-')
    plt.show()