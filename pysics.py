import numpy as np
from numba import jit
import copy
from PhysicalEngine import Velret
from wind_sim import Wind
from drawnow import drawnow
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib as mpl
import time

# General functions:
@jit
def cart2pol(x, y):
    rho = np.sqrt(np.square(x) + np.square(y))
    phi = np.arctan2(y, x)%(2*np.pi)
    return rho, phi


@jit
def angle_split(theta):
    return np.array([np.cos(theta), np.sin(theta)])


@jit
def one_hot(idx, length):
    one_hot_array = np.zeros((length,))
    one_hot_array[idx] = 1.
    return one_hot_array


@jit
def rot_mat(theta):
    R = np.zeros(shape=(2,2))
    c_th = np.cos(theta)
    s_th = np.sin(theta)
    R[0, 0] = R[1, 1] = c_th
    R[0, 1] = -s_th
    R[1, 0] = s_th
    return R

# Action Space methods:
class Box:
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape

# Environment template:
class envName:
    def __init__(self):
        # Engine = Velret(x0,v0,F,m,dt)
        pass

    def action_space(self):
        # action_space = Box(low=, high=, shape=)
        pass

    def reset(self):
        pass
        # return self.S

    @jit
    def step(self, action):
        pass
        # return self.S, self.reward, self.done, self.info

    def render(self):
        drawnow(self.make_fig)

    def make_fig(self):
        # plt.plot(state_history[:,0])
        pass

# Environmens:

class Spaceship:
    '''
    continuous linear action:
    action = [engine_A, engine_B]

    '''
    def __init__(self):
        # Engine = Velret(x0,v0,F,m,dt)
        self.dt = 1e-2
        self.eta = 1e-2
        self.k = 0
        AA = np.zeros(shape=(3,3))
        AA[0,0] = AA[1,1] = 1
        AA[0,1] = AA[1,2] = self.dt
        AA[2,0] = -self.k
        AA[2,1] = -self.eta
        self.A = np.zeros(shape=(6,6))
        self.A[:3,:3] = self.A[3:,3:] = AA

    def reset(self):
        self.reward = 0
        self.S = np.zeros(6)
        self.F = np.zeros(6)
        return self.S

    @jit
    def step(self, action):
        self.F[2] = action[0]
        self.F[5] = action[1]
        self.S = np.dot(self.A, self.S) + self.F
        self.reward = 1 + 0*np.sqrt(self.S[1]**2 + self.S[4]**2)
        self.done = False
        if np.abs(self.S[0])>1 or np.abs(self.S[3])>1:
            self.done = True
        self.info = ""
        return self.S, self.reward, self.done, self.info

    def render(self):
        drawnow(self.make_fig)

    def make_fig(self):
        plt.plot(self.S[0],self.S[3],'o')


class Unstable1D:
    '''
    discrete action space:
    action = [0,1] :
        0: throttle -1
        1: throttle +1
    '''
    def __init__(self):
        # Engine = Velret(x0,v0,F,m,dt)
        self.dt = 1e-1
        self.eta = 0
        self.k = -0.5
        self.edge = 1
        self.max_steps = int(50/self.dt)
        AA = np.zeros(shape=(3,3))
        AA[0,0] = AA[1,1] = 1
        AA[0,1] = AA[1,2] = self.dt
        AA[2,0] = -self.k
        AA[2,1] = -self.eta
        self.A = AA
        # Throttle control:
        self.throttle_max = 5
        self.throttle_span = np.arange(-self.throttle_max, self.throttle_max+1)
        self.throttle_values = np.concatenate(([-self.throttle_max], self.throttle_span))
        # for rendering
        self.x = np.linspace(-self.edge, self.edge, 100)
        self.y = self.k * self.x**2

    def reset(self):
        self.throttle = np.array([0])
        self.reward = 0
        self.S = np.zeros(3)
        self.F = np.zeros(3)
        self.S[0] = 0.01 * (2*np.random.rand()-1)
        self.steps = 0
        self.wind = 0.8*(2*np.random.rand()-1)
        return np.concatenate((self.S, self.throttle))

    @jit
    def step(self, action):
        # wind = 0.5*np.sin(2*np.pi*0.01*self.steps+self.rand_phase)
        self.throttle += 2 * action - 1
        self.throttle = self.throttle_values[np.digitize(self.throttle, self.throttle_span)]
        self.F[2] = self.throttle + self.wind
        self.S = np.dot(self.A, self.S) + self.F
        self.reward = 1
        self.steps += 1
        self.done = False
        if np.abs(self.S[0]) > self.edge or self.steps >= self.max_steps:
            self.done = True
        self.info = ""
        return np.concatenate((self.S, self.throttle)), self.reward, self.done, self.info

    def render(self):
        drawnow(self.make_fig)

    def make_fig(self):
        plt.plot(self.x, self.y, self.S[0], self.k *self.S[0]**2, 'o')
        plt.title("Time: " + str(np.round(self.steps*self.dt,2)) + "   Force: " + str(np.round(self.F[2],2)) + "\nWind: " + str(np.round(self.wind,2)) + "   Throttle: " + str(np.round(self.throttle.item(),2)))


class Unstable2D:
    '''
    discrete action space:
    action = [0,1] :
        0: x throttle -1
        1: x throttle +1
        2: y throttle -1
        3: y throttle +1
    '''
    def __init__(self):
        # Engine = Velret(x0,v0,F,m,dt)
        self.dt = 1e-1
        self.eta = 0
        self.k = -1
        self.edge = 1
        self.max_steps = int(50/self.dt)
        self.A = np.zeros(shape=(6,6))
        AA = np.zeros(shape=(3,3))
        AA[0,0] = AA[1,1] = 1
        AA[0,1] = AA[1,2] = self.dt
        AA[2,0] = -self.k
        AA[2,1] = -self.eta
        self.A[:3,:3] = self.A[3:,3:] = AA
        # Throttle control:
        self.throttle_max = 20
        self.throttle_span = np.arange(-self.throttle_max, self.throttle_max+1)
        self.throttle_values = np.concatenate(([-self.throttle_max], self.throttle_span))
        self.throttle_F_factor = 0.5
        # Wind:
        self.wind_co = 0.01
        # for rendering
        self.ax = np.linspace(-self.edge, self.edge, 25)
        self.x, self.y = np.meshgrid(self.ax,self.ax)
        self.z = -self.k * (self.x**2 + self.y**2)


    def reset(self):
        self.reward = 0
        self.S = np.zeros(6)
        self.F = np.zeros(6)
        self.S[0] = 0.01 * (2*np.random.rand()-1)
        self.S[3] = 0.01 * (2*np.random.rand()-1)
        self.steps = 0
        self.throttle = np.zeros(shape=2)
        self.wind = self.wind_co*(2*np.random.rand(2)-1)
        self.wind[1] *= 0.5
        self.wind_change_co = self.wind_co/10
        return np.concatenate((self.S, self.throttle))

    @jit
    def step(self, action):
        self.wind += self.wind_change_co * (2 * np.random.rand(2) - 1)
        if action == 0 or action == 1:
            th = 0
        elif action == 2 or action == 3:
            th = 1
            action += -2
        self.throttle[th] += 2 * action - 1
        self.throttle[th] = self.throttle_values[np.digitize(self.throttle[th], self.throttle_span)]
        self.F[2] = self.throttle_F_factor * self.throttle[0] + self.wind[0]
        self.F[5] = self.throttle_F_factor * self.throttle[1] + self.wind[1]
        self.S = np.dot(self.A, self.S) + self.F
        self.reward = 1
        self.steps += 1
        self.done = False
        if np.sqrt(self.S[0]**2 + self.S[3]**2) > self.edge or self.steps >= self.max_steps:
            self.done = True
        self.info = ""
        return np.concatenate((self.S, self.throttle)), self.reward, self.done, self.info

    def render(self):
        drawnow(self.make_fig)

    def make_fig(self):
        plt.imshow(self.z, interpolation='bilinear', cmap=cm.RdYlGn, extent=[-self.edge, self.edge, -self.edge, self.edge])
        # plt.imshow(self.z, interpolation='bilinear', cmap=cm.jet, extent=[-self.edge, self.edge, -self.edge, self.edge])
        # plt.imshow(self.z, cmap=cm.RdYlGn_r, extent=[-self.edge, self.edge, -self.edge, self.edge])
        plt.plot(self.S[0], self.S[3],'ko', markersize=8)
        plt.quiver(0, 0, self.wind[0], self.wind[1], scale=4, units='xy',color='w')
        plt.quiver(self.S[0], self.S[3], self.F[2], self.F[5], scale=4, units='xy',color='g')
        plt.quiver(self.S[0], self.S[3], self.throttle[0], self.throttle[1], scale=4, units='xy',color='k')
        plt.quiver(self.S[0], self.S[3], self.S[1], self.S[4], scale=0.5, units='xy',color='y')
        plt.title("Time: " + str(np.round(self.steps*self.dt,2)) + " sec   Throttle: " + str(np.round(self.throttle,2)) + "\nWind strength: " + str(np.round(np.sqrt(np.dot(self.wind,self.wind)),2)) + "   Wind direction: " + str(np.round(np.rad2deg(np.arctan2(self.wind[1],self.wind[0]))%360,2)) +" deg")


class GravityBalance:
    '''
    discrete action space:
    action = [0,1] :
        0: x throttle -1
        1: x throttle +1
        2: y throttle -1
        3: y throttle +1
    '''
    def __init__(self):
        # Engine = Velret(x0,v0,F,m,dt)
        self.dt = 1e-1
        self.eta = 0.1
        self.k = 0
        self.edge = 1
        self.max_steps = int(20/self.dt)
        self.A = np.zeros(shape=(6,6))
        AA = np.zeros(shape=(3,3))
        AA[0,0] = AA[1,1] = 1
        AA[0,1] = AA[1,2] = self.dt
        AA[2,0] = -self.k
        AA[2,1] = -self.eta
        self.A[:3,:3] = self.A[3:,3:] = AA
        # Throttle control:
        self.throttle_max = 20
        self.throttle_span = np.arange(-self.throttle_max, self.throttle_max+1)
        self.throttle_values = np.concatenate(([-self.throttle_max], self.throttle_span))
        self.throttle_F_factor = 1
        # Wind:
        self.wind_co = 0.01
        # Gravity:
        self.g = -0.01
        # for rendering
        self.ax = np.linspace(-self.edge, self.edge, 25)
        self.x, self.y = np.meshgrid(self.ax,self.ax)
        self.z = -self.k * (self.x**2 + self.y**2)


    def reset(self):
        self.reward = 0
        self.S = np.zeros(6)
        self.F = np.zeros(6)
        self.S[0] = 0.01 * (2*np.random.rand()-1)
        self.S[3] = 0.01 * (2*np.random.rand()-1)
        self.steps = 0
        self.throttle = np.zeros(shape=2)
        self.wind = self.wind_co*(2*np.random.rand(2)-1)
        self.wind[1] *= 0.5
        self.wind_change_co = self.wind_co/10
        return np.concatenate((self.S, self.throttle))

    @jit
    def step(self, action):
        self.wind += self.wind_change_co * (2 * np.random.rand(2) - 1)
        if action == 0 or action == 1:
            th = 0
        elif action == 2 or action == 3:
            th = 1
            action += -2
        self.throttle[th] += 2 * action - 1
        self.throttle[th] = self.throttle_values[np.digitize(self.throttle[th], self.throttle_span)]
        self.F[2] = self.throttle_F_factor * self.throttle[0] + 0*self.wind[0]
        self.F[5] = self.throttle_F_factor * self.throttle[1] + 0*self.wind[1] + self.g
        self.S = np.dot(self.A, self.S) + self.F
        self.reward = 1
        self.steps += 1
        self.done = False
        if np.abs(self.S[0]) > 1 or np.abs(self.S[3]) > 1 or self.steps >= self.max_steps:
            self.done = True
        self.info = ""
        return np.concatenate((self.S, self.throttle)), self.reward, self.done, self.info

    def render(self):
        drawnow(self.make_fig)

    def make_fig(self):
        scale_fac = 8
        plt.imshow(self.z, interpolation='bilinear', cmap=cm.RdYlGn, extent=[-self.edge, self.edge, -self.edge, self.edge])
        # plt.imshow(self.z, interpolation='bilinear', cmap=cm.jet, extent=[-self.edge, self.edge, -self.edge, self.edge])
        # plt.imshow(self.z, cmap=cm.RdYlGn_r, extent=[-self.edge, self.edge, -self.edge, self.edge])
        plt.plot(self.S[0], self.S[3],'ko', markersize=8)
        plt.quiver(0, 0, self.wind[0], self.wind[1]+ self.g, scale=scale_fac, units='xy',color='w')
        plt.quiver(self.S[0], self.S[3], self.F[2], self.F[5], scale=scale_fac, units='xy',color='g')
        plt.quiver(self.S[0], self.S[3], self.throttle[0], self.throttle[1], scale=scale_fac, units='xy',color='k')
        plt.quiver(self.S[0], self.S[3], self.S[1], self.S[4], scale=0.5, units='xy',color='y')
        plt.title("Time: " + str(np.round(self.steps*self.dt,2)) + " sec   Throttle: " + str(np.round(self.throttle,2)) + "\nWind strength: " + str(np.round(np.sqrt(np.dot(self.wind,self.wind)),2)) + "   Wind direction: " + str(np.round(np.rad2deg(np.arctan2(self.wind[1],self.wind[0]))%360,2)) +" deg")


class RodGravityBalance:
    def __init__(self):
        # Engine = Velret(x0,v0,F,m,dt)
        self.action_space = Box(low=0, high=1, shape=(2,))
        self.dt = 1e-1
        self.eta = [0, 0, 0]
        self.k = [0, 0, 0]
        self.edge = 1
        self.max_steps = int(100/self.dt)
        self.rod_length = 0.6
        self.rod_mass = 10
        self.rod_Icm = 1/12 * self.rod_mass * self.rod_length**2
        # Position
        self.A = np.zeros(shape=(9,9))
        for i in range(3):
            AA = np.zeros(shape=(3,3))
            AA[0,0] = AA[1,1] = 1
            AA[0,1] = AA[1,2] = self.dt
            AA[2,0] = -self.k[i]
            AA[2,1] = -self.eta[i]
            self.A[3*i:3*i+3, 3*i:3*i+3] = AA
        # Throttle control:
        self.throttle_max = 4
        self.throttle_span = np.arange(0, self.throttle_max+1)
        self.throttle_values = np.concatenate(([0], self.throttle_span))
        self.throttle_F_factor = 1.5
        # Wind:
        self.wind_co = 0.01
        # Gravity:
        self.grav = self.rod_mass * np.array([0, -0.1])
        # for rendering
        X = np.linspace(-self.edge, self.edge, 25)
        Y = np.linspace(0, 2*self.edge, 25)
        self.x, self.y = np.meshgrid(X,Y)
        self.z = self.y
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def reset(self):
        self.reward = 0
        self.S = np.zeros(9)
        self.F = np.zeros(9)
        self.S[0], self.S[3], self.S[6] = 0.01 * (2*np.random.rand(3)-1)
        self.S[0] += 0.5
        self.S[3] += 0.5
        self.steps = 0
        self.throttle = np.zeros(shape=2)
        self.engines_force = np.zeros(shape=2)
        self.wind = 0*self.wind_co*(2*np.random.rand(2)-1)
        self.wind[1] *= 0.5
        self.wind_change_co = self.wind_co/10
        return np.concatenate((self.S, self.throttle))

    @jit
    def step(self, action):
        self.wind += self.wind_change_co * (2 * np.random.rand(2) - 1)
        # for i, a in enumerate(action):
        #     self.throttle[i] += a
        #     self.throttle[i] = self.throttle_values[np.digitize(self.throttle[i], self.throttle_span)]
        self.throttle = action
        thrust = self.throttle_F_factor * self.throttle
        R = self.rot_mat(self.S[6])
        # --- engines thrust always in y in rod frame of reference (FoR):
        engines_rod_FoR = np.array([np.zeros(2), thrust])
        self.engines_force = np.sum(np.dot(R, engines_rod_FoR),axis=1)
        F_tmp = 1/self.rod_mass * (self.engines_force + self.wind + self.grav)
        self.F[2] = F_tmp[0]
        self.F[5] = F_tmp[1]
        self.F[8] = 1/self.rod_Icm * 0.5 * self.rod_length * (thrust[0] - thrust[1])
        self.S = np.dot(self.A, self.S) + self.F
        # self.reward = 1 - 0.1*np.abs(self.S[7]).item() - 0.1*np.mean(self.throttle).item()
        self.reward = 1
        self.steps += 1
        self.done = False
        if np.abs(self.S[0]) > self.edge or self.S[3] < 0 or self.S[0] < 0 or self.S[3] > 2*self.edge or self.steps >= self.max_steps:
            self.done = True
        self.info = ""
        return np.concatenate((self.S, self.throttle)), self.reward, self.done, self.info

    def render(self):
        drawnow(self.make_fig)

    def make_fig(self):
        scale_fac = 8
        # plt.imshow(self.z, interpolation='bilinear', cmap=cm.RdYlGn, extent=[-self.edge, self.edge, -self.edge, self.edge])
        # plt.imshow(self.z, interpolation='bilinear', cmap=cm.jet, extent=[-self.edge, self.edge, -self.edge, self.edge])
        # plt.imshow(self.z, cmap=cm.RdYlGn_r, extent=[-self.edge, self.edge, -self.edge, self.edge])
        # plt.plot(self.S[0], self.S[3],'ko', markersize=8)

        center = np.array([self.S[0], self.S[3]/2])
        dim = np.array([self.rod_length, 0.05])
        rect = mpatches.Rectangle((center - dim / 2), dim[0], dim[1])
        t = mpl.transforms.Affine2D().rotate_around(*center, self.S[6]) + self.ax.transData
        rect.set_transform(t)
        plt.gca().add_patch(rect)
        self.ax.set_aspect('equal', 'box')

        plt.quiver(0, 0, self.wind[0] + self.grav[0], self.wind[1] + self.grav[1], scale=scale_fac, units='xy', color='w')
        plt.quiver(self.S[0], self.S[3], self.F[2], self.F[5], scale=scale_fac, units='xy',color='g')
        plt.quiver(self.S[0], self.S[3], -np.sin(self.S[6]), np.cos(self.S[6]), scale=scale_fac, units='xy',color='k')
        plt.quiver(self.S[0], self.S[3], self.S[1], self.S[4], scale=0.5, units='xy',color='y')
        plt.title("Time: " + str(np.round(self.steps*self.dt,2)) + " sec   Throttle: " + str(np.round(self.throttle,2))
                  + "\nWind strength: " + str(np.round(np.sqrt(np.dot(self.wind,self.wind)),2)) + "   Wind direction: "
                  + str(np.round(np.rad2deg(np.arctan2(self.wind[1],self.wind[0]))%360,2)) +" deg")
        plt.xlim(0, self.edge)
        plt.ylim(0, 2*self.edge)


    @jit
    def rot_mat(self, theta):
        R = np.zeros(shape=(2,2))
        c_th = np.cos(theta)
        s_th = np.sin(theta)
        R[0, 0] = R[1, 1] = c_th
        R[0, 1] = -s_th
        R[1, 0] = s_th
        return R


class Painter:
    def __init__(self):
        # Engine = Velret(x0,v0,F,m,dt)
        pass

    def reset(self, img):
        self.img = img
        self.brush_color = np.zeros(4)
        self.position = np.zeros(2)
        self.X_grid, self.Y_grid = np.meshgrid(np.arange(self.img.shape[0]),np.arange(self.img.shape[1]))
        self.brush = self.Brush([self.X_grid,self.Y_grid])
        self.S = np.zeros_like(img)
        return self.S

    @jit
    def step(self, action):
        size = action[0]
        color = action[1:5]
        position = action[5:7]
        patch = self.brush.paint(position, size, color)
        self.S = self.threshold(self.S + patch)


        # return self.S, self.reward, self.done, self.info

    def render(self):
        drawnow(self.make_fig)

    def make_fig(self):
        # plt.plot(state_history[:,0])
        pass

    def threshold(self, S):
        S = np.where(S>255,255,S)
        S = np.where(S < 0, 0, S)
        return S


    class Brush:
        def __init__(self, grid):
            self.size = 1
            self.color = np.zeros(shape=4)
            self.position = np.zeros(2)
            self.texture = np.zeros(shape=(5, 5))
            self.grid = grid
            self.patch = np.zeros(shape=(*grid[0].shape, 3))

        def paint(self, position, size, color):
            self.position = position
            self.size = size
            self.color[:3] = (np.tanh(color[:3])+1)/2
            self.color[4] = np.tanh(color[4])
            for i in range(3):
                self.patch[:,:,i] = self.color[3] * self.color[i] * self.gaussian(position, size, self.grid)
            return self.patch

        def gaussian(self, mu, sigma, grid):
            return np.exp(-0.5 * (np.square(grid[0] - mu[0]) + np.square(grid[1] - mu[1])) / np.square(sigma))


class Darts:
    def __init__(self):
        # Engine = Velret(x0,v0,F,m,dt)
        self.sigma = 5
        self.max_turns = 3
        self.edge = 10
        self.center_org = 5*np.random.randn(2)

    def reset(self):
        self.center = self.center_org + 0.01*np.random.randn(2)
        self.S = np.concatenate((np.zeros(shape=4), self.center))
        self.turns = 0
        self.done = False
        return self.S

    @jit
    def step(self, action):
        d = action - self.center
        self.reward = np.exp(-(np.dot(d, d)/(2*self.sigma**2)))
        self.S[2:4] = action
        self.S[:2] = d
        self.turns += 1
        if self.turns >= self.max_turns:
            self.done = True
        self.info = [str(self.turns) + " darts were thrown."]
        return self.S, self.reward, self.done, self.info

    def render(self):
        drawnow(self.make_fig)

    def make_fig(self):
        plt.plot(self.center[0],self.center[1],'o')
        if self.turns >= 0:
            for i in range(self.turns):
                plt.plot(self.S[0]+self.center[0],self.S[1]+self.center[1],'or')
        plt.xlim(-self.edge, self.edge)
        plt.ylim(-self.edge, self.edge)


class FireCannon:
    def __init__(self, dim=2, mass=1, gravity=1):
        # Engine = Velret(x0,v0,F,m,dt)
        self.dim = dim
        self.wind = Wind(dimensions=self.dim)
        self.wind_res = 10
        self.edge = 2.5
        self.dt = 1e-2
        self.eta = 0.1*np.eye(dim)
        self.eta[-1,-1] = 2
        self.k = 3e3
        self.radius = 1
        self.t = np.linspace(0, 2 * np.pi, 50)
        self.m = mass
        self.P = np.array([[1, self.dt, 0],[0, 1, self.dt]])
        # self.P = np.tile(self.P, (dim, 1))
        self.angle = np.zeros(shape=dim)
        self.gravity = np.zeros(shape=dim)
        self.gravity[-1] = -gravity
        self.throttle = np.zeros(shape=dim)
        self.max_throttle = 5

        # X, Y = np.meshgrid(np.linspace(-20, 20, self.wind_res),
        #                    np.linspace(0, 40, self.wind_res))
        # X = X.reshape(-1,1)
        # Y = Y.reshape(-1,1)
        # self.coordinates = np.concatenate((X, Y), axis=1)

    def action_space(self):
        # action_space = Box(low=, high=, shape=)
        pass

    @jit
    def reset(self):
        self.angle = np.deg2rad(70 + 0.1*np.random.randn(1))
        self.v0 = 10 + 0.1*np.random.randn(1)
        self.S = np.zeros(shape=(self.dim, 2)) #[[x,vx], [y,vy]]
        self.S[0, 1] = self.v0 * np.cos(self.angle)
        self.S[1, 1] = self.v0 * np.sin(self.angle)
        self.S[:, 0] = 1.2 + 0.1 * np.random.randn(2)
        self.F = np.zeros(shape=(self.dim,))
        self.done = False
        return self.S

    @jit
    def step(self, action):
        self.action = action
        self.F = self.force_calc(self.action)
        if self.S[1, 0] < self.radius:
            self.F[1] += -self.k * (self.S[1, 0]-self.radius)
            self.b = self.S[1, 0]  # radius on the y-axis
            self.a = np.sqrt(self.radius**3/self.b)  # radius on the x-axis
        else:
            self.b = self.a = self.radius
        for d in range(self.dim):
            S_star = np.concatenate((self.S[d, :], self.F[d].reshape(1)))
            self.S[d,:] = np.dot(self.P, S_star)
        self.reward = 0


        if self.S[1,0] < 0:
            self.done = True
        self.info = ''
        return self.S, self.reward, self.done, self.info

    @jit
    def force_calc(self, action):
        self.throttle += action
        self.throttle = np.where(self.throttle > self.max_throttle, self.max_throttle, self.throttle)
        self.throttle = np.where(self.throttle < -self.max_throttle, -self.max_throttle, self.throttle)
        action_force = self.throttle
        drag_force = np.dot(self.eta,self.wind(self.S[:,0]) - self.S[:,1])
        self.F = self.gravity + drag_force + action_force
        self.F /= self.m
        return self.F

    def render(self):
        drawnow(self.make_fig)

    def make_fig(self):
        plt.plot(self.S[0,0] + self.a * np.cos(self.t), self.S[1,0] + self.b * np.sin(self.t))
        plt.plot(self.S[0,0], self.S[1,0],'o')
        plt.quiver(self.S[0,0], self.S[1,0],self.F[0],self.F[1], scale=100)
        plt.quiver(self.S[0,0], self.S[1,0],self.S[0,1], self.S[1,1], scale=50, color='y')
        plt.plot(np.array([self.S[0,0]-2.5,self.S[0,0]+2.5]),np.zeros(2), 'r')
        X, Y = np.meshgrid(np.linspace(self.S[0, 0] - self.edge, self.S[0, 0] + self.edge, self.wind_res),
                           np.linspace(self.S[1, 0] - self.edge, self.S[1, 0] + self.edge, self.wind_res))
        X = X.reshape(-1,1)
        Y = Y.reshape(-1,1)
        self.coordinates = np.concatenate((X, Y), axis=1)
        wind_grid = self.wind(self.coordinates)
        plt.quiver(self.coordinates[:,0], self.coordinates[:,1],wind_grid[:,0],wind_grid[:,1], scale=200, color='g')
        plt.title('throttle: ' + str(self.throttle))
        plt.xlim(self.S[0,0]-self.edge,self.S[0,0]+self.edge)
        plt.ylim(self.S[1,0]-self.edge,self.S[1,0]+self.edge)


import imageio
class NavigationGame01:
    def __init__(self):
        # Engine = Velret(x0,v0,F,m,dt)
        fpath = r'C:\Users\SHARONO1\PycharmProjects\untitled02\sat_map.tif'
        self.img = imageio.imread(fpath)
        self.img_ctr = np.array(np.array(self.img.shape) / 2, dtype=np.int64)
        self.img_ctr = self.img_ctr[:-1]
        self.edge = 32

    def action_space(self):
        # action_space = Box(low=, high=, shape=)
        pass

    def reset(self):
        self.S = np.zeros(shape=(2,2))
        self.S[:,0] = self.img_ctr
        return self.S

    @jit
    def step(self, action):
        pass
        # return self.S, self.reward, self.done, self.info

    def render(self):
        drawnow(self.make_fig)

    def make_fig(self):
        plt.imshow(self.square(self.img, self.img_ctr,self.edge))

    def square(self, img, ctr, edge):
        p = np.array([ctr[0]-edge,ctr[0]+edge,ctr[1]-edge,ctr[1]+edge])
        return self.img[p[0]:p[1],p[2]:p[3]]



class PlayTogether:
    class Player:
        def __init__(self, x0, v0, m):
            self.x0 = x0.reshape(1, -1)
            self.v0 = v0.reshape(1, -1)
            self.m = m
            self.S0 = np.concatenate((self.x0, self.v0), axis=0)
            self.radar0 = {}
            self.radar0['Players'] = []
            self.radar0['Enemy'] = []
            self.radar0['Home'] = []
            self.dim = len(x0.reshape(-1))
            self.throttle0 = np.zeros(shape=(self.dim,))
            self.brks_mat = np.eye(self.dim)
            self.action_list = []
            self.traj = np.array([])
            self.traj_v = np.array([])
            self.mu = 0.1  # drag coefficient
            self.reward = np.array([0.])
            self.reward_episode = np.array([])

        def reset(self):
            self.action_list = []
            self.S = copy.deepcopy(self.S0)
            self.radar = copy.deepcopy(self.radar0)
            self.throttle = copy.deepcopy(self.throttle0)
            self.brks_mat = np.eye(self.dim)
            self.traj = np.array([])
            self.traj_v = np.array([])
            self.mu = 0.2  # drag coefficient
            self.reward = np.array([0.])
            self.reward_episode = np.array([])

            # return [self.S, self.radar]

    class Enemy:
        def __init__(self, x0, v0=0, m=1):
            self.x0 = x0.reshape(1, -1)
            self.v0 = np.zeros_like(self.x0)
            # self.v0 = v0.reshape(1, -1)
            self.m = m
            self.S0 = np.concatenate((self.x0, self.v0), axis=0)

        def reset(self):
            self.S = copy.deepcopy(self.S0)
            # return self.S

    class Home: #landing site
        def __init__(self, x0, v0=0, m=1):
            self.x0 = x0.reshape(1, -1)
            self.v0 = np.zeros_like(self.x0)
            # self.v0 = v0.reshape(1, -1)
            self.m = m
            self.S0 = np.concatenate((self.x0, self.v0), axis=0)

        def reset(self):
            self.S = copy.deepcopy(self.S0)
            # return self.S


    def __init__(self, kill_dist=0.5, dt=5e-2, max_time=10, possible_actions=4):
        # Engine = Velret(x0,v0,F,m,dt)
        self.dt = dt
        self.P = np.array([[1, self.dt, 0], [0, 1, self.dt]])
        self.players_list = []
        self.enemy_list = []
        self.home_list = []
        self.episode = -1
        self.edge = 80
        self.kill_dist = kill_dist
        self.land_dist = 2.
        self.max_radius = 100
        self.angle_span = 0.7
        self.max_time = max_time # sec
        self.max_steps = int(self.max_time/dt)
        if possible_actions==4:
            self.action_map = {0: "Left", 1: "Right", 2: "Up", 3: "Down"}
        elif possible_actions==5:
            self.action_map = {0: "Left", 1: "Right", 2: "Up", 3: "Down", 4: "Off"}
        elif possible_actions == 6:
            self.action_map = {0: "Left", 1: "Right", 2: "Up", 3: "Down", 4: "Off", 5: "Do nothing"}

        self.action_space = Box(low=-1, high=1, shape=(len(self.action_map),))
        self.max_throttle = 1
        self.max_brks_mat = 1
        self.throttle_force_factor = 1.


    def add_player(self, x0, v0, m):
        self.players_list.append(self.Player(x0, v0, m))

    def add_enemy(self, x0):
        self.enemy_list.append(self.Enemy(x0))

    def add_home(self, x0):
        self.home_list.append(self.Home(x0))

    def dist(self,a, b):
        c = b - a
        return np.sqrt(np.dot(c,c))

    def save_game(self):
        self.enemy_list_saved = copy.deepcopy(self.enemy_list)
        self.home_list_saved = copy.deepcopy(self.home_list)

    def load_game(self):
        self.enemy_list = copy.deepcopy(self.enemy_list_saved)
        self.home_list = copy.deepcopy(self.home_list_saved)

    def reset(self):
        self.episode += 1
        self.steps = 0
        self.wind = Wind(dimensions=2,hidden=[32,32])
        self.reward_episode = np.array([])
        self.reward_episode_kills = np.array([])
        if self.episode == 0:
            self.save_game()
        else:
            self.load_game()

        for player in self.players_list:
            player.reset()
        for enemy in self.enemy_list:
            enemy.reset()
        for home in self.home_list:
            home.reset()
        self.players_in_game = np.ones(len(self.players_list))
        self.players_at_home = np.zeros(len(self.players_list))
        self.dim = len(self.players_list[-1].S[0, :])
        self.S = [None] * len(self.players_list)
        self.done = False
        self.home_mode = False
        self.info = ''
        self.S, _, _, _ = self.step()
        for player in self.players_list:
            player.reward_episode = np.array([])
        self.reward_episode = np.array([])
        self.reward_episode_kills = np.array([])
        return self.S

    @jit
    def step(self):
        self.S = []
        self.reward = np.array([0.])
        self.reward_kills = np.array([0.])
        for j, player in enumerate(self.players_list):
            player.reward = np.array([0.])
            if self.steps > 0:
                force = self.action2force(player).reshape(1,-1)
            else:
                force = np.zeros(shape=(1,self.dim))
            S_star = np.concatenate((player.S, force), axis=0)
            player.S = np.dot(self.P, S_star)
            # player.traj_v = np.concatenate((player.traj_v, ))
            player.traj = np.concatenate((player.traj, player.S[0,:],np.array([np.sqrt(np.dot(player.S[1,:],player.S[1,:]))])), axis=0)
            enemy_idx2delete = []
            # check how close enemy is to player[i]
            for i, enemy in enumerate(self.enemy_list):
                d = self.dist(enemy.S[0, :].reshape(-1), player.S[0, :].reshape(-1))
                if d < self.kill_dist:
                    enemy_idx2delete.append(i)
                    player.reward += 2*np.power((self.steps*self.dt+1), -0.2)# + 0.1*np.random.randn(1)[0]
                    self.reward += 1
                    self.reward_kills += 1
            # delete enemy that was caught
            if len(self.enemy_list)>0:
                enemy_idx2delete = np.sort(enemy_idx2delete)[::-1]
                for i in enemy_idx2delete:
                    del self.enemy_list[i]
            self.radar(player)
            # player.reward += 1*len(player.radar['Enemy'])
            if self.home_mode is True:
                player.reward += 1 * len(player.radar['Home'])
            if np.abs(player.S[0,0])<self.edge and np.abs(player.S[0,1])<self.edge:
                self.players_in_game[j] = 1
            else:
                self.players_in_game[j] = 0
            if self.home_mode is True:
                for i, home in enumerate(self.home_list):
                    d = self.dist(home.S[0, :].reshape(-1), player.S[0, :].reshape(-1))
                    if d < self.land_dist and self.players_at_home[j] == 0:
                        print("Player #", j, " has landed!")
                        player.reward += 10
                        self.reward += 1
                        self.players_at_home[j] = 1
                        player.mu = 10
            player.reward_episode = np.concatenate((player.reward_episode, player.reward))
        self.steps += 1
        # self.wind.change(amount=0.1)
        # self.reward += -1*self.dt
        # self.reward += 1e-2*self.dt
        if self.steps>self.max_steps or self.players_in_game.sum() == 0:
        # if self.steps > self.max_steps:
            self.done = True
        self.reward_episode = np.concatenate((self.reward_episode, self.reward))
        self.reward_episode_kills = np.concatenate((self.reward_episode_kills, self.reward_kills))
        # if len(self.enemy_list) == 0 or self.steps>self.max_steps:
        if len(self.enemy_list) == 0:
            self.home_mode = True
            if self.players_at_home.sum() == len(self.players_list):
                self.done = True
                print("Win!!!")
        return self.S, self.reward, self.done, self.info

    @jit
    def action2force(self, player):
        action = player.action_list[-1]
        player.throttle = self.action2throttle(action, player.throttle)
        force = self.throttle_force_factor * player.throttle
        # print(self.wind(player.S[0,:]), player.S[1,:])
        # drag = player.mu * np.dot(5e-2*self.wind(player.S[0,:]) - player.S[1,:],player.brks_mat)
        drag = player.mu * np.dot(-player.S[1,:],player.brks_mat)
        force += drag
        return force

    @jit
    def action2throttle(self, action, throttle):
        if self.action_map[action] is "Do nothing":
            pass
        elif self.action_map[action] is "Left":
            throttle[0] -= 1
            throttle[0] = np.clip(throttle[0], -self.max_throttle, self.max_throttle)
        elif self.action_map[action] is "Right":
            throttle[0] += 1
            throttle[0] = np.clip(throttle[0], -self.max_throttle, self.max_throttle)
        elif self.action_map[action] is "Up":
            throttle[1] += 1
            throttle[1] = np.clip(throttle[1], -self.max_throttle, self.max_throttle)
        elif self.action_map[action] is "Down":
            throttle[1] -= 1
            throttle[1] = np.clip(throttle[1], -self.max_throttle, self.max_throttle)
        elif self.action_map[action] is "Off":
            throttle *= 0
        return throttle

    # @jit
    # def radar_sensor(self, self_angle, other_angles, other_dist, angle_sens=0.2, dist_sens=1.25):
    #     return np.exp(-other_dist / dist_sens + 1 / angle_sens * np.cos(-other_angles + self_angle))

    @jit
    def radar_sensor(self, rho, angle, other_angles, angle_span=0.25, max_radius=1):
        return 1*(rho < max_radius) * 1 * (np.cos(-other_angles + angle) + angle_span > 1)

    @jit
    def radar(self, self_player):
        # self.dim = len(self_player.S[0, :])
        v_angle = np.arctan2(self_player.S[1, 1], self_player.S[1, 0])
        players_signal = np.zeros(shape=(len(self.players_list),self.dim+1))
        enemy_signal = np.zeros(shape=(len(self.enemy_list),self.dim+1))
        home_signal = np.zeros(shape=(len(self.home_list),self.dim+1))
        for i, player in enumerate(self.players_list):
            d = player.S[0, :] - self_player.S[0, :]
            players_signal[i, :-1] = d
            rho, phi = cart2pol(d[0], d[1])
            if rho > 0:
                players_signal[i, -1] = self.radar_sensor(rho, v_angle, phi, angle_span=self.angle_span, max_radius=self.max_radius)
            else:
                players_signal[i, -1] = 0

        for i, enemy in enumerate(self.enemy_list):
            d = enemy.S[0, :] - self_player.S[0, :]
            enemy_signal[i, :-1] = d
            rho, phi = cart2pol(d[0], d[1])
            enemy_signal[i, -1] = self.radar_sensor(rho, v_angle, phi, angle_span=self.angle_span, max_radius=self.max_radius)

        for i, home in enumerate(self.home_list):
            d = home.S[0, :] - self_player.S[0, :]
            home_signal[i, :-1] = d
            rho, phi = cart2pol(d[0], d[1])
            home_signal[i, -1] = self.radar_sensor(rho, v_angle, phi, angle_span=self.angle_span, max_radius=self.max_radius)

        self_player.players_idx = np.argwhere(players_signal[:, -1] == 1).reshape(-1)
        self_player.enemy_idx = np.argwhere(enemy_signal[:, -1] == 1).reshape(-1)
        self_player.home_idx = np.argwhere(home_signal[:, -1] == 1).reshape(-1)
        players_signal = players_signal[self_player.players_idx,:-1]
        enemy_signal = enemy_signal[self_player.enemy_idx,:-1]
        home_signal = home_signal[self_player.home_idx, :-1]
        self.store_radar(self_player, players_signal, enemy_signal, home_signal)
        # return players_signal, enemy_signal, home_signal

    @jit
    def store_radar(self, self_player, players_signal, enemy_signal, home_signal):
        self_player.radar['Players'] = players_signal
        self_player.radar['Enemy'] = enemy_signal
        self_player.radar['Home'] = home_signal

    def render(self):
        drawnow(self.make_fig)

    def make_fig(self):
        for i, home in enumerate(self.home_list):
            plt.scatter(home.S[0,1], home.S[0,0],color='green')
        for i, player in enumerate(self.players_list):
            plt.scatter(player.S[0,1],player.S[0,0], color='blue')
            plt.quiver(player.S[0,1], player.S[0,0], player.S[1,1], player.S[1,0], scale=60, color='yellow')
            # plt.quiver(player.S[0,1], player.S[0,0], player.fields[0], player.fields[1],scale=150, color='black')
            # plt.quiver(player.S[0,1], player.S[0,0], player.throttle[0], player.throttle[1],scale=20, color='magenta')
            # ??? force: plt.quiver(player.S[0,1], player.S[0,0], player.S[1,1], player.S[1,0], scale=2, color='blue')
        for i, enemy in enumerate(self.enemy_list):
            if i in self.players_list[-1].enemy_idx:
                plt.scatter(enemy.S[0,1], enemy.S[0,0],color='red')
        for i, enemy in enumerate(self.enemy_list):
            plt.scatter(enemy.S[0,1], enemy.S[0,0],color='red', alpha=0.1)
        plt.title("Time: " + str(self.steps*self.dt) + " sec    Total Score: " + str(np.sum(self.reward_episode))
                  + "\n   Throttle: " + str(self.players_list[-1].throttle)
                  + "\n   number of Enemies on radar: " + str(len(self.players_list[-1].radar['Enemy'])))
        plt.xlim(-self.edge,self.edge)
        plt.ylim(-self.edge,self.edge)


#-------------------------------------------------------------------
#-------------------------------------------------------------------

class Drone2Dsim:
    # -=-=-=-=-=-=-=-=-=-=-=-=-=- Classes -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    class Obj:
        @jit
        def __init__(self, type, team, state0, dt, m, I, eta):
            self.P = np.array([[1, dt, 0], [0, 1, dt]])
            self.type = type
            self.team = team
            self.state0 = state0.reshape(2, -1)
            self.state = copy.deepcopy(self.state0)
            self.dof = state0.shape[1] # degrees of freedom
            self.eta = np.diag(eta) # drag coef matrix
            # self.mass = effective mass = m + I :
            I = np.array([I])
            m = np.tile(np.array([m]), (self.dof - len(I),))
            self.mass = np.concatenate((m.reshape(-1), I.reshape(-1))).reshape(1, -1)
            self.done = False
            self.reward_episode = np.array([])
            self.visible = False
            self.max_dist = 10
            # self.angle_span = 1 # 1/2 -> 120°, 1 -> 180°, 2 -> 360°
            self.angle_span = 180 # degrees

        def __repr__(self):
            return self.team + " " + self.type + " in " + str(self.state[0,:2]) + " heading " + str(np.round(np.rad2deg(self.state[0,2]),1)) + "°."

        @jit
        def reset(self):
            self.done = False
            self.reward_episode = np.array([])
            self.state = copy.deepcopy(self.state0)
            self.radar_signal = np.array([])


        @jit
        def rotation(self, theta):
            R = np.zeros(shape=(2, 2))
            c_th = np.cos(theta)
            s_th = np.sin(theta)
            R[0, 0] = R[1, 1] = c_th
            R[0, 1] = -s_th
            R[1, 0] = s_th
            return R

        @jit
        def force2acceleration(self, force):
            force += self.drag()
            return force.reshape(1, -1) / self.mass

        @jit
        def drag(self):
            theta = self.state[0, 2]
            eta = self.eta
            eta[:2, :2] = np.dot(self.rotation(theta), eta[:2, :2])
            return np.dot(eta, -self.state[1, :])

        @jit
        def step_calc(self, force):
            self.acceleration = self.force2acceleration(force)
            s_star = np.concatenate((self.state, self.acceleration), axis=0)
            self.state_ = np.dot(self.P, s_star)


        @jit
        def step_execute(self):
            self.state = copy.deepcopy(self.state_)
            self.state[0, 2] = self.state[0, 2] % (2 * np.pi)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=- Drone child class -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    class Drone(Obj):
        @jit
        def __init__(self, type, team, state0, dt, m, I, eta):
            super(__class__, self).__init__(type, team, state0, dt, m, I, eta)
            self.throttle = np.zeros((self.dof,))
            self.throttle2force_fac = 1e-1 * np.array([1, 1, 1e-2])
            self.max_throttle = 1
            self.killMode = False
            self.action_map = {0: "Forward", 1: "Backward", 2: "CW", 3: "CCW", 4: "Left", 5: "Right", 6: "KillOff", 7: "KillOn" , 8: "Off", 9: "Do nothing"}
            self.action = 0

        @jit
        def reset(self):
            super(__class__, self).reset()
            self.throttle = np.zeros((self.dof,))


        @jit
        def action2throttle(self, action):
            if self.action_map[action] is "Left":
                self.throttle[1] += 1
            elif self.action_map[action] is "Right":
                self.throttle[1] -= 1
            elif self.action_map[action] is "Forward":
                self.throttle[0] += 1
            elif self.action_map[action] is "Backward":
                self.throttle[0] -= 1
            elif self.action_map[action] is "CW":
                self.throttle[2] -= 1
            elif self.action_map[action] is "CCW":
                self.throttle[2] += 1
            elif self.action_map[action] is "KillOff":
                self.killMode = False
            elif self.action_map[action] is "KillOn":
                self.killMode = True
            elif self.action_map[action] is "Off":
                self.throttle *= 0
            elif self.action_map[action] is "Do nothing":
                pass
            self.throttle = np.clip(self.throttle, -self.max_throttle, self.max_throttle)
            return self.throttle

        @jit
        def throttle2force(self, throttle=[]):
            if len(throttle) == 0:
                throttle = self.throttle
            force = self.throttle2force_fac * throttle
            theta = self.state[0, 2]
            force[:2] = np.dot(self.rotation(theta), force[:2])
            return force

        @jit
        def action2force(self, action):
            throttle = self.action2throttle(action)
            force = self.throttle2force(throttle)
            return force

        @jit
        def step_calc(self, action):
            super(__class__, self).step_calc(self.action2force(action))

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Simulation -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __init__(self, dt):
        '''
        To access any object use the syntax:
            env.obj_list['Category']['Team color']
            i.e.:
            env.obj_list['Drones']['Blue'].brain = Agent(...)
        '''
        # Engine = Velret(x0,v0,F,m,dt)
        self.action_space = Box(low=0, high=6, shape=(1,))
        self.dt = dt
        self.kill_dist = 1
        self.max_radius = 10
        self.team_names = []
        self.obj_list = {}
        self.obj_list['Drones'] = {}
        self.obj_list['Objects'] = {}
        self.obj_list['LandSites'] = {}
        # self.obj_colors = {'Drones': 'yellow', 'Objects': 'magenta', 'LandSites': 'green'}
        self.obj_shape = {'Drones': 'o', 'Objects': '*', 'LandSites': 'X'}
        self.episode = 0
        self.type_one_hot_map = {}
        self.team_name_one_hot_map = {}
        self.done = False
        self.info = ''
        self.kills_in_a_row = 0
        self.obj_radius0 = 1.5
        self.obj_radius = copy.deepcopy(self.obj_radius0)

    def add_team(self, name):
        self.team_names.append(name)
        for type in self.obj_list.keys():
            self.obj_list[type][name] = []

    def add_drone(self, team, state0, m, I, eta):
        self.obj_list['Drones'][team].append(self.Drone("Drones",team,state0, self.dt, m, I, eta))

    def add_object(self, team, state0, m, I, eta):
        self.obj_list['Objects'][team].append(self.Obj("Objects",team, state0, self.dt, m, I, eta))

    def add_landSite(self, team, state0, m, I, eta):
        self.obj_list['LandSites'][team].append(self.Obj("LandSites", team, state0, self.dt, m, I, eta))

    @jit
    def reset(self):
        if self.episode == 0:
            for i, type in enumerate(self.obj_list.keys()):
                self.type_one_hot_map[type] = one_hot(i, len(self.obj_list.keys()))
            for j, team_name in enumerate(self.obj_list[type].keys()):
                self.team_name_one_hot_map[team_name] = one_hot(j, len(self.obj_list[type].keys()))
        for type in self.obj_list.values():
            for team in type.values():
                for obj in team:
                    obj.type1hot = self.type_one_hot_map[obj.type]
                    obj.team1hot = self.team_name_one_hot_map[obj.team]
                    obj.reset()

        self.steps = 0
        self.kills = 0
        self.done = False
        self.all_invisible()
        self.all_observe(type='Drones')
        self.n_enemy = len(self.obj_list['Objects']['Red'])

        t = np.linspace(0, 2 * np.pi, self.n_enemy + 1)
        t = t[:-1]
        if self.kills_in_a_row > 5 and self.obj_radius < self.max_radius-1:
            self.obj_radius += 0.1
        elif self.kills_in_a_row == 0 and self.obj_radius > self.obj_radius0:
            self.obj_radius -= 0.1

        rand_phase = 2*np.pi*np.random.rand(1)
        for tt, obj in enumerate(self.obj_list['Objects']['Red']):
            obj.state = np.random.randn(2, 3)
            obj.state[0, 0] += self.obj_radius * np.cos(t[tt] + rand_phase)
            obj.state[0, 1] += self.obj_radius * np.sin(t[tt] + rand_phase)
            obj.state[1, :] = 0
            obj.state[0, 2] = 2 * np.pi * np.random.rand(1)


    @jit
    def in_radar_range(self, dist, angle, max_dist, angle_span):
        # return (0.5*(np.cos(angle)+1+angle_span) >= 1) and (dist <= max_dist)
        return (abs(angle) <= np.deg2rad(angle_span/2)) * (dist < max_dist)

    @jit
    def radar(self, drone):
        # radar_signal shape: position2D + velocity2D + objectypes + teams = 4+3+2 = 9
        radar_state = np.zeros(shape=(6,))
        drone.radar_signal = np.concatenate((radar_state, drone.type1hot, drone.team1hot)).reshape(1,-1)
        drone.reward = 0
        for type in self.obj_list.values():
            for team in type.values():
                for obj in team:
                    if obj.done is False:
                        theta = drone.state[0, 2]
                        d = obj.state[0, :2] - drone.state[0,:2]
                        rho, phi = cart2pol(d[0], d[1])
                        if 0 < rho < self.kill_dist and obj.type is 'Objects':
                            drone.reward += 1
                            obj.done = True
                            self.kills += 1
                            obj.visible = False
                        else:
                            relative_angle = np.abs(phi-theta)
                            if rho > 0 and self.in_radar_range(rho, relative_angle, drone.max_dist, drone.angle_span):
                                obj.visible = True
                                # Relative velocity:
                                vd = obj.state[1, :2] - drone.state[1, :2]
                                v_rho, v_phi = cart2pol(vd[0], vd[1])
                                # Radar state: relative [distance, angle, velocity, velocity angle]
                                radar_state = np.concatenate(([rho], angle_split(relative_angle), [v_rho], angle_split(v_phi)), axis=0)
                                # radar_state = np.array([rho, *angle_split(relative_angle), v_rho, *angle_split(v_phi)])
                                radar_signal = np.concatenate((radar_state, obj.type1hot, obj.team1hot))
                                # if len(drone.radar_signal>0):
                                drone.radar_signal = np.concatenate((drone.radar_signal, radar_signal.reshape(1, -1)), axis=0)
                                # else:
                                #     drone.radar_signal = radar_signal.reshape(1, -1)
        drone.reward_episode = np.append(drone.reward_episode, drone.reward)


    @jit
    def all_invisible(self):
        for type in self.obj_list.values():
            for team in type.values():
                for obj in team:
                    obj.visible = False

    @jit
    def all_observe(self, type='Drones'):
        for team in self.obj_list[type].values():
            for drone in team:
                if drone.done is False:
                    self.radar(drone)
                    xy = drone.state[0, :2]
                    theta_split = angle_split(drone.state[0, 2])
                    v = drone.state[1,:]
                    # v_abs, phi = cart2pol(v[0], v[1])
                    # v_pol = np.array([v_abs, phi-theta])
                    drone.obs = np.concatenate((xy, theta_split, v, drone.throttle)).reshape(1,-1)

    @jit
    def all_step(self, type='Drones'):
        for team in self.obj_list[type].values():
            for drone in team:
                if drone.done is False:
                    # drone.action = np.random.randint(6)
                    drone.step_calc(drone.action)

        self.done = True
        for team in self.obj_list[type].values():
            for drone in team:
                if drone.done is False:
                    drone.step_execute()
                    self.done = False
                    if np.abs(np.sqrt(np.dot(drone.state[0,:2],drone.state[0,:2]))) > self.max_radius:
                        drone.done = True

        self.S = []
        # for type in self.obj_list.values():
        #     for team in type.values():
        #         for obj in team:
        #             self.S.append([obj.state[0,:2], obj.type, obj.team, obj.done])

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Simulation Step =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    @jit
    def step(self):
        self.kills = 0
        self.all_step(type='Drones') # <-- insert drone.action before this line/function.
        self.all_invisible()
        self.all_observe(type='Drones')
        self.reward = self.kills
        self.steps += 1
        if self.n_enemy - self.kills == 0:
            self.done = True
            self.kills_in_a_row += 1
            print("Killed them all!")
        if self.done:
            self.episode += 1
            if self.n_enemy - self.kills > 0:
                self.kills_in_a_row = 0
        return self.S, self.reward, self.done, self.info

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= Render -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def render(self):
        drawnow(self.make_fig)

    def make_fig(self):
        throttle = []
        for type in self.obj_list.values():
            for team in type.values():
                for obj in team:
                    if obj.done is False:
                        x = obj.state[0,0]
                        y = obj.state[0,1]
                        theta = obj.state[0,2]
                        plt.scatter(x, y, s=100, marker=self.obj_shape[obj.type], c=obj.team.lower(), edgecolors=None, linewidth='0', alpha=0.7*obj.visible+0.3)
                        if "Drones" in obj.type:
                            throttle.append(obj.throttle)
                            # print(throttle)
                            plt.quiver(x, y, np.cos(theta), np.sin(theta))
                            plt.quiver(x, y, obj.state[1,0], obj.state[1,1], scale=10, color='y')
                            plt.quiver(x, y, obj.acceleration[0,0], obj.acceleration[0,1], scale=10, color='g')

        tt = np.linspace(0, 2*np.pi, 50)
        plt.plot(self.max_radius*np.cos(tt), self.max_radius*np.sin(tt),'--r')
        plt.title("Time: " + str(np.round(self.steps*self.dt,2)) + " sec \n throttle: " + str(throttle))
        plt.xlim(-self.max_radius, self.max_radius)
        plt.ylim(-self.max_radius, self.max_radius)
