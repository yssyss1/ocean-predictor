"""
Source from https://github.com/queezz/numerical-tests/blob/master/Waves.ipynb
"""
import numpy as np


class Wave2d:
    def __init__(self, height, width, T, nx, ny, nt, c):
        """
        :param T: final time
        :param nx: grid points in x direction
        :param ny: grid points in y direction
        :param nt: number of time steps
        :param c: wave speed
        """
        self.x = np.linspace(-0.5 * width, 0.5 * width, nx)
        self.y = np.linspace(-0.5 * height, 0.5 * height, ny)
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.t = np.linspace(0, T, nt + 1)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dt = self.t[1] - self.t[0]
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        # Gamma_x squared
        self.gx2 = c * self.dt / self.dx
        # Gamma_y squared
        self.gy2 = c * self.dt / self.dy
        # 2*(1-gamma_x^2-gamma_y^2)
        self.gamma = 2 * (1 - self.gx2 - self.gy2)

    def solve(self, ffun, gfun):
        f = ffun(self.xx, self.yy)
        g = gfun(self.xx, self.yy)
        u = np.zeros((self.ny, self.nx, self.nt + 1))
        # Set initial condition
        u[:, :, 0] = f
        """ Compute first time step """
        u[:, :, 1] = 0.5 * self.gamma * f + self.dt * g
        u[1:-1, 1:-1, 1] += 0.5 * self.gx2 * (f[1:-1, 2:] + f[1:-1, :-2])
        u[1:-1, 1:-1, 1] += 0.5 * self.gy2 * (f[:-2, 1:-1] + f[2:, 1:-1])
        for k in range(1, self.nt):
            # Every point contains these terms
            u[:, :, k + 1] = self.gamma * u[:, :, k] - u[:, :, k - 1]
            # Interior points
            u[1:-1, 1:-1, k + 1] += self.gx2 * (u[1:-1, 2:, k] + u[1:-1, :-2, k]) + \
                                    self.gy2 * (u[2:, 1:-1, k] + u[:-2, 1:-1, k])
            # Top boundary
            u[0, 1:-1, k + 1] += 2 * self.gy2 * u[1, 1:-1, k] + \
                                 self.gx2 * (u[0, 2:, k] + u[0, :-2, k])
            # Right boundary
            u[1:-1, -1, k + 1] += 2 * self.gx2 * u[1:-1, -2, k] + \
                                  self.gy2 * (u[2:, -1, k] + u[:-2, -1, k])
            # Bottom boundary
            u[-1, 1:-1, k + 1] += 2 * self.gy2 * u[-2, 1:-1, k] + \
                                  self.gx2 * (u[-1, 2:, k] + u[-1, :-2, k])
            # Left boundary
            u[1:-1, 0, k + 1] += 2 * self.gx2 * u[1:-1, 1, k] + \
                                 self.gy2 * (u[2:, 0, k] + u[:-2, 0, k])
            # Top right corner
            u[0, -1, k + 1] += 2 * self.gx2 * u[0, -2, k] + \
                               2 * self.gy2 * u[1, -1, k]
            # Bottom right corner
            u[-1, -1, k + 1] += 2 * self.gx2 * u[-1, -2, k] + \
                                2 * self.gy2 * u[-2, -1, k]
            # Bottom left corner
            u[-1, 0, k + 1] += 2 * self.gx2 * u[-1, 1, k] + \
                               2 * self.gy2 * u[-2, 0, k]
            # Top left corner
            u[0, 0, k + 1] += 2 * self.gx2 * u[0, 1, k] + \
                              2 * self.gy2 * u[1, 0, k]
        return u