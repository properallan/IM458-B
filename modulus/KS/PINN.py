import numpy as np
from sympy import Symbol, sin, cos
from modulus.tensorboard_utils.plotter import ValidatorPlotter

import modulus
from modulus.hydra import to_yaml, instantiate_arch
from modulus.hydra.config import ModulusConfig
from modulus.continuous.solvers.solver import Solver
from modulus.continuous.domain.domain import Domain
from modulus.geometry.csg.csg_1d import Line1D
from modulus.continuous.constraints.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from modulus.continuous.validator.validator import PointwiseValidator
from modulus.key import Key
from modulus.node import Node
from KSequation import KSequation

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))
    print(type(cfg.optimizer))
    print(cfg.optimizer)

    L = float(32*np.pi)
    
    # make list of nodes to unroll graph on
    ks = KSequation(ni=1.0)
    ks_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("u")],
        periodicity={"x": (0, L)},
        cfg=cfg.arch.fully_connected,
    )
    
    nodes = ks.make_nodes() + [ks_net.make_node(name="ks_network", jit=cfg.jit)]
    #nodes = [ks_net.make_node(name="ks_network", jit=cfg.jit)]

    # add constraints to solver
    # make geometry
    x, t_symbol = Symbol("x"), Symbol("t")
    geo = Line1D(0, L)
    tend = 100
    time_range = {t_symbol: (0, tend)}

    # make domain
    domain = Domain()

    # initial condition
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": cos(x/16)*(1+sin(x/16))},
        batch_size=cfg.batch_size.IC,
        bounds={x: (0, L)},
        lambda_weighting={"u": 1.0},
        param_ranges={t_symbol: 0.0},
    )
    domain.add_constraint(IC, "IC")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ks_equation": 0},
        batch_size=cfg.batch_size.interior,
        bounds={x: (0, L)},
        param_ranges=time_range,
    )
    domain.add_constraint(interior, "interior")

    deltaT = 0.2
    deltaX = L/512
    x = np.arange(0, L, deltaX)
    t = np.arange(0, tend, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
   
    # numeric solution
    u0 = np.cos(x/16)*(1+np.sin(x/16))
    uu, tt, xx = ks_sol(x, u0, tmax=tend, h=deltaT)
    plot(uu, tt, xx)

    uu = np.expand_dims(uu.flatten(), axis=-1)
    
    invar_numpy = {"x": X, "t": T}
    outvar_numpy = {"u": uu}
    plotter = ValidatorPlotter()
    validator = PointwiseValidator(invar_numpy, outvar_numpy, nodes, batch_size=512, plotter=plotter)
    domain.add_validator(validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


#validator = PointwiseValidator(invar_numpy, outvar_numpy, nodes, batch_size=128)
#domain.add_validator(validator)

#slv = Solver(cfg, domain)
#slv.solve()

def plot(uu, tt, x):
    fig = plt.figure()
    ax = fig.gca()
    tt, x = np.meshgrid(tt, x)
    #surf = ax.contourf(tt, x, uu.transpose(), cmap=cm.coolwarm, linewidth=0)
    surf = ax.pcolormesh(tt, x, uu.transpose())
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

if __name__ == "__main__":
    #run()
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    from ks_solution import generate_solution as ks_sol
    
    L = 10
    #x = np.arange(-L, L, deltaX)
    x = np.linspace(0, 32*np.pi, 512)
    #u0 = -np.sin(np.pi*x/L)
    u0 = np.cos(x/16)*(1+np.sin(x/16))
    uu, tt, x = ks_sol(x, u0, tmax=100, h=0.2)
    
    run()

    
    #plot(uu,tt,x)
