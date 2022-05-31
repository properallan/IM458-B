import numpy as np
from sympy import Symbol, sin
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
from wave_equation import WaveEquation1D


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))
    print(type(cfg.optimizer))
    print(cfg.optimizer)

    # make list of nodes to unroll graph on
    we = WaveEquation1D(c=1.0)
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = we.make_nodes() + [wave_net.make_node(name="wave_network", jit=cfg.jit)]

    # add constraints to solver
    # make geometry
    x, t_symbol = Symbol("x"), Symbol("t")
    L = float(np.pi)
    geo = Line1D(0, L)
    time_range = {t_symbol: (0, 2 * L)}

    # make domain
    domain = Domain()

    # initial condition
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": sin(x), "u__t": sin(x)},
        batch_size=cfg.batch_size.IC,
        bounds={x: (0, L)},
        lambda_weighting={"u": 1.0, "u__t": 1.0},
        param_ranges={t_symbol: 0.0},
    )
    domain.add_constraint(IC, "IC")

    # boundary condition
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0},
        batch_size=cfg.batch_size.BC,
        param_ranges=time_range,
    )
    domain.add_constraint(BC, "BC")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"wave_equation": 0},
        batch_size=cfg.batch_size.interior,
        bounds={x: (0, L)},
        param_ranges=time_range,
    )
    domain.add_constraint(interior, "interior")

    # add validation data
    deltaT = 0.01
    deltaX = 0.01
    x = np.arange(0, L, deltaX)
    t = np.arange(0, 2 * L, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    u = np.sin(X) * (np.cos(T) + np.sin(T))
    invar_numpy = {"x": X, "t": T}
    outvar_numpy = {"u": u}
    plotter = ValidatorPlotter()
    validator = PointwiseValidator(invar_numpy, outvar_numpy, nodes, batch_size=32, plotter=plotter)
    domain.add_validator(validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
