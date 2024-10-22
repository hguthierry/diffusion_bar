import torch
import numpy as np
from sympy import Symbol, Eq, Function, Number

import modulus
from modulus.sym.hydra import ModulusConfig, instantiate_arch
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.geometry import Parameterization
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.eq.pde import PDE

bar_lenght_1 = Symbol('bar_lenght_1')
bar_lenght_2 = Symbol('bar_lenght_2')
bar_lenght_3 = Symbol('bar_lenght_3')
bar_3_origin = bar_lenght_1+bar_lenght_2
k1, k2, k3 = Symbol("k1"), Symbol("k2"), Symbol("k3")


params_range = {
    k1: (1, 25),
    k2: (1, 25),
    k3: (1, 25),
    bar_lenght_1: (0.1, 1),
    bar_lenght_2: (0.1, 1),
    bar_lenght_3: (0.1, 1)
}

fixed_params_range = {
    k1: 1.0,
    k2: 5.0,
    k3: 10.0,
    bar_lenght_1: 1.0,
    bar_lenght_2: 1.0,
    bar_lenght_3: 1.0
}

params_validation = {
    "k1": 1.0,
    "k2": 5.0,
    "k3": 10.0,
    "bar_lenght_1": 1.0,
    "bar_lenght_2": 1.0,
    "bar_lenght_3": 1.0
}

Ta = 0
Td = 200
heat_flux = (Td-Ta)/(bar_lenght_1/k1+bar_lenght_2/k2+bar_lenght_3/k3)
Tb = Ta+heat_flux*k1
Tc = Tb+heat_flux*k2
Tb_validation = float(Tb.evalf(subs=params_validation))
Tc_validation = float(Tc.evalf(subs=params_validation))

print(Ta)
print(Tb)
print(Tc)
print(Td)


class Diffusion(PDE):
    name = "Diffusion"

    def __init__(self, T="T", D="D", Q=0, dim=3, time=True):
        # set params
        self.T = T
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # Temperature
        assert type(T) == str, "T needs to be string"
        T = Function(T)(*input_variables)

        # Diffusivity
        if type(D) is str:
            D = Function(D)(*input_variables)
        elif type(D) in [float, int]:
            D = Number(D)

        # Source
        if type(Q) is str:
            Q = Function(Q)(*input_variables)
        elif type(Q) in [float, int]:
            Q = Number(Q)

        # set equations
        self.equations = {}
        self.equations["diffusion_" + self.T] = (
            T.diff(t)
            - (D * T.diff(x)).diff(x)
            - (D * T.diff(y)).diff(y)
            - (D * T.diff(z)).diff(z)
            - Q
        )


class DiffusionInterface(PDE):
    name = "DiffusionInterface"

    def __init__(self, T_1, T_2, D_1, D_2, dim=3, time=True):
        # set params
        self.T_1 = T_1
        self.T_2 = T_2
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x, normal_y, normal_z = (
            Symbol("normal_x"),
            Symbol("normal_y"),
            Symbol("normal_z"),
        )

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # Diffusivity
        if type(D_1) is str:
            D_1 = Function(D_1)(*input_variables)
        elif type(D_1) in [float, int]:
            D_1 = Number(D_1)
        if type(D_2) is str:
            D_2 = Function(D_2)(*input_variables)
        elif type(D_2) in [float, int]:
            D_2 = Number(D_2)

        # variables to match the boundary conditions (example Temperature)
        T_1 = Function(T_1)(*input_variables)
        T_2 = Function(T_2)(*input_variables)

        # set equations
        self.equations = {}
        self.equations["diffusion_interface_dirichlet_" + self.T_1 + "_" + self.T_2] = (
            T_1 - T_2
        )
        flux_1 = D_1 * (
            normal_x * T_1.diff(x) + normal_y * T_1.diff(y) + normal_z * T_1.diff(z)
        )
        flux_2 = D_2 * (
            normal_x * T_2.diff(x) + normal_y * T_2.diff(y) + normal_z * T_2.diff(z)
        )
        self.equations["diffusion_interface_neumann_" + self.T_1 + "_" + self.T_2] = (
            flux_1 - flux_2
        )


@modulus.sym.main(config_path="conf", config_name="config_param")
def run(cfg: ModulusConfig) -> None:
    # make list of nodes to unroll graph on
    torch.set_num_threads(cfg.custom.num_threads)
    print('-'*30)
    number_threads = torch.get_num_threads()
    print(f'o número de threads é {number_threads}')
    print('-'*30)
    diff_u1 = Diffusion(T="u_1", D="k1", dim=1, time=False)
    diff_u2 = Diffusion(T="u_2", D="k2", dim=1, time=False)
    diff_u3 = Diffusion(T="u_3", D="k3", dim=1, time=False)
    diff_in_1 = DiffusionInterface("u_1", "u_2", "k1", "k2", dim=1, time=False)
    diff_in_2 = DiffusionInterface("u_2", "u_3", "k2", "k3", dim=1, time=False)

    if cfg.custom.parameterized:
        pr = Parameterization(params_range)
        input_keys = [
            Key("x"),
            Key("bar_lenght_1"),
            Key("bar_lenght_2"),
            Key("bar_lenght_3"),
            Key("k1"),
            Key("k2"),
            Key("k3")
        ]
    else:
        pr = Parameterization(fixed_params_range)
        input_keys=[Key("x")]
    
    # Geometry
    L1 = Line1D(0, bar_lenght_1, parameterization=pr)
    L2 = Line1D(bar_lenght_1, bar_3_origin, parameterization=pr)
    L3 = Line1D(bar_3_origin, bar_3_origin+bar_lenght_3, parameterization=pr)

    diff_net_u_1 = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("u_1")],
        cfg=cfg.arch.fully_connected,
    )
    diff_net_u_2 = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("u_2")],
        cfg=cfg.arch.fully_connected,
    )
    diff_net_u_3 = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("u_3")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
        diff_u1.make_nodes()
        + diff_u2.make_nodes()
        + diff_u3.make_nodes()
        + diff_in_1.make_nodes()
        + diff_in_2.make_nodes()
        + [diff_net_u_1.make_node(name="u1_network", jit=cfg.jit)]
        + [diff_net_u_2.make_node(name="u2_network", jit=cfg.jit)]
        + [diff_net_u_3.make_node(name="u3_network", jit=cfg.jit)]
    )

    # make domain add constraints to the solver
    domain = Domain()

    # sympy variables
    x = Symbol("x")

    # right hand side (x = 3) Pt d
    rhs = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=L3,
        outvar={"u_3": Td},
        batch_size=cfg.batch_size.rhs,
        criteria=Eq(x, bar_lenght_1+bar_lenght_2+bar_lenght_3),
        parameterization=pr,
    )
    domain.add_constraint(rhs, "right_hand_side")

    # left hand side (x = 0) Pt a
    lhs = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=L1,
        outvar={"u_1": Ta},
        batch_size=cfg.batch_size.lhs,
        criteria=Eq(x, 0),
        parameterization=pr,
    )
    domain.add_constraint(lhs, "left_hand_side")

    # interface 1-2
    interface_1_2 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=L1,
        outvar={
            "diffusion_interface_dirichlet_u_1_u_2": 0,
            "diffusion_interface_neumann_u_1_u_2": 0,
        },
        batch_size=cfg.batch_size.interface,
        criteria=Eq(x, bar_lenght_1),
        parameterization=pr,
    )
    domain.add_constraint(interface_1_2, "interface_1_2")

    # interface 2-3
    interface_2_3 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=L2,
        outvar={
            "diffusion_interface_dirichlet_u_2_u_3": 0,
            "diffusion_interface_neumann_u_2_u_3": 0,
        },
        batch_size=cfg.batch_size.interface,
        criteria=Eq(x, bar_lenght_1+bar_lenght_2),
        parameterization=pr,
    )
    domain.add_constraint(interface_2_3, "interface_2_3")

    # interior 1
    interior_u1 = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=L1,
        outvar={"diffusion_u_1": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=pr,
    )
    domain.add_constraint(interior_u1, "interior_u1")

    # interior 2
    interior_u2 = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=L2,
        outvar={"diffusion_u_2": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=pr,
    )
    domain.add_constraint(interior_u2, "interior_u2")

    # interior 3
    interior_u3 = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=L3,
        outvar={"diffusion_u_3": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=pr,
    )
    domain.add_constraint(interior_u3, "interior_u3")

    # validation data
    x = np.expand_dims(np.linspace(0, 1, 100), axis=-1)
    u_1 = 153.8*x
    invar_numpy = {"x":x}
    invar_numpy.update({key: np.full_like(x, value) for key, value in params_validation.items()})
    outvar_numpy = {"u_1": u_1}
    val = PointwiseValidator(nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy)
    domain.add_validator(val, name="Val1")

    # make validation data line 2
    x = np.expand_dims(np.linspace(1, 2, 100), axis=-1)
    u_2 = 30.77*x+123.03
    invar_numpy.update({"x": x})
    invar_numpy.update({key: np.full_like(x, value) for key, value in params_validation.items()})
    outvar_numpy = {"u_2": u_2}
    val = PointwiseValidator(nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy)
    domain.add_validator(val, name="Val2")

    # make validation data line 3
    x = np.expand_dims(np.linspace(2, 3, 100), axis=-1)
    u_3 = 15.38*x+153.81
    invar_numpy.update({"x": x})
    invar_numpy.update({key: np.full_like(x, value) for key, value in params_validation.items()})
    outvar_numpy = {"u_3": u_3}
    val = PointwiseValidator(nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy)
    domain.add_validator(val, name="Val3")

    # make monitors
    invar_numpy = {"x": [[1.0]]}
    invar_numpy.update({key:[[value]] for key, value in params_validation.items()})
    monitor = PointwiseMonitor(
        invar_numpy,
        output_names=["u_1__x"],
        metrics={"flux_u1": lambda var: torch.mean(var["u_1__x"])},
        nodes=nodes,
        requires_grad=True,
    )
    domain.add_monitor(monitor)

    monitor = PointwiseMonitor(
        invar_numpy,
        output_names=["u_2__x"],
        metrics={"flux_u2": lambda var: torch.mean(var["u_2__x"])},
        nodes=nodes,
        requires_grad=True,
    )
    domain.add_monitor(monitor)

    invar_numpy.update({"x": [[2.0]]})
    monitor = PointwiseMonitor(
        invar_numpy,
        output_names=["u_3__x"],
        metrics={"flux_u3": lambda var: torch.mean(var["u_3__x"])},
        nodes=nodes,
        requires_grad=True,
    )
    domain.add_monitor(monitor)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
