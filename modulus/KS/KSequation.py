from sympy import Symbol, Function, Number
from modulus.pdes import PDES

class KSequation(PDES):
    name = "KSequation"

    def __init__(self, ni=1.0):
        x = Symbol("x")
        t = Symbol("t")

        input_variables = {"x": x,
                           "t": t}

        u = Function("u")(*input_variables)

        if type(ni) is str:
            ni = Function(ni)(*input_variables)
        elif type(ni) in [float, int]:
            ni = Number(ni)

        self.equations = {}
        self.equations["ks_equation"] = u.diff(t) + u*u.diff(x) + u.diff(x).diff(x) + u.diff(x).diff(x).diff(x).diff(x)