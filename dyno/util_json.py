from typing import Any


class UnsupportedDynareFeature(Exception):

    pass


def get_allowed_functions():
    from math import exp, log, log10, sqrt
    from numpy import cbrt, sign
    from math import sin, cos, tan
    from math import asin, acos, atan
    from math import sinh, cosh, tanh
    from math import asinh, acosh, atanh

    # abs, max and min are builtins
    safe_list = [
        exp,
        log,
        log10,
        sqrt,
        cbrt,
        sign,
        abs,
        max,
        min,
        sin,
        cos,
        tan,
        asin,
        acos,
        atan,
        sinh,
        cosh,
        tanh,
        asinh,
        acosh,
        atanh,
    ]
    safe_dict = {f.__name__: f for f in safe_list}  # type: ignore
    safe_dict["ln"] = log  # Add alias for compatibility with Dynare
    return safe_dict


def json_safe_eval(ast: dict[str, Any], context: dict[str, dict[str, Any]]):
    endogenous_present = context["endogenous_present"]
    endogenous_future = context["endogenous_future"]
    endogenous_past = context["endogenous_past"]
    exogenous = context["exogenous"]
    parameters = context["parameters"]
    local_variables = context["local_variables"]
    allowed_functions = get_allowed_functions()
    match ast["node_type"]:
        # Bases cases
        case "NumConstNode":
            value = ast["value"]
            return float(value)

        case "VariableNode":
            name = ast["name"]
            lag = ast["lag"]
            match ast["type"]:
                case "modelLocalVariable":
                    return local_variables[name]
                case "parameter":
                    return parameters[name]
                case "exogenous":
                    return exogenous[name]
                case "endogenous":
                    match lag:
                        case 0:
                            return endogenous_present[name]
                        case 1:
                            return endogenous_future[name]
                        case -1:
                            return endogenous_past[name]
                        case _:
                            raise ValueError("Unsupported lag value")
                case _:
                    raise ValueError("Unsupported variable type")

        # Inductive cases
        case "BinaryOpNode":
            arg1 = json_safe_eval(ast["arg1"], context)
            arg2 = json_safe_eval(ast["arg2"], context)
            match ast["op"]:
                case "=" | "-":
                    return arg1 - arg2
                case "+":
                    return arg1 + arg2
                case "*":
                    return arg1 * arg2
                case "/":
                    return arg1 / arg2
                case "^":
                    return arg1**arg2
                case "<":
                    return float(arg1 < arg2)
                case ">":
                    return float(arg1 > arg2)
                case "<=":
                    return float(arg1 <= arg2)
                case ">=":
                    return float(arg1 >= arg2)
                case "==":
                    return float(arg1 == arg2)
                case "!=":
                    return float(arg1 != arg2)
                case _:
                    raise ValueError("Unknown binary operator")

        case "UnaryOpNode":
            op = ast["op"]
            arg = json_safe_eval(ast["arg"], context)
            match op:
                case "uminus":
                    return -arg
                case "uplus":
                    return arg
                case _:
                    if op not in allowed_functions.keys():
                        raise UnsupportedDynareFeature(
                            f"Function {op} is not supported (yet)"
                        )
                    return allowed_functions[op](arg)

        case _:
            raise UnsupportedDynareFeature(
                f"Node type {ast['node_type']} is not supported (yet)"
            )
