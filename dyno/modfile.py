import os 
import dolang
from lark import Token, Lark

def ast_to_yaml(node, indent=""):
    indent += "  "
    if isinstance(node, Token):
        yield f"{indent}- type: {node.type}"
        yield f"{indent}  value: {repr(node.value)}"
    else:
        yield f"{indent}- type: {node.data}"
        yield f"{indent}  children:"
        for child in node.children:
            yield from ast_to_yaml(child, indent)

dir_path = os.path.dirname(os.path.realpath(__file__))

modfile_grammar = open(f"{dir_path}/modfile_grammar.lark").read()
modfile_parser = Lark(modfile_grammar, propagate_positions=True)


class Modfile:

    def __init__(self, filename):

        self.filename = filename
        try:
            self.data = modfile_parser.parse(open(filename).read())
        except Exception as e:
            raise e
    
        self.symbols = self.__find_symbols__()
        self.calibration = self.__get_calibration__()
        self.exogenous = self.__find_sigma__()
        
    @property
    def variables(self):
        return self.symbols['endogenous'] + self.symbols['exogenous']

    @property
    def parameters(self):
        return self.symbols['parameters']

    def __find_sigma__(self):

        import numpy as np
        from dyno.language import Normal

        ne = len(self.symbols['exogenous'])

        Sigma = np.zeros((ne,ne))

        for l in self.data.children:

            if l.data.value == "shocks_block":

                for ch in l.children:
                    
                    if ch.data.value == "setstdvar_stmt" or ch.data.value == "setvar_stmt":

                        k = ch.children[0].children[0].value
                        ve = ch.children[1] #.value

                        if isinstance(ve, str):
                            v = ve
                        else:
                            v = dolang.str_expression(ve)
                        
                        from math import exp
                        context = {'exp':exp}
                        cc = self.calibration.copy()
                        vv = eval(v.replace("^","**"), context,cc)
                        i = self.symbols["exogenous"].index(k)
                        if ch.data.value == "setstdvar_stmt":
                            Sigma[i,i] = vv**2
                        else:
                            Sigma[i,i] = vv

                    elif ch.data.value == "setcovar_stmt":

                        k = ch.children[0].children[0].value
                        l = ch.children[1].children[0].value
                        ve = ch.children[2]

                        if isinstance(ve, str):
                            v = ve
                        else:
                            v = dolang.str_expression(ve)

                        context = {'exp':exp}
                        cc = self.calibration.copy()

                        vv = eval(v.replace("^","**"), context, cc)
                        
                        i = self.symbols["exogenous"].index(k)
                        j = self.symbols["exogenous"].index(l)
                        
                        Sigma[i,j] = vv
                        Sigma[j,i] = vv

        return Normal(Σ=Sigma)

    def __get_calibration__(self):
        
        calibration = {}
        for l in self.data.children:

            if l.data.value == "parassignment":

                k = l.children[0].children[0].value
                ve = l.children[1]

                v = dolang.str_expression(ve)
                try:
                    vv = eval(v.replace("^","**"))
                except:
                    vv = v

                calibration[k] = vv
        
            elif l.data.value == "initval_block":
                for ll in l.children:
                    k = ll.children[0].children[0].value
                    ve = ll.children[1]

                    v = dolang.str_expression(ve)
                    try:
                        vv = eval(v.replace("^","**"))
                    except:
                        vv = v
                    calibration[k] = vv

        return calibration

    
    def __find_symbols__(self):
        
        # so far we discard latex and names
        get_name = lambda x: x.children[0].children[0].value

        dfs = []
        for l in self.data.children:
            if l.data.value == "var_statement":
                for tp in l.children:
                    name = get_name(tp)
                    dfs.append((name, "endogenous"))
            elif l.data.value == "varexo_statement":
                for tp in l.children:
                    name = get_name(tp)
                    dfs.append((name, "exogenous"))
            elif l.data.value == "par_statement":
                for tp in l.children:
                    name = get_name(tp)
                    dfs.append((name, "parameters"))

        symbols = {
            'endogenous': tuple(e[0] for e in dfs if e[1]=="endogenous"),
            'exogenous': tuple(e[0] for e in dfs if e[1]=="exogenous"),
            'parameters': tuple(e[0] for e in dfs if e[1]=="parameters")
        }

        return symbols


