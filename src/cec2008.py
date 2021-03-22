from opfunu.cec.cec2008.F7 import Model as f7
from opfunu.cec.cec2008.F6 import Model as f6
from opfunu.cec.cec2008.F5 import Model as f5
from opfunu.cec.cec2008.F4 import Model as f4
from opfunu.cec.cec2008.F3 import Model as f3
from opfunu.cec.cec2008.F2 import Model as f2
from opfunu.cec.cec2008.F1 import Model as f1
import numpy as np



def function1(solution, d):
    return f1()._main__(solution)


def function2(solution, d):
    return f2()._main__(solution)


def function3(solution, d):
    return f3()._main__(solution)


def function4(solution, d):
    return f4()._main__(solution)


def function5(solution, d):
    return f5()._main__(solution)


def function6(solution, d):
    return f6()._main__(solution)


def function7(solution, d):
    return f7()._main__(solution)

