from typing import Any, Optional, Union, Callable, Annotated, Literal, get_origin, get_args
from enum import IntEnum

import unicodedata


import logging
logger = logging.getLogger(__name__)


DEFAULT_EPSILON = 1e-6
NFORM = "NFKD" # compatibility decomposition




# =============================================================== scorers

def score(*args) -> float:

    if not len(args):
        return 0.

    total = 0.
    for val in args:
        assert val >= 0. and val <= 1., f"Values must be 0 <= value <= 1: but got {val}"

        total += float(val)

    return total / len(args)






# =============================================================== Equality checks
"""
General equality checks for simple types (bool, int, float, str). Return 0.0 or 1.0

equal(a, b, [b1,b2]):
    Same type, same value, strings compare sensitively.


same(a, b, [b1,b2]):
    Approximately equal, allowing diff types, strings compare insensitively.
    Types bool only equal to other bool type.

b1,b2: alternative checks for any of the values
"""


def equal(val: Any,
          *other: Any,
          eps: float = DEFAULT_EPSILON,
          str_in: bool = False) -> float:

    val_type = type(val)

    for ot in other: 
        if val_type == type(ot): # only check args of same type: others don't match

            if val_type is bool or val_type is int:
                if val == ot:
                    return 1.
            
            elif val_type is float:
                if float_eq(val, ot, eps=eps):
                    return 1.

            elif val_type is str:
                if str_eq(val, ot, str_in=str_in):
                    return 1.

            else:
                raise TypeError(f"Only bool, int, float, str types supported")

    return 0.



def same(val: Any,
         *other: Any,
         eps: float = DEFAULT_EPSILON,
         str_in: bool = True) -> float:
    

    def float_from_any(val: Any) -> tuple[bool,float]:
        val_type = type(val)        
        assert val_type is not bool

        if val_type is int or val_type is float:
            return True,float(val)        
        else: # try float from str
            return is_float(val)


    src_val_type = type(val)
    src_val = val

    for ot in other:
        ot_type = type(ot)
        val = src_val
        val_type = src_val_type

        if val_type is ot_type: # equal types: pass to equal()
            if equal(val, ot, eps=eps, str_in=str_in):
                return 1.
            
        else: # different types: check case by case
            if val_type is bool or ot_type is bool: # one is bool, the other is not: can't be the same
                continue
            # no bool values below this line

            elif val_type is int or ot_type is int:
                if ot_type is int: # swap so val is int
                    val_type,ot_type = ot_type,val_type
                    val,ot = ot,val

                is_num, ot_val = float_from_any(ot) # convert ot to float
                if is_num:
                    if float_eq(float(val), ot_val, eps=eps):
                        return 1.

            elif val_type is float or ot_type is float:
                if ot_type is float: # swap so val is float
                    val_type,ot_type = ot_type,val_type
                    val,ot = ot,val

                is_num, ot_val = float_from_any(ot) # convert ot to float
                if is_num:
                    if float_eq(val, ot_val, eps=eps):
                        return 1.

            elif val_type is str or ot_type is str:
                is_num2, val2 = float_from_any(val) # convert val to float
                is_num, ot_val = float_from_any(ot) # convert ot to float
                if is_num and is_num2:
                    if float_eq(val, ot_val, eps=eps):
                        return 1.
                else:
                    if str_eq(val,ot, str_in=str_in):
                        return 1.

            else:
                raise TypeError(f"Only bool, int, float, str types supported")

    return 0.




def float_eq(val: float,
             *other: float,
             eps: float = DEFAULT_EPSILON) -> float:

    if type(val) is not float or any([type(ot) is not float for ot in other]):
        raise TypeError(f"Only args of float type allowed")

    for ot in other:
        if abs(val - ot) < eps:
            return 1.
        
    return 0.



def str_eq(val: str,
           *other,
           str_in: bool = False) -> float:
    """String equality test, insensitive by default"""

    if type(val) is not str or any([type(ot) is not str for ot in other]):
        raise TypeError(f"Only args of str type allowed")

    norm_fn = norm_in if str_in else norm

    nvalue = norm_fn(val)
    for ot in other:
        if nvalue == norm_fn(ot):
            return 1.
        
    return 0.



def str_eqin(val: str,
             *other) -> float:
    """Case-insensitive string equality test"""

    return str_eq(val, *other, str_in=True)




"""
match(a, "loop.+8")
matchin(a, "loop.+8")
"""

def norm(value: str) -> str:
    return unicodedata.normalize(NFORM, value)
def norm_in(value: str) -> str:
    return norm(value.casefold())











def is_float(val: Any) -> tuple[bool, float]:
    try:
        return True, float(val)
    except ValueError:
        return False, 0.



def bool_cast(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    elif isinstance(val, int) or isinstance(val, float):
        return val == 0.
    elif isinstance(val, str):
        is_flt,num = is_float(val)
        if is_flt:
            return val == 0.
        else:
            return val.lower() != "false"
    else:
        raise TypeError(f"Cannot cast '{val}' to bool")




# =============================================================== sub-str match

def str_sub(val: str,
            subs: Union[str,list[str]],
            split: Optional[str] = None,
            str_in: bool = False) -> float:
    """Case-sensitive substring test: is any of subs a substring of val?"""

    if not isinstance(val, str) or (not isinstance(subs, str) and not isinstance(subs, list)):
        raise TypeError(f"Only args of type str or list[str] allowed")

    norm_fn = norm_in if str_in else norm

    nvalue = norm_fn(val)

    if split:
        subs = subs.split(split)
    elif isinstance(subs, str):
        subs = [subs]

    for sub in subs:
        nsub = norm_fn(sub)
        if nvalue.find(nsub) != -1:
            return 1.
        
    return 0.



def istr_sub(val: str,
             subs: Union[str,list[str]],
             split: Optional[str] = None) -> float:
    """Case-insensitive substring test: is any of subs a substring of val?"""

    return str_sub(val, subs, 
                   split=split,
                   str_in=True)



def istr_sub_bar(val: str,
                 subs: Union[str,list[str]]) -> float:
    """Case-insensitive substring test splitting subs with "|" (vertical bar) if any. Is any of subs a substring of val?"""
    return str_sub(val, subs, 
                   split="|",
                   str_in=True)
    





# =============================================================== tests

def test_equal():

    def mcheck(a:float,
               b:float,
               fn, expected_fn):

        types = [bool, int, float, str]

        for type_a in types:
            val_a = type_a(a)
            for type_b in types:
                val_b = type_b(b)
                assert fn(val_a,val_b) == expected_fn(a,b, type_a,type_b)
         

    fn = equal
    def expected_fn(a,b, type_a, type_b):
        if type_a is not type_b:
            return 0.
        else:
            if type_a is bool or type_b is bool:
                return float(bool(a) == bool(b))
            else:
                return float(abs(a-b) < 1e-6)

    # multi type checks
    mcheck(0,0, fn, expected_fn)
    mcheck(1,1, fn, expected_fn)
    mcheck(1.1,1.1, fn, expected_fn)
    mcheck(2,1, fn, expected_fn)
    mcheck(-1,1, fn, expected_fn)

    # str
    assert equal("a", "a") == 1.
    assert equal("a", "A") == 0.    
    assert equal("a", "A", str_in=True) == 1.    
    assert equal("a", "A", "a") == 1.
    assert equal("a", "A", 1, True, 3.2) == 0.
    assert equal("a", "A", 1, True, 3.2, "a") == 1.
    assert equal("a", "A", "1", "True", "3.2", "a") == 1.
    assert equal("a", "b", 1, True, 3.2, "A", str_in=True) == 1.

    assert equal("Confirmations", "Conﬁrmations") == 1.
    assert equal("Confirmations", "conﬁrmations") == 0.
    assert equal("Confirmations", "nﬁrmations") == 0.

    # int
    assert equal(100, 100) == 1.
    assert equal(100, 101) == 0.
    assert equal(-100, -100) == 1.
    assert equal(100, -100) == 0.

    # float
    assert equal(0., 0.) == 1.
    assert equal(1., 1.) == 1.
    assert equal(100., 100.) == 1.
    assert equal(1e5, 1e5) == 1.
    assert equal(0., 1e-10) == 1.
    assert equal(0., 1e-4) == 0.
    assert equal(0., 1e-4, eps=1e-3) == 1.
    assert equal(0., 1e-4, eps=1e-5) == 0.
    assert equal(0., 1e-4, 1e-5, eps=1e-5) == 0.

    # bool
    assert equal(False, False) == 1.
    assert equal(True, False) == 0.
    assert equal(False,True) == 0.
    assert equal(True, True) == 1.
    assert equal(True, False, True) == 1.




def test_same():

    # str
    assert same("a", "a") == 1.
    assert same("a", "A") == 1.    
    assert same("a", "A", str_in=False) == 0.    
    assert same("a", "b", "a") == 1.
    assert same("a", "b", 1, True, 3.2) == 0.
    assert same("a", "b", 1, True, 3.2, "A") == 1.
    assert same("a", "b", "1", "True", "3.2", "a") == 1.
    assert same("a", "b", 1, True, 3.2, "A", str_in=False) == 0.

    assert same("Confirmations", "Conﬁrmations") == 1.
    assert same("Confirmations", "conﬁrmations") == 1.
    assert same("Confirmations", "nﬁrmations") == 0.

    # int
    assert same(100, 100) == 1.
    assert same(100, 101) == 0.
    assert same(-100, -100) == 1.
    assert same(100, -100) == 0.
    assert same(100, "100") == 1.
    assert same(100, 100.) == 1.
    assert same(100, True) == 0.
    assert same(100, True, "alsa", 9870.1, 100) == 1.


    # float
    assert same(0., 0.) == 1.
    assert same(1., 1.) == 1.
    assert same(100., 100.) == 1.
    assert same(1e5, 1e5) == 1.
    assert same(0., 1e-10) == 1.
    assert same(0., 1e-4) == 0.
    assert same(0., 1e-4, eps=1e-3) == 1.
    assert same(0., 1e-4, eps=1e-5) == 0.
    assert same(0., 1e-4, 1e-5, eps=1e-5) == 0.
    assert same(0., 1e-4, 1e-5, 1e-10, eps=1e-5) == 1.

    # bool
    assert same(False, False) == 1.
    assert same(True, False) == 0.
    assert same(False,True) == 0.
    assert same(True, True) == 1.
    assert same(True, False, True) == 1.

    assert same(True, 1, 1., "1", "True") == 0.
    assert same(True, 1, 1., "1", "True", True) == 1.


