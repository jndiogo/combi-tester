import sys, traceback

# from https://stackoverflow.com/a/28836286
class ScriptError(Exception): pass

def execx(cmd, 
          globals=None, 
          locals=None, 
          description='source string'):
    
    try:
        exec(cmd, globals, locals)
    except SyntaxError as err:
        error_class = err.__class__.__name__
        detail = err.args[0]
        line_number = err.lineno
    except Exception as err:
        error_class = err.__class__.__name__
        detail = err.args[0]
        cl, exc, tb = sys.exc_info()
        line_number = traceback.extract_tb(tb)[-1][1]
    else:
        return
    
    raise ScriptError("%s at line %d of %s: %s" % (error_class, line_number, description, detail))

