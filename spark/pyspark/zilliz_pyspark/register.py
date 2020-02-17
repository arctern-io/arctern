import inspect

def _get_funcs_in_module(module):
    for name in module.__all__:
        obj = getattr(module, name)
        if inspect.isfunction(obj):
            yield obj

def register_funcs(spark):
    from . import  _wrapper_func
    all_funcs = _get_funcs_in_module(_wrapper_func)
    for obj in all_funcs:
        #print(obj.__name__, obj)
        spark.udf.register(obj.__name__, obj)
