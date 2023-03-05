import warnings
from vujade.vujade_debug import printf


if __name__=='__main__':
    print('[1/5] [print] Hello world.')
    warnings.warn('[2/5] [warn] Hello world.')
    input('[3/5] [input] Hello world.')
    printf('[4/5] [printf] Hello world.', _is_pause=True)
    printf('[5/5] [printf] Hello world.', _is_pause=False)

