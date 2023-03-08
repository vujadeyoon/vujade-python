import warnings
from vujade.vujade_debug import printd


if __name__=='__main__':
    print('[1/5] [print] Hello world.')
    warnings.warn('[2/5] [warn] Hello world.')
    input('[3/5] [input] Hello world.')
    printd('[4/5] [printd] Hello world.', _is_pause=True)
    printd('[5/5] [printd] Hello world.', _is_pause=False)

