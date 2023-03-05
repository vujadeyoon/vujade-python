import warnings
import vujade.vujade_logger
from vujade.vujade_debug import printf


if __name__=='__main__':
    print('[print] Hello world.')
    warnings.warn('[warn] Hello world.')
    input('[input] Hello world.')
    printf('[printf] Hello world.', _is_pause=True)
    printf('[printf] Hello world.', _is_pause=False)

