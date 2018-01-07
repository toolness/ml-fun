import ctypes
from pathlib import Path


MY_DIR = Path(__file__).parent.resolve()
CDLL_DIR = MY_DIR / 'target' / 'release'
CDLL_FILE = CDLL_DIR / 'easy21'

easy21 = ctypes.CDLL(str(CDLL_FILE))

OUTPUT_ARR = ctypes.c_float * easy21.get_output_size()

output = OUTPUT_ARR()

print(easy21.run_monte_carlo(30_000, ctypes.byref(output)))

for reward in output:
    print(reward)
