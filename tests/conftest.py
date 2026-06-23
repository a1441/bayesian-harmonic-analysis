import os
import sys

# make `import harmonix` work without installing (src layout)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
