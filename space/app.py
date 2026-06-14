import sys, os

# Put Space root on path so all flat-copied modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui import build_ui

demo = build_ui()

if __name__ == '__main__':
    demo.launch()
