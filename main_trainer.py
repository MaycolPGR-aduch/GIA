"""Acceso directo a Calibrar y entrenar: GIA se abre siempre desde main.py."""
import sys

from main import main

if __name__ == "__main__":
    sys.exit(main(["--train", *sys.argv[1:]]))
