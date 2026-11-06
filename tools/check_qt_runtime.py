#!/usr/bin/env python3
"""Exercise Qt void-return bindings without a display, model, or GPU.

Some ARM64 abi3 wheels import successfully on Python 3.10 but consume a None
reference per void call. Repeated viewport painting then aborts the interpreter.
Keep this probe short so an affected runtime can report failure before aborting.
"""
import gc
import json
import sys


def check():
    from PySide6 import __version__
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF

    image = QImage(64, 64, QImage.Format_ARGB32)
    painter = QPainter(image)
    pen = QPen(QColor('red'), 1.5)
    color = QColor('red')
    polygon = QPolygonF([QPointF(1., 1.), QPointF(20., 1.), QPointF(10., 20.)])

    def paint():
        painter.setPen(pen)
        painter.setBrush(color)
        painter.drawPolygon(polygon)

    enabled = gc.isenabled()
    try:
        for _ in range(8):
            paint()
        gc.collect()
        gc.disable()
        before = sys.getrefcount(None)
        for _ in range(64):
            paint()
        delta = sys.getrefcount(None) - before
    finally:
        painter.end()
        if enabled:
            gc.enable()
    result = dict(pyside=__version__, python='.'.join(map(str, sys.version_info[:3])),
                  calls=192, none_refcount_delta=delta, passed=delta >= -16)
    print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    try:
        code = check()
    except ImportError as error:
        print(json.dumps(dict(passed=False, available=False, error=str(error))), flush=True)
        code = 2  # GUI is optional for CLI-only installations.
    raise SystemExit(code)
