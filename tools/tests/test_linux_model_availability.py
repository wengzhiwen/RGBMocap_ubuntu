"""The Linux model blacklist: Windows-only models are hidden from the GUI.

The 2026-11-20 upstream features are gated the same way: Track Face capture
(Windows base runtime) and the Unreal Live Link stream target (Win64 bridge).

QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tools/tests -p test_linux_model_availability.py -v
"""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mocap.face_runtime import AVAILABLE as FACE_CAPTURE_AVAILABLE
from mocap.model_guide import guide_html
from mocap.motion_tools import LINUX_UNAVAILABLE, MOTION_TOOLS, selectable_tools
from mocap.paths import worker_command


class Blacklist(unittest.TestCase):
    def test_blacklist_only_contains_registered_windows_only_models(self):
        self.assertEqual(set(LINUX_UNAVAILABLE), {'pear', 'mamma', 'prompthmr', 'gemx'})
        self.assertTrue(set(LINUX_UNAVAILABLE) <= set(MOTION_TOOLS))

    def test_selectable_tools_hide_blacklist_on_posix(self):
        tools = selectable_tools('posix')
        self.assertEqual(set(tools), {'gvhmr', 'flowhmr', 'exmr', 'fab'})
        self.assertEqual(tools['flowhmr'], MOTION_TOOLS['flowhmr'])

    def test_selectable_tools_keep_everything_on_windows(self):
        self.assertEqual(selectable_tools('nt'), MOTION_TOOLS)

    def test_batch_models_drop_prompthmr_on_this_platform(self):
        if os.name != 'nt':
            from mocap.video_batch_ui import BATCH_MODELS
            self.assertEqual(set(BATCH_MODELS), {'exmr', 'flowhmr', 'gvhmr'})
        else:
            from mocap.video_batch_ui import BATCH_MODELS
            self.assertIn('prompthmr', BATCH_MODELS)

    def test_model_guide_omits_blacklisted_models_on_posix(self):
        html = guide_html('flowhmr')
        for key in selectable_tools('posix'):
            self.assertIn(MOTION_TOOLS[key].label, html)
        for key in LINUX_UNAVAILABLE:
            self.assertNotIn(MOTION_TOOLS[key].label, html)


try:
    from PySide6.QtWidgets import QApplication
    from mocap.gui import MainWindow
    HAS_QT = True
except ImportError:
    HAS_QT = False


@unittest.skipUnless(HAS_QT, 'Requires the pinned GUI environment')
class RealSelector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.w = MainWindow()

    def tearDown(self):
        self.w._video_batch = None
        self.w.busy = False
        self.w.close()
        self.app.processEvents()

    def _keys(self):
        return {self.w.model_choice.itemData(i) for i in range(self.w.model_choice.count())}

    def test_selector_shows_only_linux_models(self):
        if os.name == 'nt':
            self.skipTest('Windows keeps every model entry')
        self.assertEqual(self._keys(), {'gvhmr', 'flowhmr', 'exmr', 'fab'})
        self.assertEqual(self.w.model_choice.currentData(), 'exmr')

    def test_people_tracker_has_no_sam31_on_linux(self):
        if os.name == 'nt':
            self.skipTest('Windows keeps the SAM 3.1 tracker')
        trackers = {self.w.people_tracker.itemData(i) for i in range(self.w.people_tracker.count())}
        self.assertEqual(trackers, {'default'})

    def test_track_face_hidden_and_capture_never_queued_on_linux(self):
        if os.name == 'nt':
            self.skipTest('Windows keeps Track Face capture')
        self.assertFalse(FACE_CAPTURE_AVAILABLE)
        self.assertTrue(self.w.track_face.isHidden())
        self.assertIsNone(self.w.face_job())

    def test_stream_target_drops_unreal_live_link_on_linux(self):
        if os.name == 'nt':
            self.skipTest('Windows keeps the Unreal Live Link target')
        targets = [self.w.stream_target.itemText(i) for i in range(self.w.stream_target.count())]
        self.assertEqual(targets, ['None', 'Blender'])

    def test_face_track_command_reports_linux_unavailability(self):
        if os.name == 'nt':
            self.skipTest('Windows launches the face worker normally')
        with self.assertRaisesRegex(RuntimeError, 'unavailable on Linux'):
            worker_command('face_track')

    def test_blacklisted_project_model_still_opens(self):
        # The registry keeps Windows-only models readable for existing projects.
        tool = MOTION_TOOLS['gemx']
        self.assertEqual(tool.key, 'gemx')


if __name__ == '__main__':
    unittest.main()
