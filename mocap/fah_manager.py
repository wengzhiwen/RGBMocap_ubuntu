"""FAH readiness, setup guidance and MANO import in Model Manager."""
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from . import fah_runtime


def status_line():
    rows = [row for row in fah_runtime.readiness() if row['required']]
    gaps = [row for row in rows if not row['ok']]
    if not fah_runtime.supported():
        return ('FAH runtime: not available here', 'FAH finger refinement is offered on the Linux route only.')
    if not gaps:
        stamp = fah_runtime.runtime_stamp()
        return ('FAH runtime: Ready',
                'Environment: ' + str(fah_runtime.venv()) +
                '\nPackage: FAH ' + str(stamp.get('package')) + ' at ' + str(fah_runtime.REVISION)[:12] +
                '\nPython: ' + str(stamp.get('python') or 'see ' + fah_runtime.VENV) +
                '\nGated files stay outside this app; nothing is installed or downloaded by RGB MoCap.')
    return ('FAH runtime: Setup required',
            '\n'.join(f"{row['name']}: {row['detail']}" for row in gaps))


class _RuntimePanel(QWidget):
    def __init__(self, parent):
        super().__init__()
        self._parent = parent
        self.box = QHBoxLayout(self)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.PlainText)
        self.import_button = QPushButton('Import FAH MANO...')
        self.import_button.setToolTip('Import a trusted upstream-converted MANO_RIGHT.pkl for FAH only. '
                                      'Raw Chumpy or official MANO pickles are rejected and existing files are not replaced.')
        self.import_button.clicked.connect(self.import_mano)
        self.setup_button = QPushButton('FAH Runtime Setup...')
        self.setup_button.setToolTip('Show the separate pinned FAH environment and the gated files this option needs. '
                                     'RGB MoCap never installs them for you.')
        self.setup_button.clicked.connect(lambda: setup_instructions(self._parent))
        self.box.addWidget(self.label, 1)
        self.box.addWidget(self.import_button)
        self.box.addWidget(self.setup_button)
        self.refresh()

    def refresh(self):
        text, detail = status_line()
        self.label.setText(text)
        self.label.setToolTip(detail)
        if text.endswith('Setup required') or text.endswith('not available here'):
            self.label.setStyleSheet('color:#ffb45c')
        else:
            self.label.setStyleSheet('')
        busy = bool(getattr(self._parent.window, 'busy', False))
        self.import_button.setEnabled(fah_runtime.supported() and not busy)

    def import_mano(self):
        parent = self._parent
        if not fah_runtime.supported() or bool(getattr(parent.window, 'busy', False)):
            return
        target = Path(fah_runtime.asset_roots()['mano']) / 'MANO_RIGHT.pkl'
        start = str(target.parent) if target.parent.exists() else ''
        selected, _ = QFileDialog.getOpenFileName(
            parent,
            'Import converted FAH MANO_RIGHT.pkl',
            start,
            'MANO pickles (*.pkl);;All files (*)')
        if not selected:
            return
        parent.start_model_jobs([('import_model', ['--model', 'fah_mano_right', '--video', selected])])

def runtime_panel(parent):
    return _RuntimePanel(parent)


def setup_instructions(parent):
    dialog = QDialog(parent)
    dialog.setWindowTitle('FAH separate runtime setup')
    dialog.resize(720, 560)
    layout = QVBoxLayout(dialog)
    text = QTextBrowser()
    text.setOpenExternalLinks(True)
    text.document().setDefaultStyleSheet('a { color: #8ed2ff; text-decoration: underline; }')
    paths = fah_runtime.asset_paths()
    pins = ' '.join(f'{key}="{value}"' for key, value in fah_runtime.configure_environment().items())
    lines, gaps = [], []
    for row in fah_runtime.readiness():
        (lines if row['ok'] or not row['required'] else gaps).append(
            f"<li>{row['name']}: {'ready' if row['ok'] else '<b>' + str(row['detail']) + '</b>'}</li>")
    text.setHtml(f'''<h3>Optional FAH finger refinement</h3>
<p>Default off. Selecting <b>FAH Finger Refinement</b> runs the pinned upstream
<a href="{fah_runtime.PROJECT_URL}">FAH</a> hand model in its own isolated environment
(<code>{fah_runtime.VENV}</code> in this app folder) as a child of the FlowHMR stage.
It changes only finger rotations; body, wrists, trajectory, shape and contacts are kept.
Existing HaMeR and plain FlowHMR behavior, caches and receipts are unchanged.</p>
<h4>1. Provision the pinned environment yourself</h4>
<p>Close RGB MoCap, then create <code>{fah_runtime.VENV}</code> in this app folder from the exact
upstream revision this app was checked against:</p>
<p><code>commit {fah_runtime.REVISION}</code> &nbsp; package <code>{fah_runtime.PACKAGE}=={fah_runtime.PACKAGE_VERSION}</code></p>
<p>Install FAH from Git (so the installed distribution records that commit id in its PEP 610
<code>direct_url.json</code>), together with a CUDA build of Torch. The app never installs or
upgrades anything, and it will refuse to run FAH when the recorded commit differs. If your
installer cannot record it, register the same commit deliberately in
<code>{fah_runtime.SOURCE_REGISTRATION}</code>, for example
<code>{{"project":"{fah_runtime.PROJECT}","package":"{fah_runtime.PACKAGE_VERSION}","commit":"{fah_runtime.REVISION}"}}</code>.</p>
<h4>2. Provide the gated files (they are never redistributed here)</h4>
<p>Use your own registered accounts. This app looks for them through the upstream settings:</p>
<ul>
<li>FAH weights: <code>{paths['roots']['weights']}</code> (<code>FAH_WEIGHTS</code>, under <code>FAH_HOME</code>)</li>
<li>SAM 3D Body configuration and MHR geometry: <code>{paths['roots']['sam3d']}</code> (<code>FAH_SAM3D</code>)</li>
<li>MANO right hand: <code>{paths['roots']['mano']}</code> (<code>MANO_DIR</code>). Use the converted version via <b>Import FAH MANO...</b>; current configured file: <code>{paths['roots']['mano'] / 'MANO_RIGHT.pkl'}</code>.</li>
</ul>
<p>Current readiness:</p><ul>{''.join(lines)}</ul>
{'' if not gaps else '<p><b>Still needed:</b></p><ul>' + ''.join(gaps) + '</ul>'}
<h4>3. Check it without starting a reconstruction</h4>
<p><code>./rgbmocap-cli.sh doctor</code> reports these rows, and
<code>./rgbmocap-cli.sh doctor --gpu</code> additionally imports the pinned package in
<code>{fah_runtime.VENV}</code> and runs one real CUDA operation. No weights are downloaded.</p>
<p>Optional overrides: <code>RGB_MOCAP_FAH_MODEL</code> selects the released FAH backbone
({', '.join(fah_runtime.MODEL_CHOICES)}; default <code>{fah_runtime.DEFAULT_MODEL}</code>).
Assets can also be placed with <code>FAH_HOME</code>, <code>FAH_WEIGHTS</code>, <code>FAH_SAM3D</code> and
<code>MANO_DIR</code>, exactly as upstream FAH names them, for example: <code>{pins}</code>.</p>
<p>FAH results are cached separately from every other generation, with their own source,
package and file stamps. Changing the FAH choice, its files or the pinned code starts a new
result instead of reusing another prediction.</p>''')
    layout.addWidget(text)
    actions = QHBoxLayout()
    copy = QPushButton('Copy Asset Settings')
    copy.setToolTip('Copy the environment variables that point at the gated files now in use.')
    copy.clicked.connect(lambda: QApplication.clipboard().setText(pins))
    actions.addWidget(copy)
    actions.addStretch()
    close = QPushButton('Close')
    close.setToolTip('Close these instructions. Nothing was installed or changed.')
    close.clicked.connect(dialog.accept)
    actions.addWidget(close)
    layout.addLayout(actions)
    dialog.exec()
