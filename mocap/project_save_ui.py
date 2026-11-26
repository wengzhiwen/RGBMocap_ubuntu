"""File-menu project saving, with the bulk I/O isolated in a worker."""
import tempfile
from pathlib import Path
from PySide6.QtWidgets import QFileDialog
from .project_storage import storage_root, manifest
from .storage import atomic_json


class ProjectSaving:
    def save_project_as(self):
        if self.busy or self.contact_saving or self.mamma_pending or self.video_path is None:
            return
        if any(info.get('unsaved') for info in self.contact_tracks.values()):
            self.status.setText('Save the contact timing successfully before saving a project copy.')
            return
        root = self.workspace_root or ((storage_root(self.project) or self.people_root or self.project) if self.project else None)
        suggested = root.parent / (root.name + ' Copy') if root else self.video_path.with_suffix('')
        name, _ = QFileDialog.getSaveFileName(self, 'Save Project As — New Folder', str(suggested),
                                              'Project folder (*)', options=QFileDialog.DontConfirmOverwrite)
        if name:
            self.submit_project_save(Path(name))

    def submit_project_save(self, target):
        if self.busy or self.contact_saving or self.mamma_pending or self.video_path is None:
            return False
        if any(info.get('unsaved') for info in self.contact_tracks.values()):
            self.status.setText('Save the contact timing successfully before saving a project copy.')
            return False
        try:
            from .project_save import validate_destination
            root = self.workspace_root or ((storage_root(self.project) or self.people_root or self.project) if self.project else None)
            target = validate_destination(root, target)
            self.stop_playback()
            view = self.motion_view
            self._project_save_view = {key: getattr(view, key).copy() if hasattr(getattr(view, key), 'copy') else getattr(view, key)
                for key in ('yaw', 'pitch', 'distance', 'pan', 'scene_center')}
            self._project_save_temp = tempfile.TemporaryDirectory(prefix='rgbmocap-save-')
            request = Path(self._project_save_temp.name) / 'save.json'
            pipeline = {'result': self.selected_result()}
            # A loaded project's model identity belongs to its inference, even
            # when the user has selected another model for a future run.
            pipeline['model'] = manifest(self.project).get('pipeline', {}).get('model', 'gvhmr') if self.project else self.model_choice.currentData()
            atomic_json(request, dict(source=str(root) if root else None,
                active=str(self.project) if self.project else None, video=str(self.video_path), target=str(target),
                model=self.model_choice.currentData(), pipeline=pipeline,
                preview_fps=self.info['fps'],
                workspace_view=self.workspace_view(),
                workspace_settings=self.workspace_settings(),
                mamma_inputs=self.mamma_inputs,
                mamma_active_camera=self.mamma_active_camera,
                flow_options=self.selected_flow_options(),
                exmr_options={'static_camera':self.exmr_camera.currentData(),'smoothing':'global' if self.exmr_smoothing.isChecked() else 'none'},
                fab_options=self.selected_fab_options(),
                correction=self.selected_correction_options(), frame=self.slider.value(),
                trim=[self.slider.clip_in, self.slider.clip_out]))
            self._project_save_request = request
            self._project_saved_root = None
            self.start_jobs([('save_project', ['--request', str(request)])])
            return True
        except (OSError, ValueError) as error:
            self.status.setText('Cannot save project: ' + str(error))
            self.cleanup_project_save()
            return False

    def finish_project_save(self):
        root = getattr(self, '_project_saved_root', None)
        if root is not None:
            # A successful atomic commit wins over a last-millisecond Cancel.
            self.cancelled = False
            self.load_project(root)
            for key, value in self._project_save_view.items():
                setattr(self.motion_view, key, value)
            self.motion_view.auto_fit = False
            self.motion_view.update()
            self.status.setText('Project saved: ' + str(root))
            self.task_info.setText(self.task_info.text().replace('Stopped','Done',1))
            self.progress.setRange(0,100);self.progress.setValue(100)
        elif not self.failed:
            self.status.setText('Project save cancelled. The current project was kept.')
        self.cleanup_project_save()
        self.update_readiness()

    def cleanup_project_save(self):
        temporary = getattr(self, '_project_save_temp', None)
        if temporary is not None:
            temporary.cleanup()
        self._project_save_temp = None
        self._project_save_request = None
