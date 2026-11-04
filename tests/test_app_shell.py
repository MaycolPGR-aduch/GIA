"""Ventana única: navegación, ciclo de vida de la sesión y casos límite (sin cámara real)."""
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEventLoop, QLockFile
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

import app.assistive_controls as assistive_controls
import app.main_window as main_window_module
import app.qt_helpers as qt_helpers
from app import app_runtime
from app.config_store import CALIBRATION_DIR, delete_profile, build_default_profile, save_profile
from app.home_page import NewProfileDialog, primary_action, validate_profile_name
from app.main_window import GiaMainWindow
from app.models import AppState
from app.runtime_app import AssistantPage

qt_app = QApplication.instance() or QApplication([])
qt_app.setQuitOnLastWindowClosed(False)


def pump(until=lambda: False, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not until() and time.monotonic() < deadline:
        QCoreApplication.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.005)


class FakeClassifier:
    active_version = 1
    training_summary = {"validation_accuracy": 0.9}


class FakeController:
    instances = []
    fail_on_create = False
    stop_delay_s = 0.0

    def __init__(self, profile, settings, main_gui_interface=None):
        if FakeController.fail_on_create:
            raise RuntimeError("modelo dañado")
        self.profile, self.settings, self.gui = profile, settings, main_gui_interface
        self.classifier = FakeClassifier()
        self.started = 0
        self.stopped = 0
        FakeController.instances.append(self)

    def set_camera_overlay_details(self, enabled):
        pass

    def start(self):
        self.started += 1

    def stop(self):
        time.sleep(FakeController.stop_delay_s)
        self.stopped += 1


def make_profile(calibrated: bool) -> str:
    name = f"zz_shell_{uuid.uuid4().hex[:8]}"
    profile = build_default_profile(name)
    profile["calibration"]["completed"] = calibrated
    save_profile(profile)
    if calibrated:
        folder = CALIBRATION_DIR / name
        folder.mkdir(parents=True, exist_ok=True)
        registry = {"profile": name, "active_version": 1, "versions": [{"version": 1, "validation_accuracy": 0.9, "trained_at": "2026-11-04T10:00:00", "classes": ["neutral"]}]}
        (folder / "model_registry.json").write_text(json.dumps(registry), encoding="utf-8")
    return name


class NameRulesTests(unittest.TestCase):
    def test_profile_name_validation(self):
        self.assertEqual(validate_profile_name("Maycol 2", []), "")
        self.assertEqual(validate_profile_name("José (prueba)", []), "")
        self.assertIn("Introduce", validate_profile_name("   ", []))
        self.assertIn("Ya existe", validate_profile_name("maycol", ["Maycol"]))
        for bad in ("a/b", "a\\b", "..", "x:y", "a" * 41, "fin."):
            self.assertNotEqual(validate_profile_name(bad, []), "", bad)

    def test_primary_action_depends_on_a_trained_model(self):
        self.assertEqual(primary_action({"calibrated": True, "model_version": 3}), "start")
        self.assertEqual(primary_action({"calibrated": True, "model_version": None}), "calibrate")
        self.assertEqual(primary_action({"calibrated": False, "model_version": 3}), "calibrate")


class ShellTests(unittest.TestCase):
    def setUp(self):
        FakeController.instances = []
        FakeController.fail_on_create = False
        FakeController.stop_delay_s = 0.0
        self.names = []
        patches = [
            mock.patch.object(assistive_controls, "AssistiveController", FakeController),
            mock.patch("app.main_window.save_settings"),
            mock.patch("app.home_page.save_settings"),
            mock.patch("app.trainer_app.save_settings"),
            mock.patch("app.ui_theme.save_settings", create=True),
            mock.patch.object(app_runtime, "ERROR_LOG_PATH", Path(tempfile.gettempdir()) / f"gia_test_{uuid.uuid4().hex[:8]}.log"),
            mock.patch("sys.stderr"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.window = None

    def tearDown(self):
        if self.window is not None:
            self.window.home.shutdown()
            if self.window._closer is not None:
                self.window._closer.wait(5000)
            if self.window.trainer is not None:
                self.window.trainer.shutdown()
            self.window.deleteLater()
        pump(timeout_s=0.05)
        for name in self.names:
            delete_profile(name)

    def profile(self, calibrated=True) -> str:
        name = make_profile(calibrated)
        self.names.append(name)
        return name

    def build(self, **settings) -> GiaMainWindow:
        merged = {"preflight_on_start": False, "last_profile": None}
        merged.update(settings)
        self.window = GiaMainWindow(merged)
        self.window.show()
        return self.window

    def wait_closed(self):
        pump(lambda: self.window._closer is None and self.window.controller is None)

    def test_session_starts_and_returns_home_repeatedly(self):
        name = self.profile()
        window = self.build()
        for cycle in range(1, 4):
            window.start_assistant(name)
            self.assertIsInstance(window.stack.currentWidget(), AssistantPage)
            controller = window.controller
            self.assertEqual(controller.started, 1)
            self.assertEqual(window.settings["last_profile"], name)
            window.assistant.request_end_session()
            self.assertIs(window.stack.currentWidget(), window.home)
            self.wait_closed()
            self.assertEqual(controller.stopped, 1)
            self.assertEqual(window.stack.count(), 1)
            self.assertEqual(len(FakeController.instances), cycle)

    def test_cannot_start_twice_or_during_closing(self):
        name = self.profile()
        window = self.build()
        window.start_assistant(name)
        window.start_assistant(name)
        self.assertEqual(len(FakeController.instances), 1)
        FakeController.stop_delay_s = 0.2
        window.end_session()
        window.end_session()
        window.start_assistant(name)  # aún se está cerrando: no debe abrir otra sesión sobre la cámara
        self.assertEqual(len(FakeController.instances), 1)
        self.assertTrue(window.home.lbl_busy.text())
        self.wait_closed()
        self.assertEqual(FakeController.instances[0].stopped, 1)
        self.assertEqual(window.home.lbl_busy.text(), "")

    def test_failure_on_start_returns_home_with_message(self):
        name = self.profile()
        window = self.build()
        FakeController.fail_on_create = True
        with mock.patch.object(QMessageBox, "warning") as warning:
            window.start_assistant(name)
        warning.assert_called_once()
        self.assertIn("modelo dañado", warning.call_args[0][2])
        self.assertIs(window.stack.currentWidget(), window.home)
        self.assertIsNone(window.controller)
        self.assertEqual(window.stack.count(), 1)
        self.assertEqual(window.home.lbl_busy.text(), "")
        FakeController.fail_on_create = False
        window.start_assistant(name)
        self.assertIsNotNone(window.controller)

    def test_voice_close_command_ends_session_through_the_ui(self):
        name = self.profile()
        window = self.build()
        window.start_assistant(name)
        real = object.__new__(_RealController)
        real.main_gui_interface = window.assistant
        worker = threading.Thread(target=real._quit_from_action)
        worker.start()
        worker.join()
        pump(lambda: window.stack.currentWidget() is window.home)
        self.assertIs(window.stack.currentWidget(), window.home)
        self.wait_closed()
        self.assertEqual(FakeController.instances[0].stopped, 1)

    def test_closing_window_during_session_stops_it(self):
        name = self.profile()
        window = self.build()
        window.start_assistant(name)
        controller = window.controller
        with mock.patch.object(QApplication, "quit"):
            window.close()
        self.assertEqual(controller.stopped, 1)

    def test_closing_window_while_session_closes_waits_for_it(self):
        name = self.profile()
        window = self.build()
        window.start_assistant(name)
        controller = window.controller
        FakeController.stop_delay_s = 0.2
        window.end_session()
        with mock.patch.object(QApplication, "quit"):
            window.close()
        self.assertEqual(controller.stopped, 1)

    def test_trainer_and_session_never_share_the_camera(self):
        name = self.profile(calibrated=False)
        window = self.build()
        window.open_trainer(name)
        self.assertIs(window.stack.currentWidget(), window.trainer)
        with mock.patch.object(window.trainer, "release_camera") as release:
            window.start_assistant(name)
        release.assert_called_once()
        self.assertIsNotNone(window.controller)
        page = window.stack.currentWidget()
        window.open_trainer(name)  # con una sesión activa no se abre: la cámara está ocupada
        self.assertIs(window.stack.currentWidget(), page)

    def test_training_blocks_leaving_the_trainer(self):
        name = self.profile(calibrated=False)
        window = self.build()
        window.open_trainer(name)
        window.trainer.training_worker = object()
        with mock.patch.object(QMessageBox, "information") as info:
            window.show_home()
            window.start_assistant(name)
        self.assertEqual(info.call_count, 2)
        self.assertIs(window.stack.currentWidget(), window.trainer)
        self.assertIsNone(window.controller)
        window.trainer.training_worker = None
        window.show_home()
        self.assertIs(window.stack.currentWidget(), window.home)

    def test_try_now_starts_assistant_from_trainer(self):
        name = self.profile(calibrated=False)
        window = self.build()
        window.open_trainer(name)
        window.trainer.assistant_requested.emit(name)
        self.assertIsInstance(window.stack.currentWidget(), AssistantPage)

    def test_opening_gia_never_starts_a_session_by_itself(self):
        name = self.profile()
        window = self.build(last_profile=name)
        self.assertEqual(window.home.selected_name(), name)
        pump(timeout_s=0.2)
        self.assertIsNone(window.controller)
        self.assertEqual(FakeController.instances, [])
        self.assertIs(window.stack.currentWidget(), window.home)

    def test_back_home_button_asks_and_ends_session(self):
        name = self.profile()
        window = self.build()
        window.start_assistant(name)
        controller = window.controller
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            window.assistant.btn_home.click()
        self.assertIs(window.controller, controller)
        self.assertEqual(controller.stopped, 0)
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            window.assistant.btn_home.click()
        self.assertIs(window.stack.currentWidget(), window.home)
        self.wait_closed()
        self.assertEqual(controller.stopped, 1)

    def test_open_trainer_at_launch_ignores_unknown_profile(self):
        name = self.profile(calibrated=False)
        window = self.build(last_profile=name)
        window.open_trainer_at_launch("no_existe_este_perfil")
        self.assertIs(window.stack.currentWidget(), window.home)
        window.open_trainer_at_launch(name)
        self.assertIs(window.stack.currentWidget(), window.trainer)

    def test_home_offers_calibration_for_new_profile_and_start_for_ready_one(self):
        ready, new = self.profile(), self.profile(calibrated=False)
        window = self.build()
        window.home.refresh(new)
        self.assertEqual(window.home.btn_primary.text(), "Calibrar y entrenar")
        self.assertTrue(window.home.btn_train.isHidden())
        window.home.refresh(ready)
        self.assertEqual(window.home.btn_primary.text(), "Iniciar asistente")
        self.assertFalse(window.home.btn_train.isHidden())
        pump(lambda: not window.home._inspectors)

    def test_damaged_model_is_reported_on_home(self):
        ready = self.profile()  # el registro apunta a un modelo que no existe
        window = self.build()
        window.home.refresh(ready)
        pump(lambda: not window.home._inspectors)
        pump(timeout_s=0.1)
        self.assertIn("No se pudo leer", window.home.lbl_notice.text())

    def test_empty_state_when_there_are_no_profiles(self):
        window = self.build()
        with mock.patch("app.home_page.list_profiles", return_value=[]):
            window.home.refresh()
        self.assertTrue(window.home.details.isHidden())
        self.assertFalse(window.home.empty_state.isHidden())

    def test_new_profile_goes_to_calibration(self):
        name = f"zz_shell_{uuid.uuid4().hex[:8]}"
        self.names.append(name)
        window = self.build()

        def fill(dialog):
            dialog.txt_name.setText(name)
            dialog._accept_if_valid()
            return QDialog.Accepted

        with mock.patch.object(NewProfileDialog, "exec", fill):
            window.home.create_profile()
        self.assertIs(window.stack.currentWidget(), window.trainer)
        self.assertEqual(window.trainer.current_profile_name, name)

    def test_deleting_last_profile_forgets_it(self):
        name = self.profile()
        window = self.build(last_profile=name)
        window.home.refresh(name)
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            window.home.delete_selected_profile()
        self.assertIsNone(window.settings["last_profile"])
        self.assertNotIn(name, [window.home.profile_list.item(i).data(0x0100) for i in range(window.home.profile_list.count())])

    def test_settings_are_shared_by_reference_after_voice_config(self):
        window = self.build(camera_index=3)
        shared = window.settings
        dialog = mock.MagicMock()
        dialog.exec.return_value = True
        dialog.get_updated_settings.return_value = {"camera_index": 3, "voice_model_size": "tiny"}
        with mock.patch.object(main_window_module, "VoiceCommandConfigDialog", return_value=dialog), mock.patch.object(QMessageBox, "information"):
            window.open_voice_commands()
        self.assertIs(window.settings, shared)
        self.assertEqual(shared["voice_model_size"], "tiny")

    def test_repeated_errors_show_a_single_notice(self):
        window = self.build()
        with mock.patch("app.main_window.QMessageBox") as box:
            window.show_unexpected_error("uno")
            window.show_unexpected_error("dos")
        self.assertEqual(box.call_count, 1)


class _RealController(assistive_controls.AssistiveController):
    """El controlador real, para probar métodos sueltos sin cámara."""

    def __init__(self):  # pragma: no cover - se crea con object.__new__
        raise AssertionError


class ControllerLifecycleTests(unittest.TestCase):
    def _controller(self):
        controller = object.__new__(_RealController)
        controller._stopped = False
        controller.running = True
        controller.state = AppState.READY
        controller.emergency_hotkey = mock.MagicMock()
        controller.usage_recorder = None
        controller._usage_hotkeys = []
        controller.control_thread = None
        controller.audio_level_monitor = mock.MagicMock()
        controller.provider = mock.MagicMock()
        controller.tts = None
        controller.logger = mock.MagicMock()
        controller.main_gui_interface = None
        return controller

    def test_stop_is_idempotent(self):
        controller = self._controller()
        controller.stop()
        controller.stop()
        controller.provider.close.assert_called_once()
        controller.logger.export_excel.assert_called_once()
        self.assertEqual(controller.state, AppState.STOPPED)

    def test_stop_works_if_never_started(self):
        controller = self._controller()
        controller.running = False
        controller.stop()
        controller.provider.close.assert_called_once()

    def test_stop_does_not_close_provider_while_camera_loop_is_stuck(self):
        controller = self._controller()
        release = threading.Event()
        controller.control_thread = threading.Thread(target=release.wait, daemon=True)
        controller.control_thread.start()
        with mock.patch.object(assistive_controls, "CONTROL_THREAD_STOP_TIMEOUT_S", 0.05):
            controller.stop()
        controller.provider.close.assert_not_called()
        release.set()

    def test_camera_is_released_by_its_own_thread_even_after_errors(self):
        controller = self._controller()
        controller.cap = mock.MagicMock()
        controller._run_camera_loop = mock.MagicMock(side_effect=RuntimeError("fallo"))
        with self.assertRaises(RuntimeError):
            controller._main_loop()
        controller.cap.release.assert_called_once()

    def test_whisper_model_is_shared_between_sessions(self):
        loads = []
        with mock.patch.object(assistive_controls, "WhisperModel", lambda *a, **k: loads.append(a) or object()), mock.patch.object(assistive_controls, "_whisper_models", {}):
            first = assistive_controls.get_shared_whisper("tiny")
            second = assistive_controls.get_shared_whisper("tiny")
        self.assertIs(first, second)
        self.assertEqual(len(loads), 1)


class CameraThreadTests(unittest.TestCase):
    def test_stop_during_slow_open_does_not_hang(self):
        opened = threading.Event()

        class SlowCapture:
            def __init__(self, index):
                opened.set()
                time.sleep(0.2)

            def set(self, *args):
                return True

            def isOpened(self):
                return True

            def read(self):
                return False, None

            def release(self):
                pass

        with mock.patch.object(qt_helpers.cv2, "VideoCapture", SlowCapture), mock.patch.object(qt_helpers, "LandmarkProvider", mock.MagicMock()):
            thread = qt_helpers.CameraThread()
            thread.start()
            self.assertTrue(opened.wait(3))
            started = time.monotonic()
            thread.stop()
            self.assertLess(time.monotonic() - started, 3)
            self.assertTrue(thread.isFinished())

    def test_retries_when_camera_is_still_busy_then_reports(self):
        attempts = []

        class BusyCapture:
            def __init__(self, index):
                attempts.append(index)

            def set(self, *args):
                return True

            def isOpened(self):
                return False

            def release(self):
                pass

        errors = []
        with mock.patch.object(qt_helpers.cv2, "VideoCapture", BusyCapture), mock.patch.object(qt_helpers, "LandmarkProvider", mock.MagicMock()), mock.patch.object(qt_helpers.CameraThread, "OPEN_RETRY_S", 0.01):
            thread = qt_helpers.CameraThread()
            thread.camera_error.connect(errors.append)
            thread.run()
        self.assertEqual(len(attempts), qt_helpers.CameraThread.OPEN_ATTEMPTS)
        self.assertEqual(len(errors), 1)


class ProcessServicesTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)

    def test_second_instance_is_refused_until_first_is_released(self):
        with mock.patch.object(app_runtime, "LOCK_PATH", self.folder / "gia.lock"), mock.patch.object(app_runtime, "DATA_DIR", self.folder):
            first = app_runtime.acquire_single_instance_lock()
            self.assertIsInstance(first, QLockFile)
            self.assertIsNone(app_runtime.acquire_single_instance_lock())
            first.unlock()
            again = app_runtime.acquire_single_instance_lock()
            self.assertIsNotNone(again)
            again.unlock()

    def test_unexpected_errors_are_logged_and_reported(self):
        log_path = self.folder / "errores.log"
        with mock.patch.object(app_runtime, "ERROR_LOG_PATH", log_path):
            reporter = app_runtime.ErrorReporter()
            received = []
            reporter.error_occurred.connect(received.append)
            try:
                raise ValueError("algo falló")
            except ValueError as exc:
                with mock.patch("sys.stderr"):
                    reporter._on_exception(type(exc), exc, exc.__traceback__)
        self.assertEqual(received, ["ValueError: algo falló"])
        self.assertIn("algo falló", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
