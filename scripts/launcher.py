#!/usr/bin/env python3
"""PySide6 native desktop launcher for llm-pick-and-place.

Native Qt6 app — no browser, no server required.
Delegates all simulation work to scripts/run_sim.py via subprocess.Popen.
Never imports Isaac / Omni modules.

Usage:
    python3 scripts/launcher.py
    pip install 'llm-pick-and-place[ui]'   # installs PySide6
"""
from __future__ import annotations

import html
import json
import os
import re
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

try:
    from PySide6.QtCore import (
        QSize,
        QThread,
        Qt,
        QTimer,
        Signal,
    )
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QSizePolicy,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print(
        "PySide6 is not installed.\n"
        "Install with:  pip install 'llm-pick-and-place[ui]'\n"
        "  or:          pip install PySide6",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parent.parent
_RUN_SIM = _ROOT / "scripts" / "run_sim.py"

# ── Find Isaac Sim python.sh at startup (result cached module-level) ──────────
def _find_isaac_python() -> Optional[Path]:
    for candidate in [
        os.environ.get("ISAACSIM_PATH", ""),
        os.path.expanduser("~/isaacsim"),
        "/home/user/isaacsim",
        "/opt/isaacsim",
    ]:
        if candidate:
            p = Path(candidate) / "python.sh"
            if p.exists():
                return p
    return None

_ISAAC_PY    = _find_isaac_python()
_ISAACLAB_SH = Path("~/IsaacLab/isaaclab.sh").expanduser()

# ── Load project metadata (no Isaac/Omni imports) ─────────────────────────────
sys.path.insert(0, str(_ROOT))

try:
    from llm_manip.robots import ISAAC_SUPPORTED_ROBOTS
    _ISAAC_ROBOTS: list[str] = sorted(ISAAC_SUPPORTED_ROBOTS)
except ImportError:
    _ISAAC_ROBOTS = ["panda"]

try:
    from llm_manip.scenes import SCENES as _SCENE_CFG
    _SCENE_NAMES: list[str] = list(_SCENE_CFG.keys())
except ImportError:
    _SCENE_NAMES = ["two_objects", "stack_three", "cluttered"]

# ── UI constants ──────────────────────────────────────────────────────────────
_PRESETS: list[tuple[str, str]] = [
    ("빨간 → 파란",    "put the red cube on the blue cube"),
    ("빨간 집기",      "빨간 박스를 집어"),
    ("가장 가까운 것", "pick up the nearest cube"),
]

_PIPELINE_TAGS = (
    "[IsaacEnv", "[RESET]", "[PERCEIVE]", "[PLAN]", "[EXEC]",
    "[PickSkill", "[PlaceSkill", "[MoveToSkill", "[DONE]",
    "[LlmPlanner", "[Orchestrator", "[run_sim]",
    "success=", "ERROR:", "WARNING:",
)

def _is_pipeline(line: str) -> bool:
    s = line.strip()
    return any(s.startswith(t) for t in _PIPELINE_TAGS)


# ════════════════════════════════════════════════════════════════════════════════
# Preflight check functions — pure Python, called from QThread
# ════════════════════════════════════════════════════════════════════════════════

def _chk_gpu() -> tuple[str, str]:
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        return "fail", "nvidia-smi를 찾을 수 없습니다. NVIDIA 드라이버를 설치하세요."
    except subprocess.TimeoutExpired:
        return "warn", "nvidia-smi 응답 시간 초과."
    if r.returncode != 0:
        return "fail", "nvidia-smi 실패. NVIDIA 드라이버를 확인하세요."
    gpus = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append(f"{parts[0]}, {parts[1]}, 드라이버 {parts[2]}")
    return ("ok", " | ".join(gpus)) if gpus else ("fail", "GPU 정보를 파싱할 수 없습니다.")


def _chk_isaac() -> tuple[str, str]:
    if _ISAAC_PY is None:
        return "fail", (
            "Isaac Sim을 찾을 수 없습니다. "
            "ISAACSIM_PATH 환경변수를 설정하거나 ~/isaacsim 에 설치하세요."
        )
    msg = str(_ISAAC_PY)
    if not _ISAACLAB_SH.exists():
        return "warn", msg + "  |  IsaacLab 미발견 (~/IsaacLab/isaaclab.sh)"
    return "ok", msg + f"  |  {_ISAACLAB_SH}"


def _chk_ollama() -> tuple[str, str]:
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=3
        ) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        if not models:
            return "warn", "Ollama 실행 중이지만 모델이 없습니다.  실행: ollama pull llama3.2:1b"
        return "ok", "실행 중  |  모델: " + ", ".join(models)
    except urllib.error.URLError:
        return "fail", "Ollama가 실행되지 않습니다.  터미널에서 실행: ollama serve"
    except Exception as e:
        return "warn", f"Ollama 확인 오류: {e}"


def _chk_openai_in_isaac() -> tuple[str, str]:
    if _ISAAC_PY is None:
        return "warn", "Isaac Sim 미설치 — 건너뜀"
    try:
        r = subprocess.run(
            [str(_ISAAC_PY), "-c",
             "import openai; print(openai.__version__)"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            return "ok", f"openai {r.stdout.strip()} (Isaac Python)"
        return "warn", (
            f"openai가 Isaac Python에 없습니다. "
            f"설치: {_ISAAC_PY} -m pip install openai"
        )
    except subprocess.TimeoutExpired:
        return "warn", "확인 시간 초과 (Isaac Python 시작 지연)"
    except Exception as exc:
        return "warn", str(exc)


def _fetch_ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=3
        ) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ════════════════════════════════════════════════════════════════════════════════
# Background QThread workers
# ════════════════════════════════════════════════════════════════════════════════

class _PreflightWorker(QThread):
    result = Signal(str, str)   # (status, message)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            status, msg = self._fn()
        except Exception as exc:
            status, msg = "fail", str(exc)
        self.result.emit(status, msg)


class _OllamaFetcher(QThread):
    done = Signal(list)

    def run(self):
        self.done.emit(_fetch_ollama_models())


# ════════════════════════════════════════════════════════════════════════════════
# PreflightRow: coloured dot + label + message
# ════════════════════════════════════════════════════════════════════════════════
_DOT_COLORS = {
    "ok":   "#22c55e",   # green-500  (semantic status)
    "warn": "#eab308",   # yellow-500
    "fail": "#ef4444",   # red-500
    "pend": "#a1a1aa",   # zinc-400
}

class _PreflightRow(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        lay.setSpacing(8)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(16)
        lay.addWidget(self._dot)

        lbl = QLabel(label)
        lbl.setFixedWidth(130)
        lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(lbl)

        self._msg = QLabel("확인 중…")
        self._msg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._msg.setWordWrap(True)
        lay.addWidget(self._msg, 1)

        self.status = "pend"
        self._paint()

    def update(self, status: str, message: str):
        self.status = status
        self._msg.setText(message)
        self._msg.setToolTip(message)
        self._paint()

    def _paint(self):
        c = _DOT_COLORS.get(self.status, _DOT_COLORS["pend"])
        self._dot.setStyleSheet(f"color: {c}; font-size: 11px; background: transparent;")


# ════════════════════════════════════════════════════════════════════════════════
# PreflightPanel: four checks + Re-check button
# ════════════════════════════════════════════════════════════════════════════════

class PreflightPanel(QGroupBox):
    changed = Signal(bool, bool)   # (gpu_ok, isaac_ok) — drives Execute gate

    def __init__(self, parent=None):
        super().__init__("Preflight 점검", parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(2)

        self._rows: dict[str, _PreflightRow] = {
            "gpu":    _PreflightRow("GPU"),
            "isaac":  _PreflightRow("Isaac Sim"),
            "ollama": _PreflightRow("Ollama"),
            "openai": _PreflightRow("openai (Isaac)"),
        }
        for row in self._rows.values():
            lay.addWidget(row)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn = QPushButton("다시 점검")
        self._btn.setObjectName("recheck")
        self._btn.clicked.connect(self.run_checks)
        btn_row.addWidget(self._btn)
        lay.addLayout(btn_row)

        self._workers: list[_PreflightWorker] = []

    def run_checks(self):
        for w in self._workers:
            w.terminate()
        self._workers.clear()
        for row in self._rows.values():
            row.update("pend", "확인 중…")

        fns: dict[str, object] = {
            "gpu":    _chk_gpu,
            "isaac":  _chk_isaac,
            "ollama": _chk_ollama,
            "openai": _chk_openai_in_isaac,
        }
        for key, fn in fns.items():
            w = _PreflightWorker(fn, self)
            w.result.connect(lambda st, msg, k=key: self._done(k, st, msg))
            self._workers.append(w)
            w.start()

    def _done(self, key: str, status: str, msg: str):
        self._rows[key].update(status, msg)
        self.changed.emit(self.gpu_ok, self.isaac_ok)

    @property
    def gpu_ok(self) -> bool:
        return self._rows["gpu"].status in ("ok", "warn")

    @property
    def isaac_ok(self) -> bool:
        return self._rows["isaac"].status == "ok"


# ════════════════════════════════════════════════════════════════════════════════
# LogPanel: fixed-height scrollable area with pipeline filter
# ════════════════════════════════════════════════════════════════════════════════

def _line_color(line: str) -> str:
    s = line.strip()
    if s.startswith("ERROR"):      return "#dc2626"   # red-600
    if s.startswith("WARNING"):    return "#d97706"   # amber-600
    if "success=True" in s:       return "#16a34a"   # green-600
    if "success=False" in s:      return "#dc2626"   # red-600
    if "[DONE]" in s:             return "#16a34a"   # green-600
    return "#3f3f46"                                   # zinc-700 (readable on #fafafa)


class LogPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("로그", parent)
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addStretch()
        self._full_cb = QCheckBox("전체 로그")
        self._full_cb.setToolTip(
            "파이프라인 이벤트([PERCEIVE]/[PLAN]/[EXEC]/…)만 표시합니다.\n"
            "전체 로그 ON: Isaac 부팅 메시지 포함 전체 출력."
        )
        self._full_cb.toggled.connect(self._redraw)
        top.addWidget(self._full_cb)
        lay.addLayout(top)

        self._edit = QTextEdit()
        self._edit.setReadOnly(True)
        self._edit.setMinimumHeight(200)
        mono = QFont("Courier New", 11)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._edit.setFont(mono)
        lay.addWidget(self._edit, 1)

        self._all: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────
    def clear(self):
        self._all.clear()
        self._edit.clear()

    def append_line(self, raw: str):
        for line in raw.rstrip("\n").splitlines():
            self._all.append(line)
            if self._full_cb.isChecked() or _is_pipeline(line):
                self._insert(line)

    @property
    def all_lines(self) -> list[str]:
        return self._all

    # ── Internal ──────────────────────────────────────────────────────────────
    def _insert(self, line: str):
        color   = _line_color(line)
        escaped = html.escape(line)
        self._edit.append(
            f'<span style="color:{color};'
            f'font-family:\'Courier New\',monospace;font-size:11px">'
            f'{escaped}</span>'
        )
        sb = self._edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _redraw(self):
        self._edit.clear()
        show_all = self._full_cb.isChecked()
        for line in self._all:
            if show_all or _is_pipeline(line):
                self._insert(line)


# ════════════════════════════════════════════════════════════════════════════════
# Sim process reader thread
# ════════════════════════════════════════════════════════════════════════════════

class _ProcReader(QThread):
    """Read a subprocess.Popen's stdout line by line on a background thread."""
    line_ready    = Signal(str)
    finished_with = Signal(int)   # exit code

    def __init__(self, proc: "subprocess.Popen[str]", parent=None):
        super().__init__(parent)
        self._proc = proc

    def run(self):
        for line in self._proc.stdout:
            self.line_ready.emit(line.rstrip("\n"))
        code = self._proc.wait()
        self.finished_with.emit(code if code is not None else -1)


# ════════════════════════════════════════════════════════════════════════════════
# MainWindow
# ════════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("llm-pick-and-place Launcher")
        self.setMinimumSize(QSize(960, 700))

        self._proc:     Optional[subprocess.Popen] = None
        self._reader:   Optional[_ProcReader]     = None
        self._sim_pgid: Optional[int]             = None   # PGID for killpg

        cw = QWidget()
        cw.setObjectName("page")
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # ── App header ────────────────────────────────────────────────────────
        root.addWidget(self._build_header())

        # ── Row 1: Preflight ──────────────────────────────────────────────────
        self._pf = PreflightPanel()
        self._pf.changed.connect(self._gate_execute)
        self._pf.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        root.addWidget(self._pf)

        # ── Row 2: Config (left) + Execute (right) ────────────────────────────
        mid = QWidget()
        mid.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        mid_lay = QHBoxLayout(mid)
        mid_lay.setContentsMargins(0, 0, 0, 0)
        mid_lay.setSpacing(16)
        root.addWidget(mid)

        mid_lay.addWidget(self._build_config(), 1)
        mid_lay.addWidget(self._build_execute(), 2)

        # ── Row 3: Log (expands to fill remaining space) ──────────────────────
        self._log = LogPanel()
        root.addWidget(self._log, 1)

        # ── Status bar (bottom) ───────────────────────────────────────────────
        self.statusBar().showMessage("준비")

        # Kick off async work after the event loop starts
        QTimer.singleShot(100, self._pf.run_checks)
        QTimer.singleShot(400, self._load_ollama_models)

    # ── App header ────────────────────────────────────────────────────────────
    def _build_header(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)

        title = QLabel("Robot Pick & Place")
        title.setObjectName("app_title")
        lay.addWidget(title)

        sub = QLabel("llm-pick-and-place  ·  Isaac Sim 5.1  ·  IsaacLab 2.3")
        sub.setObjectName("app_subtitle")
        lay.addWidget(sub)

        return w

    # ── Config group ──────────────────────────────────────────────────────────
    def _build_config(self) -> QGroupBox:
        g = QGroupBox("설정")
        lay = QVBoxLayout(g)
        lay.setSpacing(10)

        def _row(label: str, widget: QWidget, lw: int = 70):
            h = QHBoxLayout()
            lb = QLabel(label)
            lb.setFixedWidth(lw)
            h.addWidget(lb)
            h.addWidget(widget, 1)
            lay.addLayout(h)

        self._robot_cb = QComboBox()
        self._robot_cb.addItems(_ISAAC_ROBOTS)
        _row("Robot", self._robot_cb)

        self._scene_cb = QComboBox()
        self._scene_cb.addItems(_SCENE_NAMES)
        _default_scene = next(
            (s for s in ("tabletop_rb",) if s in _SCENE_NAMES),
            _SCENE_NAMES[0] if _SCENE_NAMES else "tabletop_rb",
        )
        self._scene_cb.setCurrentText(_default_scene)
        _row("Scene", self._scene_cb)

        self._model_cb = QComboBox()
        self._model_cb.setEnabled(False)   # enabled once Ollama models are loaded
        _row("Model", self._model_cb)

        lay.addStretch()
        return g

    # ── Execute group ─────────────────────────────────────────────────────────
    def _build_execute(self) -> QGroupBox:
        g = QGroupBox("실행")
        lay = QVBoxLayout(g)
        lay.setSpacing(10)

        # Section label
        lbl = QLabel("명령어")
        lbl.setObjectName("section_label")
        lay.addWidget(lbl)

        # Instruction input (prominent)
        self._instr_edit = QLineEdit()
        self._instr_edit.setObjectName("instr_input")
        self._instr_edit.setPlaceholderText("예) put the red cube on the blue cube")
        self._instr_edit.textChanged.connect(
            lambda _: self._gate_execute(self._pf.gpu_ok, self._pf.isaac_ok)
        )
        lay.addWidget(self._instr_edit)

        # Preset chips
        chips = QHBoxLayout()
        chips.setSpacing(6)
        chips.setContentsMargins(0, 2, 0, 0)
        for chip_label, instruction in _PRESETS:
            btn = QPushButton(chip_label)
            btn.setObjectName("preset_chip")
            btn.setToolTip(instruction)
            btn.clicked.connect(lambda _, inst=instruction: self._instr_edit.setText(inst))
            chips.addWidget(btn)
        chips.addStretch()
        lay.addLayout(chips)

        lay.addStretch()

        # [▶ Execute | ■ Stop]
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_exec = QPushButton("▶  Execute")
        self._btn_exec.setObjectName("execute")
        self._btn_exec.setEnabled(False)
        self._btn_exec.clicked.connect(self._do_execute)
        btn_row.addWidget(self._btn_exec, 3)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setObjectName("stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setToolTip("Isaac Sim을 종료합니다 (SIGTERM → SIGKILL).")
        self._btn_stop.clicked.connect(self._do_stop)
        btn_row.addWidget(self._btn_stop, 2)

        lay.addLayout(btn_row)
        return g

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _assemble_cmd(self) -> list[str]:
        instr = self._instr_edit.text().strip() or "…"
        return [
            sys.executable, str(_RUN_SIM),
            "--robot",       self._robot_cb.currentText(),
            "--scene",       self._scene_cb.currentText(),
            "--perception",  "oracle",
            "--planner",     "llm",
            "--executor",    "ik",
            "--instruction", instr,
        ]

    def _gate_execute(self, gpu_ok: bool, isaac_ok: bool):
        ready     = gpu_ok and isaac_ok
        has_instr = bool(self._instr_edit.text().strip())
        self._btn_exec.setEnabled(ready and self._proc is None and has_instr)
        tips: list[str] = []
        if not gpu_ok:
            tips.append("GPU 점검 실패")
        if not isaac_ok:
            tips.append("Isaac Sim 미발견")
        if not has_instr:
            tips.append("명령 입력 필요")
        self._btn_exec.setToolTip("실행 불가: " + ", ".join(tips) if tips else "")

    def _load_ollama_models(self):
        fetcher = _OllamaFetcher(self)
        fetcher.done.connect(self._on_ollama_models)
        fetcher.start()

    def _on_ollama_models(self, models: list):
        self._model_cb.clear()
        if models:
            self._model_cb.addItems(models)
            if "llama3.2:1b" in models:
                self._model_cb.setCurrentText("llama3.2:1b")
            self._model_cb.setEnabled(True)
        else:
            self._model_cb.addItem("(모델 없음)")
            self._model_cb.setEnabled(False)

    # ── Execute / Stop ────────────────────────────────────────────────────────
    def _do_execute(self):
        instr = self._instr_edit.text().strip()
        if not instr:
            self.statusBar().showMessage("명령(instruction)을 입력하세요.")
            return

        cmd = self._assemble_cmd()
        cmd[-1] = instr   # replace placeholder with actual instruction

        self._log.clear()

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        model = self._model_cb.currentText()
        if model and model != "(모델 없음)":
            env["OLLAMA_MODEL"] = model

        try:
            proc = subprocess.Popen(
                cmd,
                start_new_session=True,   # child gets own session → PGID = proc.pid
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as e:
            self.statusBar().showMessage(f"프로세스 시작 실패: {e}")
            return

        self._proc     = proc
        self._sim_pgid = proc.pid   # new session leader: SID = PGID = PID

        self._reader = _ProcReader(proc, self)
        self._reader.line_ready.connect(self._on_line)
        self._reader.finished_with.connect(self._on_finished)
        self._reader.start()

        self._btn_exec.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self.statusBar().showMessage("실행 중…")

    def _do_stop(self):
        """Stop Isaac Sim: SIGTERM to process group + pkill safety net, SIGKILL after 2 s."""
        if self._proc is None and self._sim_pgid is None:
            return
        self.statusBar().showMessage("Isaac Sim 종료 중…")
        self._btn_stop.setEnabled(False)
        if self._sim_pgid is not None:
            try:
                os.killpg(self._sim_pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        subprocess.run(["pkill", "-TERM", "-f", "scripts/run_sim.py"],
                       capture_output=True)
        QTimer.singleShot(2000, self._force_kill_sim)

    def _force_kill_sim(self):
        """SIGKILL to process group + pkill -9 if still alive after SIGTERM."""
        if self._sim_pgid is not None:
            try:
                os.killpg(self._sim_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        subprocess.run(["pkill", "-9", "-f", "scripts/run_sim.py"],
                       capture_output=True)
        self._sim_pgid = None

    def _on_line(self, line: str):
        self._log.append_line(line)

    def _on_finished(self, exit_code: int):
        self._proc     = None
        self._reader   = None
        self._sim_pgid = None
        has_instr = bool(self._instr_edit.text().strip())
        self._btn_exec.setEnabled(self._pf.gpu_ok and self._pf.isaac_ok and has_instr)
        self._btn_stop.setEnabled(False)

        log_text = "\n".join(self._log.all_lines)
        m = re.search(r"success=(True|False)\s+re-plans=(\d+)", log_text)
        if m:
            ok      = m.group(1) == "True"
            replans = int(m.group(2))
            icon    = "✓" if ok else "✗"
            self.statusBar().showMessage(
                f"{icon} {'완료' if ok else '실패'}  |  re-plans: {replans}"
            )
        elif exit_code != 0:
            self.statusBar().showMessage(f"종료 코드 {exit_code}")
        else:
            self.statusBar().showMessage("완료")


# ════════════════════════════════════════════════════════════════════════════════
# Application stylesheet  (modern light monochrome — design-token pass)
# ════════════════════════════════════════════════════════════════════════════════
_STYLE = """
/* ── Page & reset ──────────────────────────────────────────────────────────── */
QMainWindow { background-color: #f4f4f5; }
/* Central widget sets page bg; all other QWidgets default to transparent */
QWidget#page { background-color: #f4f4f5; }
QWidget {
    color: #18181b;
    font-size: 13px;
    font-family: "Inter", "-apple-system", "Segoe UI", Roboto, sans-serif;
}

/* ── Cards — white rounded panels on the page ──────────────────────────────── */
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 12px;
    margin-top: 0;
    padding: 16px;
    padding-top: 36px;
}
QGroupBox::title {
    subcontrol-origin: border;
    subcontrol-position: top left;
    left: 16px;
    top: 10px;
    color: #71717a;
    font-size: 11px;
    font-weight: 600;
    background-color: transparent;
    padding: 0 2px;
}

/* ── App header ─────────────────────────────────────────────────────────────── */
QLabel#app_title {
    font-size: 20px;
    font-weight: 600;
    color: #18181b;
    background: transparent;
}
QLabel#app_subtitle {
    font-size: 12px;
    color: #71717a;
    background: transparent;
}

/* ── Section label (e.g. "명령어") ─────────────────────────────────────────── */
QLabel#section_label {
    font-size: 11px;
    font-weight: 700;
    color: #52525b;
    background: transparent;
}

/* ── Instruction input (prominent) ─────────────────────────────────────────── */
QLineEdit#instr_input {
    min-height: 48px;
    font-size: 14px;
    padding: 11px 14px;
    border-radius: 10px;
    border: 1.5px solid #d4d4d8;
}
QLineEdit#instr_input:focus {
    border: 2px solid #18181b;
}

/* ── Preset chips ───────────────────────────────────────────────────────────── */
QPushButton#preset_chip {
    background-color: #f4f4f5;
    border: 1px solid #e4e4e7;
    border-radius: 14px;
    padding: 0 12px;
    color: #52525b;
    font-size: 12px;
    font-weight: 400;
    min-height: 28px;
    max-height: 28px;
}
QPushButton#preset_chip:hover {
    background-color: #e4e4e7;
    border-color: #a1a1aa;
    color: #18181b;
}
QPushButton#preset_chip:pressed { background-color: #d4d4d8; }

/* ── Labels & checkboxes (transparent so card bg shows through) ─────────────── */
QLabel { background-color: transparent; color: #18181b; }
QCheckBox { color: #3f3f46; spacing: 6px; background: transparent; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    background-color: #ffffff;
    border: 1px solid #d4d4d8;
    border-radius: 4px;
}
QCheckBox::indicator:hover   { border-color: #a1a1aa; }
QCheckBox::indicator:checked { background-color: #18181b; border-color: #18181b; }

/* ── Form controls ──────────────────────────────────────────────────────────── */
QComboBox, QLineEdit {
    background-color: #ffffff;
    border: 1px solid #d4d4d8;
    border-radius: 8px;
    padding: 6px 10px;
    color: #18181b;
    min-height: 34px;
    selection-background-color: #e4e4e7;
    selection-color: #18181b;
}
QComboBox:focus, QLineEdit:focus { border-color: #18181b; }
QComboBox:disabled, QLineEdit:disabled {
    background-color: #f4f4f5; color: #a1a1aa; border-color: #e4e4e7;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #d4d4d8;
    border-radius: 8px;
    selection-background-color: #f4f4f5;
    selection-color: #18181b;
    outline: none;
    padding: 4px;
}

/* ── Buttons — base ─────────────────────────────────────────────────────────── */
QPushButton {
    background-color: #ffffff;
    border: 1px solid #d4d4d8;
    border-radius: 8px;
    padding: 10px 16px;
    color: #3f3f46;
    min-height: 40px;
    font-size: 13px;
    font-weight: 500;
}
QPushButton:hover   { background-color: #f4f4f5; border-color: #a1a1aa; }
QPushButton:pressed { background-color: #e4e4e7; }
QPushButton:disabled { background-color: #f4f4f5; color: #a1a1aa; border-color: #e4e4e7; }

/* Execute — primary, maximum contrast, no hue */
QPushButton#execute {
    background-color: #18181b;
    border-color: #18181b;
    color: #ffffff;
    font-weight: 600;
    min-height: 44px;
}
QPushButton#execute:hover   { background-color: #27272a; border-color: #27272a; }
QPushButton#execute:pressed { background-color: #3f3f46; }
QPushButton#execute:disabled {
    background-color: #e4e4e7; border-color: #e4e4e7;
    color: #a1a1aa; font-weight: 500;
}

/* Stop — secondary outline */
QPushButton#stop {
    background-color: transparent;
    border: 1px solid #d4d4d8;
    color: #3f3f46;
    font-weight: 500;
    min-height: 44px;
}
QPushButton#stop:hover    { background-color: #f4f4f5; border-color: #a1a1aa; }
QPushButton#stop:pressed  { background-color: #e4e4e7; }
QPushButton#stop:disabled { background-color: transparent; color: #a1a1aa; border-color: #e4e4e7; }

/* 다시 점검 — ghost, small */
QPushButton#recheck {
    background-color: transparent;
    border: 1px solid #e4e4e7;
    color: #71717a;
    min-height: 28px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 400;
    border-radius: 6px;
}
QPushButton#recheck:hover   { background-color: #f4f4f5; color: #3f3f46; border-color: #d4d4d8; }
QPushButton#recheck:pressed { background-color: #e4e4e7; }

/* ── Log (QTextEdit) ────────────────────────────────────────────────────────── */
QTextEdit {
    background-color: #fafafa;
    color: #3f3f46;
    border: 1px solid #e4e4e7;
    border-radius: 8px;
    padding: 10px;
    font-family: "JetBrains Mono", Menlo, Consolas, monospace;
    font-size: 11px;
    selection-background-color: #e4e4e7;
    selection-color: #18181b;
}

/* ── Scrollbars ─────────────────────────────────────────────────────────────── */
QScrollBar:vertical   { background: transparent; width: 8px; margin: 2px 0; }
QScrollBar:horizontal { background: transparent; height: 8px; margin: 0 2px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #d4d4d8; border-radius: 4px;
}
QScrollBar::handle:vertical   { min-height: 24px; }
QScrollBar::handle:horizontal { min-width: 24px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: #a1a1aa; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::corner { background: transparent; }

/* ── Status bar ─────────────────────────────────────────────────────────────── */
QStatusBar {
    background-color: #f4f4f5;
    color: #71717a;
    font-size: 12px;
    border-top: 1px solid #e4e4e7;
}
QStatusBar::item { border: none; }

/* ── Tooltip ────────────────────────────────────────────────────────────────── */
QToolTip {
    background-color: #18181b;
    color: #f4f4f5;
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}
"""


# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(_STYLE)
    app.setApplicationName("llm-pick-and-place Launcher")

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
