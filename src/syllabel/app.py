import sys
import os
import glob
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import re
import matplotlib
from matplotlib import colors as mcolors
import matplotlib.cm as cm

from PySide6 import QtCore, QtGui, QtWidgets

import librosa

try:
    from .propose_labels import make_proposals  # type: ignore
except Exception:
    make_proposals = None  # type: ignore


@dataclass
class Syllable:
    start: float
    stop: float
    name: str
    patch: Optional[object] = None  # QGraphicsRectItem
    label_artist: Optional[object] = None  # QGraphicsSimpleTextItem
    selected: bool = False
    top_line: Optional[object] = None  # QGraphicsLineItem
    bottom_line: Optional[object] = None  # QGraphicsLineItem


def natural_name_key(s: str) -> Tuple[int, int, str]:
    if not isinstance(s, str):
        return (1, 0, str(s))
    m = re.fullmatch(r"[Ll](\d+)", s.strip())
    if m:
        try:
            return (0, int(m.group(1)), s.lower())
        except Exception:
            pass
    return (1, 0, s.lower())


def _contrasting_text_qcolor(r: int, g: int, b: int) -> QtGui.QColor:
    try:
        yiq = (299 * r + 587 * g + 114 * b) / 1000.0
    except Exception:
        yiq = 128.0
    return QtGui.QColor(0, 0, 0) if yiq > 160 else QtGui.QColor(255, 255, 255)


def list_wavs_and_annotations(root: str) -> List[Tuple[str, Optional[str]]]:
    wavs = sorted(glob.glob(os.path.join(root, "*.wav")))
    pairs: List[Tuple[str, Optional[str]]] = []
    for wav in wavs:
        base = os.path.splitext(os.path.basename(wav))[0]
        ann = os.path.join(root, f"{base}_annotations.csv")
        if os.path.exists(ann):
            pairs.append((wav, ann))
        else:
            pairs.append((wav, None))
    return pairs


def load_annotations(path: Optional[str]) -> List[Syllable]:
    if path is None:
        return []
    if not os.path.exists(path):
        return []
    sylls: List[Syllable] = []
    try:
        df = pd.read_csv(path)
    except Exception:
        return sylls
    # Normalize possible column names
    cols = {c.lower(): c for c in df.columns}
    start_col = cols.get("start_seconds") or cols.get("start")
    stop_col = cols.get("stop_seconds") or cols.get("stop")
    name_col = cols.get("name")
    if not start_col or not stop_col:
        return sylls
    for _, row in df.iterrows():
        try:
            start = float(row[start_col])
            stop = float(row[stop_col])
            name = str(row[name_col]) if name_col and pd.notna(row[name_col]) else ""
            if stop <= start:
                continue
            sylls.append(Syllable(start=start, stop=stop, name=name))
        except Exception:
            continue
    return sylls


def fft_spectrogram(y: np.ndarray, sr: int, n_fft: int = 512, hop_length: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frames = np.arange(S_db.shape[1])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    return S_db, freqs, times


class SpectrogramRow(QtWidgets.QGraphicsView):
    selection_changed = QtCore.Signal()

    def __init__(self, wav_path: str, annotations: List[Syllable], color_map: Dict[str, Tuple[float, float, float]], parent=None, y_limits: Optional[Tuple[float, float]] = None, x_limits: Optional[Tuple[float, float]] = None, show_xticks: bool = True, default_name_fn: Optional[Callable[[], str]] = None, time_offset: float = 0.0):
        super().__init__(parent)
        self.setObjectName("SpectrogramRowView")
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        # Fix row height so each spectrogram is 150 px tall
        self.setFixedHeight(150)
        # Black background
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))

        self.wav_path = wav_path
        self.annotations: List[Syllable] = annotations
        self.color_map = color_map
        self.y_limits = y_limits  # (min_freq, max_freq) in Hz; if None, use local
        self.x_limits = x_limits  # (t0, t1) in seconds for shared time axis
        self.show_xticks = show_xticks  # unused in graphics view; kept for API compat
        self.default_name_fn = default_name_fn
        self.edit_mode: bool = False
        self.time_offset: float = float(time_offset)

        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
        self.setScene(self.scene)
        # Ensure we get hover move events without pressing a mouse button
        self.viewport().setMouseTracking(True)
        self.setMouseTracking(True)

        self._drag_active = False
        self._drag_x0: Optional[float] = None
        self._drag_rect_item: Optional[QtWidgets.QGraphicsRectItem] = None
        self._drag_mode: Optional[str] = None  # 'resize_left', 'resize_right', 'move'
        self._drag_syll: Optional[Syllable] = None
        self._drag_move_offset: float = 0.0
        self._lasso_add: bool = False
        # New syllable creation state
        self._adding: bool = False
        self._new_syll: Optional[Syllable] = None

        # Spectrogram content
        self._pixmap_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self._qimage_ref: Optional[QtGui.QImage] = None  # keep ref to buffer
        self._ymax_khz: float = 1.0
        self._duration: float = 0.0
        # Cached spectrogram/time for alignment on a uniform dt grid
        self._S_db: Optional[np.ndarray] = None         # shape (F, T)
        self._times: Optional[np.ndarray] = None        # shape (T,)
        self._spec_uniform: Optional[np.ndarray] = None # shape (F, Tu)
        self._env_dt: float = 0.01  # seconds (common time grid for alignment)

        self._build_view()

    def _build_view(self):
        # Title
        title_item = self.scene.addSimpleText(os.path.basename(self.wav_path))
        title_item.setBrush(QtGui.QBrush(QtGui.QColor('white')))
        title_item.setPos(0.0, -12.0)  # small offset above if visible

        # Load audio + compute spectrogram image
        y, sr = librosa.load(self.wav_path, sr=None, mono=True)
        S_db, freqs, times = fft_spectrogram(y, sr)
        duration = float(times[-1]) if times.size > 0 else float(len(y))/float(sr)
        self._duration = duration
        # y limits in kHz
        if self.y_limits is not None:
            ymax_hz = float(self.y_limits[1])
        else:
            ymax_hz = float(freqs[-1]) if freqs.size > 0 else sr/2.0
        self._ymax_khz = ymax_hz / 1000.0

        # Normalize S_db to 0..1 and map to turbo
        S_norm = (S_db - S_db.min()) / max(1e-9, (S_db.max() - S_db.min()))
        cmap = cm.get_cmap('turbo')
        rgba = (cmap(S_norm)[:, :, :3] * 255).astype(np.uint8)
        h, w, _ = rgba.shape
        # Create QImage and QPixmap
        qimg = QtGui.QImage(rgba.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        qimg = qimg.copy()  # own the buffer
        self._qimage_ref = qimg
        pix = QtGui.QPixmap.fromImage(qimg)
        self._pixmap_item = self.scene.addPixmap(pix)
        # Scale pixmap so that its scene size equals (duration seconds) x (ymax_khz)
        # Invert vertically so low frequencies are at the bottom, high at the top.
        sx = (duration / w) if w > 0 else 1.0
        sy = (self._ymax_khz / h) if h > 0 else 1.0
        self._pixmap_item.setTransform(QtGui.QTransform().scale(sx, -sy))
        self._pixmap_item.setPos(self.time_offset, self._ymax_khz)

        # Cache spectrogram and a time-uniform resampling for alignment
        try:
            self._S_db = np.array(S_db, dtype=float)
            self._times = np.array(times, dtype=float)
            t_raw = self._times if self._times is not None and self._times.size > 0 else np.linspace(0.0, duration, num=self._S_db.shape[1] if self._S_db is not None else 2)
            if t_raw.size < 2:
                t_raw = np.array([0.0, max(1e-6, duration)], dtype=float)
            t_uniform = np.arange(0.0, max(duration, 0.0), self._env_dt, dtype=float)
            if t_uniform.size == 0:
                t_uniform = np.array([0.0], dtype=float)
            if self._S_db is not None and t_raw.size == self._S_db.shape[1]:
                F = int(self._S_db.shape[0])
                Tu = int(t_uniform.size)
                Su = np.empty((F, Tu), dtype=float)
                for f in range(F):
                    Su[f, :] = np.interp(t_uniform, t_raw, self._S_db[f, :])
                self._spec_uniform = Su
            else:
                self._spec_uniform = None
        except Exception:
            self._S_db = None
            self._times = None
            self._spec_uniform = None

        # Draw syllable overlays
        y0, y1 = self._y_limits_khz()
        for syll in self.annotations:
            rect_item = self.scene.addRect(
                QtCore.QRectF(
                    syll.start + self.time_offset,
                    y0,
                    max(0.0, syll.stop - syll.start),
                    y1 - y0,
                ),
                pen=QtGui.QPen(QtGui.QColor('white'), 0.0),
                brush=QtGui.QBrush(QtGui.QColor(255, 255, 255, 60)),
            )
            rect_item.setZValue(1.0)
            # Use the rect coordinates to place the label near the top edge
            r = rect_item.rect()
            x_left, x_right = r.left(), r.right()
            width = max(0.0, x_right - x_left)
            txt_item = self.scene.addSimpleText(syll.name or "")
            # Small font, stable pixel size
            f = txt_item.font()
            f.setPointSizeF(8.0)
            txt_item.setFont(f)
            txt_item.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            txt_item.setZValue(3.0)
            # Position label along the top boundary, slightly inset from the left
            txt_item.setPos(x_left + 0.02 * width, y0 - 0.02 * (y1 - y0))
            # Colored boundary lines aligned to rect edges (top at y0, bottom at y1)
            base_pen = QtGui.QPen(QtGui.QColor('white'))
            base_pen.setWidthF(1.2)
            base_pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
            top_line = self.scene.addLine(x_left, y0, x_right, y0, base_pen)
            bottom_line = self.scene.addLine(x_left, y1, x_right, y1, base_pen)
            top_line.setZValue(2.0)
            bottom_line.setZValue(2.0)
            syll.patch = rect_item
            syll.label_artist = txt_item
            syll.top_line = top_line
            syll.bottom_line = bottom_line
            syll.selected = False
            self._update_patch_style(syll)

        # Initial view to current x_limits
        if self.x_limits is None:
            self.x_limits = (0.0, self._duration)
        self._apply_fit()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_fit()

    def _apply_fit(self):
        y0, y1 = self._y_limits_khz()
        t0, t1 = self._x_limits_seconds()
        rect = QtCore.QRectF(t0, y0, max(1e-6, t1 - t0), max(1e-6, y1 - y0))
        self.fitInView(rect, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)

    # Intercept arrow keys locally to avoid default view panning
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        win = self.window()
        handled = False
        # Cancel in-progress add with Escape
        if key == QtCore.Qt.Key.Key_Escape and self.edit_mode and self._adding:
            self._cancel_new_syllable()
            # restore cursor to default edit cursor
            self.viewport().setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            event.accept()
            return
        if key == QtCore.Qt.Key.Key_Up and hasattr(win, '_zoom_in'):
            win._zoom_in()
            handled = True
        elif key == QtCore.Qt.Key.Key_Down and hasattr(win, '_zoom_out'):
            win._zoom_out()
            handled = True
        elif key == QtCore.Qt.Key.Key_Left and hasattr(win, '_pan_left'):
            win._pan_left()
            handled = True
        elif key == QtCore.Qt.Key.Key_Right and hasattr(win, '_pan_right'):
            win._pan_right()
            handled = True
        if handled:
            event.accept()
            return
        super().keyPressEvent(event)

    # Let normal vertical wheel scroll the list; use horizontal wheel (or Shift+wheel) to pan time
    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        ad = event.angleDelta()
        pd = event.pixelDelta()
        dx = ad.x() if not ad.isNull() else pd.x()
        dy = ad.y() if not ad.isNull() else pd.y()
        mods = event.modifiers() if hasattr(event, 'modifiers') else QtCore.Qt.KeyboardModifier.NoModifier
        win = self.window()

        # Horizontal pan when horizontal wheel is used, or Shift+vertical wheel
        horiz_pan = (dx != 0) or ((mods & QtCore.Qt.KeyboardModifier.ShiftModifier) and dy != 0)
        if hasattr(win, '_pan_left') and hasattr(win, '_pan_right') and horiz_pan:
            base = dx if dx != 0 else dy
            steps = int(base / 120) if base != 0 else 0
            if steps == 0:
                steps = int(round(base / 40.0))
            for _ in range(abs(steps)):
                if steps > 0:
                    win._pan_left()
                else:
                    win._pan_right()
            event.accept()
            return
        event.ignore()

    def _update_patch_style(self, syll: Syllable):
        if not isinstance(syll.patch, QtWidgets.QGraphicsRectItem):
            return
        face = self.color_map.get(syll.name, (1.0, 1.0, 1.0))
        r, g, b = [int(255 * c) for c in face]
        alpha = 90 if syll.selected else 60
        pen_color = QtGui.QColor('yellow') if syll.selected else QtGui.QColor('white')
        syll.patch.setBrush(QtGui.QBrush(QtGui.QColor(r, g, b, alpha)))
        syll.patch.setPen(QtGui.QPen(pen_color, 0.0))
        line_pen = QtGui.QPen(QtGui.QColor(r, g, b))
        line_pen.setWidthF(2.0 if syll.selected else 1.2)
        line_pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
        if isinstance(syll.top_line, QtWidgets.QGraphicsLineItem):
            syll.top_line.setPen(line_pen)
        if isinstance(syll.bottom_line, QtWidgets.QGraphicsLineItem):
            syll.bottom_line.setPen(line_pen)
        if isinstance(syll.label_artist, QtWidgets.QGraphicsSimpleTextItem):
            text_col = _contrasting_text_qcolor(r, g, b)
            syll.label_artist.setBrush(QtGui.QBrush(text_col))

    def get_selected(self) -> List[Syllable]:
        return [s for s in self.annotations if s.selected]

    def clear_selection(self):
        any_changed = False
        for s in self.annotations:
            if s.selected:
                s.selected = False
                self._update_patch_style(s)
                any_changed = True

    def apply_name_to_selected(self, new_name: str):
        changed = False
        for s in self.annotations:
            if s.selected:
                s.name = new_name
                self._update_patch_style(s)
                if isinstance(s.label_artist, QtWidgets.QGraphicsSimpleTextItem):
                    s.label_artist.setText(new_name)
                changed = True

    def refresh_colors(self):
        for s in self.annotations:
            self._update_patch_style(s)

    def set_edit_mode(self, enabled: bool):
        self.edit_mode = enabled
        # Update cursor to indicate editing capability
        if self.edit_mode:
            self.viewport().setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        else:
            # Cancel any in-progress adding
            if self._adding:
                self._cancel_new_syllable()
            self.viewport().unsetCursor()

    # Drag selection across time window within this row
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        pos_scene = self.mapToScene(event.position().toPoint())
        t_scene = float(pos_scene.x())
        t_file = t_scene - self.time_offset
        # Right-click: cancel add if in progress, else delete syllable under cursor
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            if self._adding:
                self._cancel_new_syllable()
                if self.edit_mode:
                    self.viewport().setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
                return
            s = self._find_syllable_at_time(t_scene)
            if s is not None:
                # Delete only the syllable under the cursor, regardless of selection
                self._delete_syllable(s)
                self.selection_changed.emit()
                return
        # Edit mode: add / resize / move
        if self.edit_mode and event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self._adding:
                self._finalize_new_syllable(t_file)
                self.viewport().setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
                return
            # Prefer resize when near any edge, even if slightly outside the rect
            vx = float(event.position().x())
            best_s: Optional[Syllable] = None
            best_edge: Optional[str] = None  # 'resize_left' or 'resize_right'
            best_d = 1e9
            tol = 4.0
            for s0 in self.annotations:
                sp = self.mapFromScene(QtCore.QPointF(s0.start + self.time_offset, 0)).x()
                ep = self.mapFromScene(QtCore.QPointF(s0.stop + self.time_offset, 0)).x()
                d_left = abs(vx - sp)
                d_right = abs(vx - ep)
                if d_left <= tol and d_left < best_d:
                    best_d = d_left
                    best_s = s0
                    best_edge = 'resize_left'
                if d_right <= tol and d_right < best_d:
                    best_d = d_right
                    best_s = s0
                    best_edge = 'resize_right'
            if best_s is not None and best_edge is not None:
                self._drag_syll = best_s
                self._drag_mode = best_edge
                self._drag_active = True
                self.viewport().setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
                return
            # Otherwise, move if clicking inside a syllable
            s = self._find_syllable_at_time(t_scene)
            if s is not None:
                self._drag_mode = 'move'
                self._drag_move_offset = t_file - s.start
                self._drag_active = True
                self._drag_syll = s
                self.viewport().setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                return
            # Empty background: start adding a syllable (first boundary)
            self._start_new_syllable(t_file)
            self.viewport().setCursor(QtCore.Qt.CursorShape.CrossCursor)
            return
        # Selection / lasso (non edit)
        if not self.edit_mode and event.button() == QtCore.Qt.MouseButton.LeftButton:
            mods = QtWidgets.QApplication.keyboardModifiers()
            add_mode = bool(mods & (QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.MetaModifier | QtCore.Qt.KeyboardModifier.ControlModifier))
            s = self._find_syllable_at_time(t_scene)
            if s is not None:
                if add_mode:
                    s.selected = not s.selected
                    self._update_patch_style(s)
                    self.selection_changed.emit()
                else:
                    win = self.window()
                    if hasattr(win, '_clear_selection'):
                        win._clear_selection()
                    s.selected = True
                    self._update_patch_style(s)
                    self.selection_changed.emit()
                return
            win = self.window()
            if not add_mode and hasattr(win, '_clear_selection'):
                win._clear_selection()
            self._drag_active = True
            self._drag_x0 = t_scene
            self._lasso_add = add_mode
            y0, y1 = self._y_limits_khz()
            if self._drag_rect_item is None:
                r = QtCore.QRectF(self._drag_x0, y0, 0.0, y1 - y0)
                self._drag_rect_item = self.scene.addRect(r, pen=QtGui.QPen(QtGui.QColor(255, 255, 0, 200), 0.0), brush=QtGui.QBrush(QtGui.QColor(255, 255, 0, 40)))
            else:
                self._drag_rect_item.setRect(QtCore.QRectF(self._drag_x0, y0, 0.0, y1 - y0))
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        pos_scene = self.mapToScene(event.position().toPoint())
        t_scene = float(pos_scene.x())
        t_file = t_scene - self.time_offset
        # Hover cursor in edit mode
        if self.edit_mode and not self._drag_active:
            if self._adding:
                self.viewport().setCursor(QtCore.Qt.CursorShape.CrossCursor)
            else:
                # Check proximity to any edge using pixel tolerance, independent of being inside the rect
                vx = float(event.position().x())
                tol = 4.0
                edge_near = False
                best_d = 1e9
                for s0 in self.annotations:
                    sp = self.mapFromScene(QtCore.QPointF(s0.start + self.time_offset, 0)).x()
                    ep = self.mapFromScene(QtCore.QPointF(s0.stop + self.time_offset, 0)).x()
                    d_left = abs(vx - sp)
                    d_right = abs(vx - ep)
                    d = min(d_left, d_right)
                    if d <= tol and d < best_d:
                        best_d = d
                        edge_near = True
                if edge_near:
                    self.viewport().setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
                else:
                    # Open hand when hovering inside a syllable (move), otherwise pointing hand
                    s_inside = self._find_syllable_at_time(t_scene)
                    if s_inside is not None:
                        self.viewport().setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
                    else:
                        self.viewport().setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        # Live update for in-progress add
        if self.edit_mode and self._adding and self._new_syll is not None:
            self._update_new_syllable(t_file)
            return
        # Dragging in edit mode
        if self.edit_mode and self._drag_active and self._drag_syll is not None:
            s = self._drag_syll
            tmin, tmax = self._x_limits_seconds()
            min_w = 1e-3
            x = t_file
            if self._drag_mode == 'resize_left':
                new_start = max(tmin, min(x, s.stop - min_w))
                s.start = float(new_start)
            elif self._drag_mode == 'resize_right':
                new_stop = min(tmax, max(x, s.start + min_w))
                s.stop = float(new_stop)
            elif self._drag_mode == 'move':
                width = s.stop - s.start
                new_start = x - self._drag_move_offset
                new_start = max(tmin, min(new_start, tmax - width))
                s.start = float(new_start)
                s.stop = float(new_start + width)
            self._update_syllable_artists(s)
            return
        # Lasso drag update
        if not self.edit_mode and self._drag_active and self._drag_rect_item is not None:
            x1 = t_scene
            x0 = self._drag_x0 if self._drag_x0 is not None else x1
            left = min(x0, x1)
            width = abs(x1 - x0)
            y0, y1 = self._y_limits_khz()
            self._drag_rect_item.setRect(QtCore.QRectF(left, y0, width, y1 - y0))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        # Finalize new syllable on release (click-and-drag add)
        if self.edit_mode and self._adding:
            pos_scene = self.mapToScene(event.position().toPoint())
            self._finalize_new_syllable(float(pos_scene.x()))
            # restore cursor to default edit cursor
            self.viewport().setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            return
        if not self._drag_active:
            super().mouseReleaseEvent(event)
            return
        # Finish edit drag
        if self.edit_mode:
            self._drag_active = False
            self._drag_mode = None
            self._drag_syll = None
            self.viewport().setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            return
        # Finish lasso drag
        self._drag_active = False
        if self._drag_rect_item is None or self._drag_x0 is None:
            return
        pos_scene = self.mapToScene(event.position().toPoint())
        x1 = float(pos_scene.x())
        x0 = float(self._drag_x0)
        t0, t1 = (min(x0, x1), max(x0, x1))
        # convert to file time for overlap tests
        t0f, t1f = t0 - self.time_offset, t1 - self.time_offset
        any_changed = False
        for s in self.annotations:
            prev = s.selected
            overl = (s.start < t1f and s.stop > t0f)
            if self._lasso_add:
                # Union-add overlapped
                s.selected = s.selected or overl
            else:
                # Replace selection with overlapped
                s.selected = overl
            if s.selected != prev:
                self._update_patch_style(s)
                any_changed = True
        # Hide selection rectangle
        self._drag_rect_item.setRect(QtCore.QRectF(0, 0, 0, 0))
        self._drag_x0 = None
        self._lasso_add = False
        if any_changed:
            self.selection_changed.emit()

    def _find_syllable_at_time(self, t: float) -> Optional[Syllable]:
        t_file = t - self.time_offset
        for s in self.annotations:
            if s.start <= t_file <= s.stop:
                return s
        return None

    def _delete_syllable(self, s: Syllable):
        # Remove visual artists
        if isinstance(s.patch, QtWidgets.QGraphicsItem):
            self.scene.removeItem(s.patch)
        if isinstance(s.label_artist, QtWidgets.QGraphicsItem):
            self.scene.removeItem(s.label_artist)
        if isinstance(s.top_line, QtWidgets.QGraphicsItem):
            self.scene.removeItem(s.top_line)
        if isinstance(s.bottom_line, QtWidgets.QGraphicsItem):
            self.scene.removeItem(s.bottom_line)
        # Remove from list
        try:
            self.annotations.remove(s)
        except ValueError:
            pass

    def _update_syllable_artists(self, s: Syllable):
        # Update positions in scene units
        if isinstance(s.patch, QtWidgets.QGraphicsRectItem):
            y0, y1 = self._y_limits_khz()
            s.patch.setRect(QtCore.QRectF(s.start + self.time_offset, y0, max(0.0, s.stop - s.start), y1 - y0))
        if isinstance(s.label_artist, QtWidgets.QGraphicsSimpleTextItem) or isinstance(s.top_line, QtWidgets.QGraphicsLineItem) or isinstance(s.bottom_line, QtWidgets.QGraphicsLineItem):
            # Derive positions from rect to ensure perfect alignment
            if isinstance(s.patch, QtWidgets.QGraphicsRectItem):
                r = s.patch.rect()
                x_left, x_right = r.left(), r.right()
                width = max(0.0, x_right - x_left)
                y0, y1 = self._y_limits_khz()
                if isinstance(s.label_artist, QtWidgets.QGraphicsSimpleTextItem):
                    s.label_artist.setPos(x_left + 0.02 * width, y0 - 0.02 * (y1 - y0))
                if isinstance(s.top_line, QtWidgets.QGraphicsLineItem):
                    s.top_line.setLine(x_left, y0, x_right, y0)
                if isinstance(s.bottom_line, QtWidgets.QGraphicsLineItem):
                    s.bottom_line.setLine(x_left, y1, x_right, y1)

    def _y_limits_khz(self) -> Tuple[float, float]:
        if self.y_limits is not None:
            return (self.y_limits[0] / 1000.0, self.y_limits[1] / 1000.0)
        return (0.0, self._ymax_khz)

    def _x_limits_seconds(self) -> Tuple[float, float]:
        return self.x_limits if self.x_limits is not None else (0.0, self._duration)

    # Update the horizontal offset of this row and reposition all artists
    def set_time_offset(self, new_offset: float):
        try:
            self.time_offset = float(new_offset)
            if isinstance(self._pixmap_item, QtWidgets.QGraphicsPixmapItem):
                self._pixmap_item.setPos(self.time_offset, self._ymax_khz)
            for s in self.annotations:
                self._update_syllable_artists(s)
        except Exception:
            pass

    # Adding syllable helpers
    def _start_new_syllable(self, t0: float):
        self._adding = True
        y0, y1 = self._y_limits_khz()
        y_min, y_max = y0, y1
        default_name = self.default_name_fn() if self.default_name_fn is not None else ""
        # t0 is file time; place at scene time t0 + offset
        rect_item = self.scene.addRect(QtCore.QRectF(t0 + self.time_offset, y_min, 1e-6, y_max - y_min),
                                       pen=QtGui.QPen(QtGui.QColor('white'), 0.0),
                                       brush=QtGui.QBrush(QtGui.QColor(255, 255, 255, 60)))
        text_item = self.scene.addSimpleText(default_name)
        f = text_item.font()
        f.setPointSizeF(8.0)
        text_item.setFont(f)
        text_item.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        text_item.setZValue(3.0)
        text_item.setPos(t0 + self.time_offset, y_min - 0.02 * (y_max - y_min))
        # boundary lines (top at y_min, bottom at y_max); initial length 0, will update on motion
        r0 = rect_item.rect()
        left = r0.left()
        top_line = self.scene.addLine(left, y_min, left, y_min, QtGui.QPen(QtGui.QColor('white'), 1.0))
        bottom_line = self.scene.addLine(left, y_max, left, y_max, QtGui.QPen(QtGui.QColor('white'), 1.0))
        top_line.setZValue(2.0)
        bottom_line.setZValue(2.0)
        syll = Syllable(start=float(t0), stop=float(t0) + 1e-6, name=default_name, patch=rect_item, label_artist=text_item, selected=True, top_line=top_line, bottom_line=bottom_line)
        self.annotations.append(syll)
        self._update_patch_style(syll)
        self._new_syll = syll
        self.selection_changed.emit()

    def _update_new_syllable(self, t1: float):
        if self._new_syll is None:
            return
        tmin, tmax = self._x_limits_seconds()
        t1 = max(tmin, min(t1, tmax))
        self._new_syll.stop = float(t1)
        # Ensure start <= stop for drawing; width can be negative in data terms but we represent width positive with x=min
        s = self._new_syll
        left = min(s.start, s.stop)
        right = max(s.start, s.stop)
        s.start, s.stop = left, right
        self._update_syllable_artists(s)

    def _finalize_new_syllable(self, t1: float):
        if self._new_syll is None:
            return
        self._update_new_syllable(t1)
        # Enforce minimum width
        min_w = 1e-3
        if (self._new_syll.stop - self._new_syll.start) < min_w:
            # remove it
            self._delete_syllable(self._new_syll)
        # finish
        self._adding = False
        self._new_syll = None

    def _cancel_new_syllable(self):
        # Remove any in-progress temporary syllable and reset state
        if self._new_syll is not None:
            self._delete_syllable(self._new_syll)
        self._adding = False
        self._new_syll = None


class LegendWidget(QtWidgets.QWidget):
    label_clicked = QtCore.Signal(str)

    def __init__(self, color_map: Dict[str, Tuple[float, float, float]], parent=None):
        super().__init__(parent)
        self.color_map = color_map
        self.setMinimumWidth(150)
        self._sorted_names: List[str] = []
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def sizeHint(self):
        return QtCore.QSize(150, 220)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())

        y = 10
        line_h = 20
        margin = 10
        self._sorted_names = sorted(self.color_map.keys(), key=natural_name_key)
        for name in self._sorted_names:
            rgb = self.color_map[name]
            color = QtGui.QColor.fromRgbF(*rgb)
            rect = QtCore.QRect(margin, y, 18, 18)
            painter.fillRect(rect, color)
            painter.setPen(self.palette().text().color())
            painter.drawRect(rect)
            painter.drawText(margin + 26, y + 14, name)
            y += line_h

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        # hit-test by row index
        margin = 10
        line_h = 20
        y = int(event.position().y())
        idx = (y - margin) // line_h
        if 0 <= idx < len(self._sorted_names):
            name = self._sorted_names[idx]
            if name:
                self.label_clicked.emit(name)
                return
        return super().mousePressEvent(event)


class BirdIconWidget(QtWidgets.QWidget):
    pass


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, songs_dir: str, global_annotations_csv: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Song Syllable Labeler")
        self.resize(1100, 800)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self.songs_dir = songs_dir
        self.global_annotations_csv_path: Optional[str] = global_annotations_csv
        pairs = list_wavs_and_annotations(self.songs_dir)
        if not pairs:
            QtWidgets.QMessageBox.warning(self, "No audio", f"No WAV files found in {songs_dir}")
        # Load annotations to infer names
        all_names: List[str] = []
        rows_data: List[Tuple[str, List[Syllable]]] = []
        # Determine global y-limits based on maximum sampling rate (Nyquist)
        max_sr = 0
        # Load annotations either from global CSV or per-file CSVs
        ann_map: Dict[str, List[Syllable]] = {}
        if global_annotations_csv:
            ann_map = self._load_global_annotations(global_annotations_csv)
        for wav, ann in pairs:
            base = os.path.basename(wav)
            if global_annotations_csv and base in ann_map:
                sylls = ann_map[base]
            else:
                sylls = load_annotations(ann)
            rows_data.append((wav, sylls))
            all_names.extend([s.name for s in sylls if s.name])
            try:
                sr = librosa.get_samplerate(wav)
                if sr and sr > max_sr:
                    max_sr = sr
            except Exception:
                pass
        self.global_y_limits: Optional[Tuple[float, float]] = None
        if max_sr > 0:
            self.global_y_limits = (0.0, float(max_sr) / 2.0)

        unique_names = sorted({n for n in all_names if n}, key=natural_name_key)
        # Initialize color map for existing names
        base_colors = distinct_colors(max(1, len(unique_names)))
        self.name_colors: Dict[str, Tuple[float, float, float]] = {}
        for i, name in enumerate(unique_names):
            self.name_colors[name] = base_colors[i % len(base_colors)]

        # Autoname counter
        self.autoname_counter = 1

        # Central layout: left scroll with spectrogram rows; right control panel
        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(6)
        self.setCentralWidget(central)

        # Left: scroll area with rows
        self.scroll_area = QtWidgets.QScrollArea()
        # Allow vertical scrolling when rows exceed viewport height
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setWidgetResizable(True)
        self.rows_container = QtWidgets.QWidget()
        self.rows_layout = QtWidgets.QVBoxLayout(self.rows_container)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(2)

        self.rows: List[SpectrogramRow] = []
        # Determine global x-limits based on longest duration
        max_dur = 0.0
        for wav, _ in rows_data:
            try:
                dur = float(librosa.get_duration(path=wav))
            except TypeError:
                dur = float(librosa.get_duration(filename=wav))
            except Exception:
                dur = 0.0
            max_dur = max(max_dur, dur)
        self.global_x_limits: Optional[Tuple[float, float]] = (0.0, max_dur if max_dur > 0 else 0.0)
        self.current_x_limits: Tuple[float, float] = self.global_x_limits
        self.global_duration: float = max_dur

        self.rows: List[SpectrogramRow] = []
        for idx, (wav, sylls) in enumerate(rows_data):
            show_xticks = (idx == len(rows_data) - 1)
            row = SpectrogramRow(
                wav,
                sylls,
                self.name_colors,
                y_limits=self.global_y_limits,
                x_limits=self.current_x_limits,
                show_xticks=show_xticks,
                default_name_fn=self._next_autoname,
                time_offset=0.0,
            )
            row.selection_changed.connect(self._update_selection_count)
            self.rows_layout.addWidget(row)
            self.rows.append(row)
        # Do not add stretch here; it can suppress scrollbars in QScrollArea
        self.scroll_area.setWidget(self.rows_container)
        self._update_rows_container_min_height()

        # Right: controls
        controls = self._build_controls()

        root_layout.addWidget(self.scroll_area, 1)
        root_layout.addWidget(controls)

        self._update_selection_count()

        # Set up global shortcuts and key handling so arrow keys work without clicking
        self._setup_shortcuts()
        QtWidgets.QApplication.instance().installEventFilter(self)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setFocus()

    def _update_rows_container_min_height(self):
        try:
            count = len(self.rows)
        except Exception:
            count = 0
        spacing = self.rows_layout.spacing() if hasattr(self, 'rows_layout') else 0
        total = 0
        for r in getattr(self, 'rows', []):
            h = r.minimumHeight() or r.sizeHint().height()
            total += max(0, h)
        if count > 1:
            total += (count - 1) * spacing
        # small padding
        total += 4
        self.rows_container.setMinimumHeight(max(total, 150))

    def _build_controls(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.selection_label = QtWidgets.QLabel("Selected: 0")
        layout.addWidget(self.selection_label)

        btn_assign_auto = QtWidgets.QPushButton("Assign Auto-Name (N)")
        btn_assign_auto.clicked.connect(self._assign_autoname)
        layout.addWidget(btn_assign_auto)

        btn_assign_custom = QtWidgets.QPushButton("Assign Custom Name…")
        btn_assign_custom.clicked.connect(self._assign_custom_name)
        layout.addWidget(btn_assign_custom)

        # Edit mode toggle
        layout.addSpacing(6)
        self.btn_edit_mode = QtWidgets.QPushButton("Edit Mode [E]")
        self.btn_edit_mode.setCheckable(True)
        self.btn_edit_mode.toggled.connect(self._toggle_edit_mode)
        layout.addWidget(self.btn_edit_mode)

        layout.addSpacing(8)
        btn_propose = QtWidgets.QPushButton("Propose Labels")
        btn_propose.clicked.connect(self._propose_labels)
        layout.addWidget(btn_propose)

        # Align rows by cross-correlation of spectrogram envelopes
        layout.addSpacing(6)
        btn_align = QtWidgets.QPushButton("Align Songs (XCorr)")
        btn_align.setToolTip("Align all rows horizontally by maximizing cross-correlation of spectrogram energy envelopes.")
        btn_align.clicked.connect(self._align_by_xcorr)
        layout.addWidget(btn_align)

        layout.addSpacing(8)
        btn_export = QtWidgets.QPushButton("Export CSV…")
        btn_export.clicked.connect(self._export_csv)
        layout.addWidget(btn_export)

        layout.addSpacing(12)
        layout.addWidget(QtWidgets.QLabel("Name Colors"))
        self.legend = LegendWidget(self.name_colors)
        legend_frame = QtWidgets.QFrame()
        legend_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        legend_layout = QtWidgets.QVBoxLayout(legend_frame)
        legend_layout.setContentsMargins(4, 4, 4, 4)
        legend_layout.addWidget(self.legend)
        layout.addWidget(legend_frame, 1)

        return panel

    def _selected_syllables(self) -> List[Syllable]:
        res: List[Syllable] = []
        for r in self.rows:
            res.extend(r.get_selected())
        return res

    def _update_selection_count(self):
        n = len(self._selected_syllables())
        self.selection_label.setText(f"Selected: {n}")

    def _clear_selection(self):
        for r in self.rows:
            r.clear_selection()
        self._update_selection_count()
        self.legend.label_clicked.connect(self._on_legend_label_clicked)

    def _setup_shortcuts(self):
        self._shortcuts: List[QtGui.QShortcut] = []
        sc_e = QtGui.QShortcut(QtGui.QKeySequence("E"), self, activated=self.btn_edit_mode.toggle)
        sc_e.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._shortcuts.append(sc_e)
        sc_auto = QtGui.QShortcut(QtGui.QKeySequence("N"), self, activated=self._assign_autoname)
        sc_auto.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._shortcuts.append(sc_auto)
        sc_up = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Up), self, activated=self._zoom_in)
        sc_up.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._shortcuts.append(sc_up)
        sc_dn = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Down), self, activated=self._zoom_out)
        sc_dn.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._shortcuts.append(sc_dn)
        sc_lt = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self, activated=self._pan_left)
        sc_lt.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._shortcuts.append(sc_lt)
        sc_rt = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self, activated=self._pan_right)
        sc_rt.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._shortcuts.append(sc_rt)

    def _assign_autoname(self):
        new_name = self._next_available_autoname()
        self._apply_new_name(new_name)

    def _assign_custom_name(self):
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Assign Name", "New syllable name:")
        if ok and new_name.strip():
            self._apply_new_name(new_name.strip())

    def _apply_new_name(self, new_name: str):
        if new_name not in self.name_colors:
            next_color = distinct_colors(len(self.name_colors) + 1)[-1]
            self.name_colors[new_name] = next_color
            self.legend.update()
        for r in self.rows:
            r.apply_name_to_selected(new_name)
            r.refresh_colors()
        self._update_selection_count()

    def _on_legend_label_clicked(self, name: str):
        if not name:
            return
        self._apply_new_name(name)

    def _propose_labels(self):
        if make_proposals is None:
            QtWidgets.QMessageBox.warning(self, "Unavailable", "Propose Labels requires syllabel.propose_labels.make_proposals and its dependencies installed.")
            return
        wav_folder = self.songs_dir
        import tempfile
        # Determine annotations CSV to pass in
        ann_csv_path = getattr(self, 'global_annotations_csv_path', None)
        temp_file: Optional[str] = None
        if not ann_csv_path:
            # Create a temporary combined CSV from current annotations
            records = []
            for row in self.rows:
                base = os.path.basename(row.wav_path)
                for s in row.annotations:
                    records.append({
                        "filename": base,
                        "start_seconds": float(s.start),
                        "stop_seconds": float(s.stop),
                        "name": s.name,
                    })
            df = pd.DataFrame.from_records(records, columns=["filename", "start_seconds", "stop_seconds", "name"])
            # Write to a temporary file
            ntf = tempfile.NamedTemporaryFile(delete=False, suffix="_annotations.csv")
            ann_csv_path = ntf.name
            temp_file = ntf.name
            ntf.close()
            try:
                df.to_csv(ann_csv_path, index=False, float_format="%.9f")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not write temporary annotations CSV:\n{e}")
                return

        # Show blocking modal progress dialog and run synchronously on main thread
        dlg = QtWidgets.QProgressDialog("Generating label proposals…", None, 0, 0, self)
        dlg.setWindowTitle("Propose Labels")
        dlg.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.show()
        QtWidgets.QApplication.processEvents()

        try:
            result_df = make_proposals(wav_folder, ann_csv_path)
        except Exception as e:
            dlg.cancel()
            QtWidgets.QMessageBox.critical(self, "Propose Labels failed", str(e))
            if temp_file:
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
            return
        finally:
            dlg.cancel()

        self._apply_proposals_df(result_df)
        if temp_file:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    def _apply_proposals_df(self, df):
        try:
            if not isinstance(df, pd.DataFrame):
                return
            cols = {c.lower(): c for c in df.columns}
            file_col = cols.get('filename') or cols.get('file')
            start_col = cols.get('start_seconds') or cols.get('start')
            stop_col = cols.get('stop_seconds') or cols.get('stop')
            name_col = cols.get('name')
            if not file_col or not start_col or not stop_col or not name_col:
                return
            rows_by_file = {}
            for row in self.rows:
                base = os.path.basename(row.wav_path)
                rows_by_file.setdefault(base, []).append(row)
            tol = 1e-3
            for _, r in df.iterrows():
                base = r[file_col]
                if not isinstance(base, str):
                    continue
                t0 = float(r[start_col])
                t1 = float(r[stop_col])
                new_name = str(r[name_col]) if pd.notna(r[name_col]) else ''
                for row in rows_by_file.get(base, []):
                    best = None
                    best_err = 1e9
                    for s in row.annotations:
                        err = abs(s.start - t0) + abs(s.stop - t1)
                        if err < best_err:
                            best_err = err
                            best = s
                    if best is not None and best_err <= 2 * tol:
                        if new_name and best.name != new_name:
                            self._ensure_name_color(new_name)
                            best.name = new_name
                            if isinstance(best.label_artist, QtWidgets.QGraphicsSimpleTextItem):
                                best.label_artist.setText(new_name)
                            row._update_patch_style(best)
            for row in self.rows:
                row.refresh_colors()
            self.legend.update()
        except Exception:
            pass

    def _toggle_edit_mode(self, enabled: bool):
        for r in self.rows:
            r.set_edit_mode(enabled)

    # Autoname helper for new syllables
    def _ensure_name_color(self, name: str):
        if name not in self.name_colors:
            next_color = distinct_colors(len(self.name_colors) + 1)[-1]
            self.name_colors[name] = next_color
            if hasattr(self, 'legend'):
                self.legend.update()
            for r in getattr(self, 'rows', []):
                r.refresh_colors()

    def _next_autoname(self) -> str:
        name = self._next_available_autoname()
        self._ensure_name_color(name)
        return name

    def _existing_names(self) -> Set[str]:
        names: Set[str] = set()
        for r in self.rows:
            for s in r.annotations:
                if s.name:
                    names.add(s.name)
        return names

    def _next_available_autoname(self) -> str:
        names = self._existing_names()
        max_num = 0
        for n in names:
            m = re.fullmatch(r"[Ll](\d+)", n.strip())
            if m:
                try:
                    num = int(m.group(1))
                    if num > max_num:
                        max_num = num
                except Exception:
                    continue
        candidate = max_num + 1 if max_num > 0 else 1
        lower_set = {x.lower() for x in names}
        while f"l{candidate}" in lower_set:
            candidate += 1
        return f"L{candidate}"

    # Time window navigation helpers
    def _apply_time_window_to_rows(self):
        for r in self.rows:
            r.x_limits = self.current_x_limits
            r._apply_fit()

    def _set_time_window(self, t0: float, t1: float):
        eps = 1e-6
        t0 = max(0.0, t0)
        t1 = min(self.global_duration, t1)
        if t1 - t0 < 0.01:
            mid = (t0 + t1) / 2.0
            half = 0.005
            t0, t1 = max(0.0, mid - half), min(self.global_duration, mid + half)
        self.current_x_limits = (t0, t1)
        self._apply_time_window_to_rows()

    def _window_size(self) -> float:
        return max(0.0, self.current_x_limits[1] - self.current_x_limits[0])

    def _zoom_in(self):
        w = self._window_size()
        if w <= 0:
            return
        c = sum(self.current_x_limits) / 2.0
        new_w = max(0.01, w / 1.2)
        self._set_time_window(c - new_w / 2.0, c + new_w / 2.0)

    def _zoom_out(self):
        w = self._window_size()
        if w <= 0:
            return
        c = sum(self.current_x_limits) / 2.0
        new_w = min(self.global_duration, w * 1.2)
        self._set_time_window(c - new_w / 2.0, c + new_w / 2.0)

    def _pan_left(self):
        w = self._window_size()
        if w <= 0:
            return
        shift = w / 5.0
        new_t0 = max(0.0, self.current_x_limits[0] - shift)
        new_t1 = new_t0 + w
        if new_t1 > self.global_duration:
            new_t1 = self.global_duration
            new_t0 = max(0.0, new_t1 - w)
        self._set_time_window(new_t0, new_t1)

    def _align_by_xcorr(self):
        # Align by cross-correlation over the full spectrogram (time axis)
        if not self.rows:
            return
        try:
            # Reference is the first row
            ref = self.rows[0]
            if ref._spec_uniform is None or ref._spec_uniform.size == 0:
                QtWidgets.QMessageBox.warning(self, "Align Songs", "Reference row has no spectrogram data; cannot align.")
                return

            def _xcorr_fullspec_lag_seconds(A: np.ndarray, B: np.ndarray, dt: float) -> float:
                # A: (F, Ta), B: (F, Tb). Use sum over frequencies of 1D correlations along time.
                A = np.asarray(A, dtype=float)
                B = np.asarray(B, dtype=float)
                if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[0]:
                    return 0.0
                F, Ta = A.shape
                _, Tb = B.shape
                if Ta < 2 or Tb < 2:
                    return 0.0
                # Standardize each frequency band over time to reduce DC bias
                eps = 1e-12
                A = A - A.mean(axis=1, keepdims=True)
                B = B - B.mean(axis=1, keepdims=True)
                A_std = A.std(axis=1, keepdims=True)
                B_std = B.std(axis=1, keepdims=True)
                A = A / np.maximum(A_std, eps)
                B = B / np.maximum(B_std, eps)

                L = Ta + Tb - 1
                nfft = 1
                while nfft < L:
                    nfft <<= 1
                # Accumulate correlation across frequency
                C = np.zeros(nfft, dtype=np.float64)
                for f in range(F):
                    fa = np.fft.rfft(A[f, :], nfft)
                    fb = np.fft.rfft(B[f, :], nfft)
                    cf = np.fft.irfft(fa * np.conj(fb), nfft)
                    C += cf
                C = C[:L]
                # Normalize by overlap length to reduce edge bias
                lags = np.arange(-(Tb - 1), Ta, dtype=int)
                overlap = np.where(lags >= 0, np.minimum(Ta, Tb - lags), np.minimum(Ta + lags, Tb))
                overlap = np.maximum(1, overlap).astype(np.float64)
                NCC = C / overlap
                k = int(np.argmax(NCC))
                lag_samples = k - (Tb - 1)  # same mapping as numpy.correlate full
                # Shift to apply to B to best match A is -lag
                return float(-lag_samples) * float(dt)

            dt = float(getattr(self.rows[0], '_env_dt', 0.01))
            shifts: List[float] = []
            durations: List[float] = []
            for idx, row in enumerate(self.rows):
                durations.append(float(getattr(row, '_duration', 0.0)))
                if idx == 0:
                    shifts.append(0.0)
                    continue
                if row._spec_uniform is None or row._spec_uniform.size == 0:
                    shifts.append(0.0)
                    continue
                try:
                    lag_sec = _xcorr_fullspec_lag_seconds(ref._spec_uniform, row._spec_uniform, dt)
                except Exception:
                    lag_sec = 0.0
                shifts.append(lag_sec)

            # Normalize shifts so earliest starts at 0
            min_shift = min(shifts) if shifts else 0.0
            norm_shifts = [s - min_shift for s in shifts]

            # Apply offsets
            for row, off in zip(self.rows, norm_shifts):
                row.set_time_offset(float(off))

            # Update time window to include full extents after alignment
            max_end = 0.0
            for off, dur in zip(norm_shifts, durations):
                max_end = max(max_end, float(off) + float(dur))
            self.global_duration = float(max_end)
            self.global_x_limits = (0.0, float(max_end))
            self._set_time_window(0.0, float(max_end))
        except Exception as e:
            try:
                QtWidgets.QMessageBox.critical(self, "Align Songs failed", str(e))
            except Exception:
                pass

    def _pan_right(self):
        w = self._window_size()
        if w <= 0:
            return
        shift = w / 5.0
        new_t1 = min(self.global_duration, self.current_x_limits[1] + shift)
        new_t0 = new_t1 - w
        if new_t0 < 0.0:
            new_t0 = 0.0
            new_t1 = min(self.global_duration, new_t0 + w)
        self._set_time_window(new_t0, new_t1)

    def _load_global_annotations(self, csv_path: str) -> Dict[str, List[Syllable]]:
        result: Dict[str, List[Syllable]] = {}
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return result
        cols = {c.lower(): c for c in df.columns}
        file_col = cols.get('filename') or cols.get('file')
        start_col = cols.get('start_seconds') or cols.get('start')
        stop_col = cols.get('stop_seconds') or cols.get('stop')
        name_col = cols.get('name')
        if not file_col or not start_col or not stop_col:
            return result
        for _, row in df.iterrows():
            fname = row[file_col]
            if not isinstance(fname, str) or not fname:
                continue
            try:
                start = float(row[start_col])
                stop = float(row[stop_col])
            except Exception:
                continue
            name = str(row[name_col]) if name_col and pd.notna(row[name_col]) else ''
            s = Syllable(start=start, stop=stop, name=name)
            result.setdefault(fname, []).append(s)
        return result

    # Global key handling
    def eventFilter(self, obj, event):
        etype = event.type()
        if etype == QtCore.QEvent.Type.ShortcutOverride:
            key = event.key()
            if key in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
                event.accept()
                if key == QtCore.Qt.Key.Key_Up:
                    self._zoom_in()
                elif key == QtCore.Qt.Key.Key_Down:
                    self._zoom_out()
                elif key == QtCore.Qt.Key.Key_Left:
                    self._pan_left()
                elif key == QtCore.Qt.Key.Key_Right:
                    self._pan_right()
                return True
        if etype == QtCore.QEvent.Type.KeyPress:
            key = event.key()
            if key == QtCore.Qt.Key.Key_Up:
                self._zoom_in()
                return True
            if key == QtCore.Qt.Key.Key_Down:
                self._zoom_out()
                return True
            if key == QtCore.Qt.Key.Key_Left:
                self._pan_left()
                return True
            if key == QtCore.Qt.Key.Key_Right:
                self._pan_right()
                return True
        return super().eventFilter(obj, event)

    def _export_csv(self):
        # Default export path: <songs>/<basename(songs)>.csv
        base = os.path.basename(os.path.abspath(self.songs_dir.rstrip(os.sep)))
        default_path = os.path.join(self.songs_dir, f"{base}.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", default_path, "CSV Files (*.csv)")
        if not path:
            return
        try:
            records = []
            for row in self.rows:
                base = os.path.basename(row.wav_path)
                for s in row.annotations:
                    records.append({
                        "filename": base,
                        "start_seconds": float(s.start),
                        "stop_seconds": float(s.stop),
                        "name": s.name,
                    })
            df = pd.DataFrame.from_records(records, columns=["filename", "start_seconds", "stop_seconds", "name"])
            df.to_csv(path, index=False, float_format="%.9f")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"Could not write CSV:\n{e}")
            return


def distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    try:
        import colorcet as cc  # type: ignore
        palette = list(cc.glasbey_dark)
        rgb = [mcolors.to_rgb(c) for c in palette[: max(1, n)]]
        return [tuple(map(float, c)) for c in rgb[:n]]
    except Exception:
        colors = []
        for i in range(n):
            h = (i / max(1, n)) % 1.0
            s = 0.75
            v = 0.95
            colors.append(tuple(QtGui.QColor.fromHsvF(h, s, v).getRgbF()[:3]))
        return colors


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Song Syllable Labeler")
    parser.add_argument("--songs", dest="songs", required=True, help="Path to songs directory containing WAVs (required)")
    parser.add_argument("--annotations", dest="annotations", default=None, help="Optional path to a global CSV (filename,start_seconds,stop_seconds,name). Defaults to <songs>/<basename(songs)>.csv if omitted.")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    # Set application icon from packaged icon
    try:
        pkg_dir = os.path.dirname(__file__)
        icon_path = os.path.join(pkg_dir, "icon.png")
        if os.path.exists(icon_path):
            app.setWindowIcon(QtGui.QIcon(icon_path))
    except Exception:
        pass

    songs_dir = args.songs
    if not os.path.isdir(songs_dir):
        QtWidgets.QMessageBox.critical(None, "Missing folder", f"Songs folder not found: {songs_dir}")
        sys.exit(1)

    # Default annotations CSV path if not provided
    ann_path = args.annotations
    if not ann_path:
        base = os.path.basename(os.path.abspath(songs_dir.rstrip(os.sep)))
        ann_path = os.path.join(songs_dir, f"{base}.csv")

    win = MainWindow(songs_dir, global_annotations_csv=ann_path)
    try:
        if os.path.exists(icon_path):
            win.setWindowIcon(QtGui.QIcon(icon_path))
    except Exception:
        pass
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
