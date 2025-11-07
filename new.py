from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


# -------------------------------------------------------------
# Core simulation data classes
# -------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v)
    return (v / n).astype(np.float32)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (mat / norms).astype(np.float32)


def _reflect(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return v - 2.0 * np.dot(v, normal) * normal


@dataclass
class Camera:
    distance: float = 6.0
    yaw: float = 0.0
    pitch: float = 0.0
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def forward(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        return np.array([cp * cy, sp, cp * sy], dtype=np.float32)

    def right(self) -> np.ndarray:
        fwd = self.forward()
        r = np.cross(fwd, self.WORLD_UP)
        n = float(np.linalg.norm(r)) or 1.0
        return (r / n).astype(np.float32)

    def up(self) -> np.ndarray:
        fwd = self.forward()
        r = self.right()
        u = np.cross(r, fwd)
        n = float(np.linalg.norm(u)) or 1.0
        return (u / n).astype(np.float32)

    def position(self) -> np.ndarray:
        return self.target - self.forward() * self.distance


@dataclass
class OrbitParameters:
    radius: float
    angular_velocity: float


class OrbitSimulator:
    def __init__(self, params: OrbitParameters):
        self.params = params
        self._angle = 0.0

    def update(self, dt: float) -> None:
        self._angle = (self._angle + self.params.angular_velocity * dt) % (2 * math.pi)

    def set_radius(self, radius: float) -> None:
        self.params.radius = max(radius, 1.2)

    def set_speed(self, angular_velocity: float) -> None:
        self.params.angular_velocity = max(0.0, angular_velocity)

    def position(self) -> np.ndarray:
        x = self.params.radius * math.cos(self._angle)
        z = self.params.radius * math.sin(self._angle)
        return np.array([x, 0.0, z], dtype=np.float32)

    def set_angle(self, angle: float) -> None:
        self._angle = angle % (2 * math.pi)


@dataclass
class PlanetConfig:
    base_radius: float = 2.0
    height_scale: float = 0.20
    lat_steps: int = 128
    lon_steps: int = 256


class Planet:
    def __init__(self, cfg: PlanetConfig | None = None):
        self.cfg = cfg or PlanetConfig()
        self.heights = np.zeros((self.cfg.lat_steps, self.cfg.lon_steps), dtype=np.float32)

    def generate(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        lat = np.linspace(-math.pi / 2, math.pi / 2, self.cfg.lat_steps, dtype=np.float32)
        lon = np.linspace(0.0, 2 * math.pi, self.cfg.lon_steps, endpoint=False, dtype=np.float32)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        base = self._fbm(lat_grid, lon_grid, rng)
        craters = self._craters(lat_grid, lon_grid, rng)

        h = (0.7 * base + 0.3 * craters).astype(np.float32)
        self.heights = np.clip(h, -1.0, 1.0)

    def _fbm(self, latg, long, rng):
        total = np.zeros_like(latg, dtype=np.float32)
        freq, amp = 1.0, 1.0
        norm = 0.0
        for _ in range(5):
            p1 = rng.uniform(0, 2 * math.pi)
            p2 = rng.uniform(0, 2 * math.pi)
            s = np.sin(latg * freq + p1) * np.cos(long * freq + p2)
            total += (amp * s).astype(np.float32)
            norm += amp
            freq *= 2.0
            amp *= 0.5
        total /= max(norm, 1e-6)
        return total

    def _craters(self, latg, long, rng):
        out = np.zeros_like(latg, dtype=np.float32)
        count = int(rng.integers(12, 24))
        for _ in range(count):
            cl = float(rng.uniform(-math.pi / 2, math.pi / 2))
            co = float(rng.uniform(0, 2 * math.pi))
            r = float(rng.uniform(0.05, 0.18))
            d = float(rng.uniform(-0.5, -0.1))
            dist = self._ang_dist(latg, long, cl, co)
            out = np.minimum(out, (d * np.exp(-(dist * dist) / (2 * r * r))).astype(np.float32))
        return out

    def _ang_dist(self, latg, long, cl, co):
        dlat = latg - cl
        dlon = long - co
        a = (np.sin(dlat * 0.5) ** 2).astype(np.float32) + np.cos(latg) * math.cos(cl) * (np.sin(dlon * 0.5) ** 2)
        a = np.clip(a.astype(np.float32), 0.0, 1.0)
        return (2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a) + 1e-6)).astype(np.float32)


@dataclass
class Light:
    direction: np.ndarray
    intensity: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.98, 0.92], dtype=np.float32))


@dataclass
class RenderSettings:
    width: int = 256
    height: int = 144
    fov: float = math.radians(60.0)


# -------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------


class PlanetMesh:
    def __init__(self, planet: Planet):
        self.planet = planet
        self.lat_steps = planet.cfg.lat_steps
        self.lon_steps = planet.cfg.lon_steps
        self._latitudes = np.linspace(-math.pi / 2, math.pi / 2, self.lat_steps, dtype=np.float32)
        self._longitudes = np.linspace(0.0, 2 * math.pi, self.lon_steps, endpoint=False, dtype=np.float32)
        self.indices = self._build_indices()
        self.positions = np.zeros((self.lat_steps * self.lon_steps, 3), dtype=np.float32)
        self.normals = np.zeros_like(self.positions)
        self.altitudes = np.zeros(self.lat_steps * self.lon_steps, dtype=np.float32)
        self.max_radius = planet.cfg.base_radius
        self.update()

    def _build_indices(self) -> np.ndarray:
        idx = []
        for i in range(self.lat_steps - 1):
            for j in range(self.lon_steps):
                i0 = i * self.lon_steps + j
                i1 = i * self.lon_steps + ((j + 1) % self.lon_steps)
                i2 = (i + 1) * self.lon_steps + j
                i3 = (i + 1) * self.lon_steps + ((j + 1) % self.lon_steps)
                idx.append((i0, i2, i1))
                idx.append((i2, i3, i1))
        return np.array(idx, dtype=np.int32)

    def update(self) -> None:
        cfg = self.planet.cfg
        heights = self.planet.heights.astype(np.float32)
        radius = cfg.base_radius + cfg.height_scale * heights

        cos_lat = np.cos(self._latitudes).astype(np.float32)[:, None]
        sin_lat = np.sin(self._latitudes).astype(np.float32)[:, None]
        cos_lon = np.cos(self._longitudes).astype(np.float32)[None, :]
        sin_lon = np.sin(self._longitudes).astype(np.float32)[None, :]

        dir_x = cos_lat * cos_lon
        dir_y = sin_lat * np.ones_like(cos_lon)
        dir_z = cos_lat * sin_lon

        x = radius * dir_x
        y = radius * dir_y
        z = radius * dir_z

        positions = np.stack([x, y, z], axis=-1)
        self.positions = positions.reshape(-1, 3).astype(np.float32)
        self.altitudes = (radius - cfg.base_radius).reshape(-1).astype(np.float32)
        self.max_radius = float(np.max(radius))

        normals = np.zeros_like(positions, dtype=np.float32)
        for i in range(self.lat_steps):
            ip = min(i + 1, self.lat_steps - 1)
            im = max(i - 1, 0)
            for j in range(self.lon_steps):
                jp = (j + 1) % self.lon_steps
                jm = (j - 1) % self.lon_steps
                v_lat = positions[ip, j] - positions[im, j]
                v_lon = positions[i, jp] - positions[i, jm]
                n = np.cross(v_lon, v_lat)
                normals[i, j] = _normalize(n)
        self.normals = normals.reshape(-1, 3).astype(np.float32)


class SphereMesh:
    def __init__(self, radius: float, lat_steps: int = 32, lon_steps: int = 64):
        self.radius = radius
        latitudes = np.linspace(-math.pi / 2, math.pi / 2, lat_steps, dtype=np.float32)
        longitudes = np.linspace(0.0, 2 * math.pi, lon_steps, endpoint=False, dtype=np.float32)
        cos_lat = np.cos(latitudes).astype(np.float32)[:, None]
        sin_lat = np.sin(latitudes).astype(np.float32)[:, None]
        cos_lon = np.cos(longitudes).astype(np.float32)[None, :]
        sin_lon = np.sin(longitudes).astype(np.float32)[None, :]

        dir_x = cos_lat * cos_lon
        dir_y = sin_lat * np.ones_like(cos_lon)
        dir_z = cos_lat * sin_lon
        directions = np.stack([dir_x, dir_y, dir_z], axis=-1)
        self.directions = directions.reshape(-1, 3).astype(np.float32)
        self.vertices = (self.directions * radius).astype(np.float32)

        idx = []
        for i in range(lat_steps - 1):
            for j in range(lon_steps):
                i0 = i * lon_steps + j
                i1 = i * lon_steps + ((j + 1) % lon_steps)
                i2 = (i + 1) * lon_steps + j
                i3 = (i + 1) * lon_steps + ((j + 1) % lon_steps)
                idx.append((i0, i2, i1))
                idx.append((i2, i3, i1))
        self.indices = np.array(idx, dtype=np.int32)


# -------------------------------------------------------------
# Rendering helpers
# -------------------------------------------------------------


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = center - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    u = up / (np.linalg.norm(up) + 1e-8)
    s = np.cross(f, u)
    s = s / (np.linalg.norm(s) + 1e-8)
    u2 = np.cross(s, f)

    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u2
    M[2, :3] = -f
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u2, eye)
    M[2, 3] = np.dot(f, eye)
    return M


def perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


def project_points(points: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points, ones], axis=1)
    clip = (matrix @ homo.T).T
    w = clip[:, 3:4]
    mask = w[:, 0] > 1e-4
    ndc = np.zeros_like(clip[:, :3])
    ndc[mask] = clip[mask, :3] / w[mask]
    return ndc, w[:, 0], mask


def _ray_sphere_intersection(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float) -> bool:
    oc = origin - center
    b = np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return False
    s = math.sqrt(max(disc, 0.0))
    t = -b - s
    if t > 1e-4:
        return True
    t = -b + s
    return t > 1e-4


class StarField:
    def __init__(self, count: int = 1400, seed: int = 12345):
        rng = np.random.default_rng(seed)
        phi = rng.uniform(0.0, 2 * math.pi, count)
        cos_theta = rng.uniform(-1.0, 1.0, count)
        sin_theta = np.sqrt(1.0 - np.clip(cos_theta * cos_theta, 0.0, 1.0))
        self.directions = np.stack([
            sin_theta * np.cos(phi),
            cos_theta,
            sin_theta * np.sin(phi),
        ], axis=1).astype(np.float32)
        self.base_intensity = rng.uniform(0.55, 1.0, count).astype(np.float32)
        self.sizes = rng.uniform(0.6, 2.0, count).astype(np.float32)
        self.twinkle_freq = rng.uniform(1.5, 4.5, count).astype(np.float32)
        self.twinkle_phase = rng.uniform(0.0, 2 * math.pi, count).astype(np.float32)

    def iter_visible(self, scene_rot: np.ndarray, camera: Camera, settings: RenderSettings, view: np.ndarray,
                     proj: np.ndarray, time_now: float) -> Iterable[tuple[float, float, float, float]]:
        vp = proj @ view
        far_distance = 60.0
        dirs_world = (scene_rot @ self.directions.T).T
        points = dirs_world * far_distance
        ndc, _, mask = project_points(points, vp)
        width, height = settings.width, settings.height
        sx = (ndc[:, 0] * 0.5 + 0.5) * width
        sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
        visible = mask & (ndc[:, 2] < 1.0)
        brightness = self.base_intensity * (0.6 + 0.4 * np.sin(self.twinkle_phase + time_now * self.twinkle_freq))
        for x, y, br, sz, ok in zip(sx, sy, brightness, self.sizes, visible):
            if ok and 0.0 <= x < width and 0.0 <= y < height:
                yield float(x), float(y), float(max(br, 0.0)), float(sz)


# -------------------------------------------------------------
# Main rendering widget
# -------------------------------------------------------------


class SoftwareRendererWidget(QtWidgets.QWidget):
    frame_ready = QtCore.pyqtSignal(object)

    def __init__(
        self,
        planet: Planet,
        orbit: OrbitSimulator,
        light: Light,
        camera: Camera,
        settings: RenderSettings,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.planet = planet
        self.orbit = orbit
        self.light = light
        self.camera = camera
        self.settings = settings

        self.planet_mesh = PlanetMesh(planet)
        self.satellite_mesh = SphereMesh(radius=0.35, lat_steps=24, lon_steps=48)
        self.star_field = StarField()

        self._scene_rot = np.identity(3, dtype=np.float32)
        self._scene_trans = np.zeros(3, dtype=np.float32)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(16)
        self._last_time = time.time()
        self._start_time = self._last_time
        self._record_callback: Optional[Callable[[np.ndarray], None]] = None

    def rotate_scene_euler_deg(self, dx: float, dy: float, dz: float) -> None:
        def Rx(a: float) -> np.ndarray:
            c, s = math.cos(a), math.sin(a)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

        def Ry(a: float) -> np.ndarray:
            c, s = math.cos(a), math.sin(a)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        def Rz(a: float) -> np.ndarray:
            c, s = math.cos(a), math.sin(a)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        R = Rz(math.radians(dz)) @ Ry(math.radians(dy)) @ Rx(math.radians(dx))
        self._scene_rot = (R @ self._scene_rot).astype(np.float32)
        self.update()

    def translate_scene(self, dx: float, dy: float, dz: float) -> None:
        self._scene_trans += np.array([dx, dy, dz], dtype=np.float32)
        self.update()

    def update_planet_mesh(self) -> None:
        self.planet_mesh.update()
        self.update()

    def set_record_callback(self, fn: Optional[Callable[[np.ndarray], None]]) -> None:
        self._record_callback = fn

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(960, 540)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        now = time.time()
        dt = min(max(now - self._last_time, 0.0), 0.050)
        self._last_time = now
        self.orbit.update(dt)

        image = self._render(now)

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        target_rect = self.rect()
        painter.drawImage(target_rect, image)
        painter.end()

        if self._record_callback is not None:
            img_rgb = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
            ptr = img_rgb.bits()
            ptr.setsize(img_rgb.height() * img_rgb.bytesPerLine())
            arr = np.frombuffer(ptr, np.uint8).reshape(img_rgb.height(), img_rgb.width(), 3).copy()
            self._record_callback(arr)

    # ---------------------------------------------------------
    # Rendering pipeline
    # ---------------------------------------------------------

    def _render(self, now: float) -> QtGui.QImage:
        width, height = self.settings.width, self.settings.height
        image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.QColor(0, 0, 0))
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        camera_pos = self.camera.position().astype(np.float32)
        view = look_at(camera_pos, self.camera.target.astype(np.float32), self.camera.up())
        aspect = float(width) / max(float(height), 1.0)
        proj = perspective(self.settings.fov, aspect, 0.1, 120.0)
        vp = proj @ view

        self._draw_sky(painter, now, view, proj, width, height)
        planet_center_world = self._scene_rot @ np.zeros(3, dtype=np.float32) + self._scene_trans
        sat_center_local = self.orbit.position().astype(np.float32)
        sat_center_world = (self._scene_rot @ sat_center_local) + self._scene_trans

        self._draw_planet(painter, vp, camera_pos, sat_center_world)
        self._draw_satellite(painter, vp, camera_pos, sat_center_world, planet_center_world)
        self._draw_axes(painter, width, height)

        painter.end()
        return image

    def _draw_sky(self, painter: QtGui.QPainter, now: float, view: np.ndarray, proj: np.ndarray, width: int, height: int) -> None:
        gradient = QtGui.QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0.0, QtGui.QColor(0, 3, 8))
        gradient.setColorAt(1.0, QtGui.QColor(0, 0, 2))
        painter.fillRect(0, 0, width, height, gradient)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        for x, y, intensity, size in self.star_field.iter_visible(self._scene_rot, self.camera, self.settings, view, proj, now - self._start_time):
            alpha = int(np.clip(intensity * 255.0, 30, 255))
            radius = size * 0.5
            color = QtGui.QColor(215, 225, 255, alpha)
            painter.setBrush(QtGui.QBrush(color))
            painter.drawEllipse(QtCore.QPointF(x, y), radius, radius)

    def _draw_planet(self, painter: QtGui.QPainter, vp: np.ndarray, camera_pos: np.ndarray, sat_center_world: np.ndarray) -> None:
        positions_model = self.planet_mesh.positions
        normals_model = self.planet_mesh.normals
        altitudes = self.planet_mesh.altitudes
        rot = self._scene_rot
        trans = self._scene_trans

        positions_world = (rot @ positions_model.T).T + trans
        normals_world = _normalize_rows((rot @ normals_model.T).T)

        ndc, _, mask = project_points(positions_world, vp)
        sx = (ndc[:, 0] * 0.5 + 0.5) * self.settings.width
        sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self.settings.height
        depth = ndc[:, 2]

        triangles = []
        cam = camera_pos.astype(np.float32)
        light_dir = _normalize(-self.light.direction.astype(np.float32))
        for tri in self.planet_mesh.indices:
            if not (mask[tri[0]] and mask[tri[1]] and mask[tri[2]]):
                continue
            pts = np.array([
                [sx[tri[0]], sy[tri[0]]],
                [sx[tri[1]], sy[tri[1]]],
                [sx[tri[2]], sy[tri[2]]],
            ], dtype=np.float32)
            if np.any(np.isnan(pts)):
                continue
            world_pts = positions_world[tri]
            normal = _normalize(np.mean(normals_world[tri], axis=0))
            centroid = np.mean(world_pts, axis=0)
            view_dir = _normalize(cam - centroid)
            if np.dot(normal, view_dir) <= 0.0:
                continue
            altitude = float(np.mean(altitudes[tri]))
            color = self._shade_planet(centroid, normal, altitude, cam, light_dir, sat_center_world)
            avg_depth = float(np.mean(depth[tri]))
            triangles.append((avg_depth, pts, color))

        triangles.sort(key=lambda item: item[0], reverse=True)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for _, pts, color in triangles:
            qcolor = QtGui.QColor()
            qcolor.setRgbF(color[0], color[1], color[2])
            painter.setBrush(QtGui.QBrush(qcolor))
            path = QtGui.QPainterPath()
            path.moveTo(float(pts[0, 0]), float(pts[0, 1]))
            path.lineTo(float(pts[1, 0]), float(pts[1, 1]))
            path.lineTo(float(pts[2, 0]), float(pts[2, 1]))
            path.closeSubpath()
            painter.drawPath(path)

    def _draw_satellite(self, painter: QtGui.QPainter, vp: np.ndarray, camera_pos: np.ndarray, sat_center_world: np.ndarray,
                        planet_center_world: np.ndarray) -> None:
        mesh = self.satellite_mesh
        rot = self._scene_rot
        cam = camera_pos.astype(np.float32)
        positions_world = (rot @ mesh.vertices.T).T + sat_center_world
        normals_world = _normalize_rows((rot @ mesh.directions.T).T)

        ndc, _, mask = project_points(positions_world, vp)
        sx = (ndc[:, 0] * 0.5 + 0.5) * self.settings.width
        sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self.settings.height
        depth = ndc[:, 2]

        light_dir = _normalize(-self.light.direction.astype(np.float32))
        triangles = []
        for tri in mesh.indices:
            if not (mask[tri[0]] and mask[tri[1]] and mask[tri[2]]):
                continue
            pts = np.array([
                [sx[tri[0]], sy[tri[0]]],
                [sx[tri[1]], sy[tri[1]]],
                [sx[tri[2]], sy[tri[2]]],
            ], dtype=np.float32)
            if np.any(np.isnan(pts)):
                continue
            world_pts = positions_world[tri]
            normal = _normalize(np.mean(normals_world[tri], axis=0))
            centroid = np.mean(world_pts, axis=0)
            view_dir = _normalize(cam - centroid)
            if np.dot(normal, view_dir) <= 0.0:
                continue
            color = self._shade_satellite(centroid, normal, view_dir, light_dir, planet_center_world)
            avg_depth = float(np.mean(depth[tri]))
            triangles.append((avg_depth, pts, color))

        triangles.sort(key=lambda item: item[0], reverse=True)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for _, pts, color in triangles:
            qcolor = QtGui.QColor()
            qcolor.setRgbF(color[0], color[1], color[2])
            painter.setBrush(QtGui.QBrush(qcolor))
            path = QtGui.QPainterPath()
            path.moveTo(float(pts[0, 0]), float(pts[0, 1]))
            path.lineTo(float(pts[1, 0]), float(pts[1, 1]))
            path.lineTo(float(pts[2, 0]), float(pts[2, 1]))
            path.closeSubpath()
            painter.drawPath(path)

    def _shade_planet(
        self,
        position: np.ndarray,
        normal: np.ndarray,
        altitude: float,
        camera_pos: np.ndarray,
        light_dir: np.ndarray,
        sat_center_world: np.ndarray,
    ) -> np.ndarray:
        ambient = 0.08
        base = np.array([0.22, 0.65, 0.28], dtype=np.float32)
        sky = np.array([0.96, 0.96, 0.98], dtype=np.float32)

        diffuse = 0.0
        specular = 0.0
        n_dot_l = float(np.dot(normal, light_dir))
        shadow = False
        if n_dot_l > 0.0:
            shadow = self._shadowed_by_satellite(position, light_dir, sat_center_world)
        if n_dot_l > 0.0 and not shadow:
            diffuse = max(n_dot_l, 0.0)
            reflect_dir = _reflect(-light_dir, normal)
            specular = pow(max(np.dot(reflect_dir, _normalize(camera_pos - position)), 0.0), 32.0)
        color = base * (ambient + diffuse) + 0.22 * specular
        color *= self.light.intensity.astype(np.float32)
        haze = np.clip(altitude / (self.planet.cfg.base_radius * 0.6), 0.0, 1.0)
        color = color * (1.0 - 0.10 * haze) + sky * (0.06 * haze)
        color = np.clip(color, 0.0, 1.0)
        color = np.power(color, 1.0 / 2.2)
        return color

    def _shade_satellite(
        self,
        position: np.ndarray,
        normal: np.ndarray,
        view_dir: np.ndarray,
        light_dir: np.ndarray,
        planet_center_world: np.ndarray,
    ) -> np.ndarray:
        base = np.array([0.85, 0.85, 0.90], dtype=np.float32)
        diffuse = 0.0
        specular = 0.0
        shadow = self._shadowed_by_planet(position, light_dir, planet_center_world)
        n_dot_l = float(np.dot(normal, light_dir))
        if n_dot_l > 0.0 and not shadow:
            diffuse = max(n_dot_l, 0.0)
            reflect_dir = _reflect(-light_dir, normal)
            specular = pow(max(np.dot(reflect_dir, view_dir), 0.0), 64.0)
        rim = pow(max(1.0 - max(np.dot(normal, view_dir), 0.0), 0.0), 3.0) * 0.12
        color = base * (0.08 + diffuse) + 0.4 * specular + rim
        color *= self.light.intensity.astype(np.float32)
        color = np.clip(color, 0.0, 1.0)
        color = np.power(color, 1.0 / 2.2)
        return color

    def _shadowed_by_satellite(self, point: np.ndarray, light_dir: np.ndarray, sat_center_world: np.ndarray) -> bool:
        to_sat = sat_center_world - point
        proj_len = float(np.dot(to_sat, light_dir))
        if proj_len <= 0.0:
            return False
        closest = to_sat - proj_len * light_dir
        dist_sq = float(np.dot(closest, closest))
        return dist_sq < (self.satellite_mesh.radius ** 2)

    def _shadowed_by_planet(self, point: np.ndarray, light_dir: np.ndarray, planet_center_world: np.ndarray) -> bool:
        offset = point + light_dir * 0.05
        radius = max(self.planet_mesh.max_radius, self.planet.cfg.base_radius)
        return _ray_sphere_intersection(offset, light_dir, planet_center_world, radius)

    def _draw_axes(self, painter: QtGui.QPainter, width: int, height: int) -> None:
        base = np.array([width * 0.10, height * 0.84], dtype=np.float32)
        length = width * 0.12
        axes = [
            (self._scene_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32), QtGui.QColor(255, 60, 60)),
            (self._scene_rot @ np.array([0.0, 1.0, 0.0], dtype=np.float32), QtGui.QColor(60, 255, 60)),
            (self._scene_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32), QtGui.QColor(80, 130, 255)),
        ]
        cam_right = self.camera.right().astype(np.float32)
        cam_up = self.camera.up().astype(np.float32)

        for direction, color in axes:
            dir2d = np.array([
                np.dot(direction, cam_right),
                np.dot(direction, cam_up),
            ], dtype=np.float32)
            n = float(np.linalg.norm(dir2d))
            if n < 1e-4:
                continue
            d = dir2d / n
            tip = base + np.array([d[0], -d[1]]) * length
            pen = QtGui.QPen(color, 2)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(base[0], base[1]), QtCore.QPointF(tip[0], tip[1]))
            arrow = length * 0.18
            perp = np.array([-d[1], d[0]])
            p1 = tip - np.array([d[0], -d[1]]) * arrow + perp * (arrow * 0.6)
            p2 = tip - np.array([d[0], -d[1]]) * arrow - perp * (arrow * 0.6)
            painter.drawLine(QtCore.QPointF(tip[0], tip[1]), QtCore.QPointF(p1[0], p1[1]))
            painter.drawLine(QtCore.QPointF(tip[0], tip[1]), QtCore.QPointF(p2[0], p2[1]))


# -------------------------------------------------------------
# UI components
# -------------------------------------------------------------


@dataclass
class UiState:
    seed: int = 1
    recording: bool = False
    frame_index: int = 0


class RenderWidget(QtWidgets.QWidget):
    def __init__(self, renderer: SoftwareRendererWidget, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(renderer)


class ControlPanel(QtWidgets.QWidget):
    parameters_changed = QtCore.pyqtSignal()
    seed_changed = QtCore.pyqtSignal(int)
    scene_rotated = QtCore.pyqtSignal(float, float, float)
    scene_translated = QtCore.pyqtSignal(float, float, float)
    record_toggled = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._init_palette()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._create_translation_group())
        layout.addWidget(self._create_rotation_group())
        layout.addWidget(self._create_scaling_group())
        layout.addWidget(self._create_orbit_group())
        layout.addStretch(1)

    def _init_palette(self) -> None:
        palette = self.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#f4f5fb"))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#ffffff"))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#e2e6f2"))
        self.setAutoFillBackground(True)
        self.setPalette(palette)
        self.setStyleSheet(
            "QGroupBox {"
            "  background-color: #ffffff;"
            "  border: 1px solid #cfd5e4;"
            "  border-radius: 8px;"
            "  margin-top: 12px;"
            "  padding: 12px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 12px;"
            "  padding: 0 4px;"
            "  color: #3a3f58;"
            "  font-weight: 600;"
            "}"
            "QPushButton {"
            "  background-color: #d8def0;"
            "  border: 1px solid #bcc6dd;"
            "  border-radius: 6px;"
            "  padding: 6px 10px;"
            "  font-weight: 500;"
            "}"
            "QPushButton:hover {"
            "  background-color: #cdd6ec;"
            "}"
        )

    def _spin(self, value=0.0, minimum=-10.0, maximum=10.0, step=0.1) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(2)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def _create_translation_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Перенос сцены")
        form = QtWidgets.QGridLayout(group)
        self.dx, self.dy, self.dz = [self._spin() for _ in range(3)]
        form.addWidget(QtWidgets.QLabel("dX"), 0, 0)
        form.addWidget(self.dx, 0, 1)
        form.addWidget(QtWidgets.QLabel("dY"), 1, 0)
        form.addWidget(self.dy, 1, 1)
        form.addWidget(QtWidgets.QLabel("dZ"), 2, 0)
        form.addWidget(self.dz, 2, 1)
        btn = QtWidgets.QPushButton("Перенести")
        btn.clicked.connect(self._emit_translate)
        form.addWidget(btn, 3, 0, 1, 2)
        return group

    def _emit_translate(self) -> None:
        self.scene_translated.emit(self.dx.value(), self.dy.value(), self.dz.value())
        self.dx.setValue(0.0)
        self.dy.setValue(0.0)
        self.dz.setValue(0.0)

    def _create_rotation_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Поворот сцены (градусы)")
        form = QtWidgets.QGridLayout(group)
        self.angle_x = self._spin(maximum=360)
        self.angle_y = self._spin(maximum=360)
        self.angle_z = self._spin(maximum=360)
        form.addWidget(QtWidgets.QLabel("X"), 0, 0)
        form.addWidget(self.angle_x, 0, 1)
        form.addWidget(QtWidgets.QLabel("Y"), 1, 0)
        form.addWidget(self.angle_y, 1, 1)
        form.addWidget(QtWidgets.QLabel("Z"), 2, 0)
        form.addWidget(self.angle_z, 2, 1)
        btn = QtWidgets.QPushButton("Повернуть сцену")
        btn.clicked.connect(self._emit_rotate)
        form.addWidget(btn, 3, 0, 1, 2)
        return group

    def _emit_rotate(self) -> None:
        dx, dy, dz = self.angle_x.value(), self.angle_y.value(), self.angle_z.value()
        self.scene_rotated.emit(dx, dy, dz)
        self.angle_x.setValue(0.0)
        self.angle_y.setValue(0.0)
        self.angle_z.setValue(0.0)

    def _create_scaling_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Масштабирование (равномерно)")
        form = QtWidgets.QGridLayout(group)
        self.scale_box = self._spin(value=1.0, minimum=0.1, maximum=10.0, step=0.1)
        form.addWidget(QtWidgets.QLabel("scale"), 0, 0)
        form.addWidget(self.scale_box, 0, 1)
        btn = QtWidgets.QPushButton("Масштабировать")
        btn.clicked.connect(self.parameters_changed.emit)
        form.addWidget(btn, 1, 0, 1, 2)
        return group

    def _create_orbit_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Параметры орбиты и Seed")
        form = QtWidgets.QGridLayout(group)
        self.orbit_radius = self._spin(value=4.0, minimum=1.5, maximum=12.0)
        self.orbit_speed = self._spin(value=1.2, minimum=0.0, maximum=6.0)
        self.seed_box = QtWidgets.QSpinBox()
        self.seed_box.setRange(0, 10000)
        self.seed_box.setValue(1)
        form.addWidget(QtWidgets.QLabel("Радиус"), 0, 0)
        form.addWidget(self.orbit_radius, 0, 1)
        form.addWidget(QtWidgets.QLabel("Скорость (рад/с)"), 1, 0)
        form.addWidget(self.orbit_speed, 1, 1)
        form.addWidget(QtWidgets.QLabel("Seed"), 2, 0)
        form.addWidget(self.seed_box, 2, 1)
        btn_apply = QtWidgets.QPushButton("Применить")
        btn_apply.clicked.connect(self._apply_clicked)
        form.addWidget(btn_apply, 3, 0, 1, 2)
        self.record_button = QtWidgets.QPushButton("Начать запись")
        self.record_button.setCheckable(True)
        self.record_button.toggled.connect(self.record_toggled)
        form.addWidget(self.record_button, 4, 0, 1, 2)
        return group

    def _apply_clicked(self) -> None:
        self.parameters_changed.emit()
        self.seed_changed.emit(self.seed_box.value())


class MainWindow(QtWidgets.QMainWindow):
    BASE_WIDTH = 1200
    BASE_HEIGHT = 720

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Visualiser (software renderer)")
        self.setMinimumSize(self.BASE_WIDTH, self.BASE_HEIGHT)
        self.resize(self.BASE_WIDTH, self.BASE_HEIGHT)

        self.ui_state = UiState()

        self.camera = Camera(distance=8.0, yaw=math.radians(90.0), pitch=0.0)
        self.camera.target[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.planet = Planet(PlanetConfig())
        self.planet.generate(self.ui_state.seed)
        self.orbit = OrbitSimulator(OrbitParameters(radius=4.0, angular_velocity=1.2))
        light_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        self.light = Light(light_dir, np.array([1.0, 0.98, 0.92], dtype=np.float32))
        self.settings = RenderSettings(width=256, height=144)

        self.renderer = SoftwareRendererWidget(self.planet, self.orbit, self.light, self.camera, self.settings)

        self.panel = ControlPanel()
        self.render_widget = RenderWidget(self.renderer)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.panel)
        splitter.addWidget(self.render_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.panel.setMinimumWidth(300)
        splitter.setSizes([300, self.BASE_WIDTH - 300])
        self.setCentralWidget(splitter)

        QtGui.QShortcut(QtGui.QKeySequence("0"), self, activated=lambda: self._set_quality("ULTRA"))
        QtGui.QShortcut(QtGui.QKeySequence("1"), self, activated=lambda: self._set_quality("FAST"))
        QtGui.QShortcut(QtGui.QKeySequence("2"), self, activated=lambda: self._set_quality("BALANCED"))
        QtGui.QShortcut(QtGui.QKeySequence("3"), self, activated=lambda: self._set_quality("HIGH"))

        self.panel.parameters_changed.connect(self._apply_controls)
        self.panel.seed_changed.connect(self._regenerate_planet)
        self.panel.scene_rotated.connect(self._rotate_scene)
        self.panel.scene_translated.connect(self._translate_scene)
        self.panel.record_toggled.connect(self._toggle_recording)

        self.statusBar().showMessage(
            "Готово (software). Качество: 0=ULTRA, 1=FAST (по умолчанию), 2=BALANCED, 3=HIGH. Поворот/перенос — у сцены."
        )

        QtCore.QTimer.singleShot(0, self._center_on_screen)

    def _center_on_screen(self) -> None:
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        geo = self.frameGeometry()
        geo.moveCenter(screen.availableGeometry().center())
        self.move(geo.topLeft())

    def _set_quality(self, preset: str) -> None:
        if preset == "ULTRA":
            self.settings.width, self.settings.height = 192, 108
        elif preset == "FAST":
            self.settings.width, self.settings.height = 256, 144
        elif preset == "HIGH":
            self.settings.width, self.settings.height = 800, 450
        else:
            self.settings.width, self.settings.height = 480, 270
        self.statusBar().showMessage(f"Качество: {preset}. Поворот/перенос — у сцены.")

    def _apply_controls(self) -> None:
        scale = max(float(self.panel.scale_box.value()), 0.1)
        self.camera.distance = float(np.clip(self.camera.distance / scale, 1.5, 50.0))
        self.panel.scale_box.setValue(1.0)
        self.orbit.set_radius(self.panel.orbit_radius.value())
        self.orbit.set_speed(self.panel.orbit_speed.value())
        self.renderer.update_planet_mesh()

    def _rotate_scene(self, dx: float, dy: float, dz: float) -> None:
        self.renderer.rotate_scene_euler_deg(dx, dy, dz)

    def _translate_scene(self, dx: float, dy: float, dz: float) -> None:
        self.renderer.translate_scene(dx, dy, dz)

    def _regenerate_planet(self, seed: int) -> None:
        self.planet.generate(seed)
        self.renderer.update_planet_mesh()

    def _toggle_recording(self, active: bool) -> None:
        self.ui_state.recording = active
        self.panel.record_button.setText("Остановить запись" if active else "Начать запись")
        if active:
            self.ui_state.frame_index = 0
            Path("frames").mkdir(exist_ok=True)
            self.renderer.set_record_callback(self._save_frame)
        else:
            self.renderer.set_record_callback(None)

    def _save_frame(self, image: np.ndarray) -> None:
        out = Path("frames")
        out.mkdir(exist_ok=True)
        fname = out / f"frame_{self.ui_state.frame_index:05d}.png"
        qimage = QtGui.QImage(image.data, image.shape[1], image.shape[0], 3 * image.shape[1], QtGui.QImage.Format.Format_RGB888)
        qimage.save(str(fname))
        self.ui_state.frame_index += 1


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------


def run() -> None:
    import sys

    os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
