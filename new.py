# ============================================================
# Satellite Visualiser — FIXED FULL VERSION
# OpenGL 3.3 Core + PyQt6 + PyOpenGL
# ФИКС: QT_OPENGL=desktop (главная проблема)
# ============================================================

from __future__ import annotations
import os
os.environ["QT_OPENGL"] = "desktop"

import math
import time
import ctypes
from dataclasses import dataclass, field
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL import GL


# ============================================================
# CAMERA
# ============================================================

@dataclass
class Camera:
    distance: float = 8.0
    yaw: float = math.radians(90.0)
    pitch: float = 0.0
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def forward(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        return np.array([cp * cy, sp, cp * sy], dtype=np.float32)

    def right(self) -> np.ndarray:
        fwd = self.forward()
        r = np.cross(self.WORLD_UP, fwd)
        n = float(np.linalg.norm(r)) or 1.0
        return (r / n).astype(np.float32)

    def up(self) -> np.ndarray:
        r = self.right()
        fwd = self.forward()
        u = np.cross(fwd, r)
        n = float(np.linalg.norm(u)) or 1.0
        return (u / n).astype(np.float32)

    def position(self) -> np.ndarray:
        return self.target - self.forward() * self.distance


# ============================================================
# ORBIT
# ============================================================

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


# ============================================================
# PLANET
# ============================================================

@dataclass
class PlanetConfig:
    base_radius: float = 2.0
    height_scale: float = 0.20
    lat_steps: int = 256
    lon_steps: int = 512


class Planet:
    def __init__(self, cfg: PlanetConfig | None = None):
        self.cfg = cfg or PlanetConfig()
        self.heights = np.zeros((self.cfg.lat_steps, self.cfg.lon_steps), dtype=np.float32)

    def generate(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        lat = np.linspace(-math.pi/2, math.pi/2, self.cfg.lat_steps, dtype=np.float32)
        lon = np.linspace(0.0, 2*math.pi, self.cfg.lon_steps, endpoint=False, dtype=np.float32)
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
            p1 = rng.uniform(0, 2*math.pi)
            p2 = rng.uniform(0, 2*math.pi)
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
            cl = float(rng.uniform(-math.pi/2, math.pi/2))
            co = float(rng.uniform(0, 2*math.pi))
            r = float(rng.uniform(0.05, 0.18))
            d = float(rng.uniform(-0.5, -0.1))
            dist = self._ang_dist(latg, long, cl, co)
            out = np.minimum(out, (d * np.exp(-(dist*dist)/(2*r*r))).astype(np.float32))

        return out

    def _ang_dist(self, latg, long, cl, co):
        dlat = latg - cl
        dlon = long - co
        a = (np.sin(dlat*0.5)**2).astype(np.float32) + np.cos(latg)*math.cos(cl)*(np.sin(dlon*0.5)**2)
        a = np.clip(a, 0.0, 1.0)
        return (2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a) + 1e-6)).astype(np.float32)


# ============================================================
# MATRIX UTILS
# ============================================================

def mat4_identity():
    return np.eye(4, dtype=np.float32)

def look_at(eye, center, up):
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

    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye

    return M @ T

def perspective(fov, aspect, near, far):
    f = 1.0 / math.tan(fov/2)
    M = np.zeros((4,4), dtype=np.float32)
    M[0,0] = f/aspect
    M[1,1] = f
    M[2,2] = (far+near)/(near-far)
    M[2,3] = (2*far*near)/(near-far)
    M[3,2] = -1.0
    return M

def ortho(l, r, b, t, n, f):
    M = np.eye(4, dtype=np.float32)
    M[0,0] = 2/(r-l)
    M[1,1] = 2/(t-b)
    M[2,2] = -2/(f-n)
    M[0,3] = -(r+l)/(r-l)
    M[1,3] = -(t+b)/(t-b)
    M[2,3] = -(f+n)/(f-n)
    return M

def mul(a, b):
    return (a @ b).astype(np.float32)


# ============================================================
# SPHERE GENERATOR
# ============================================================

def build_uv_sphere(stacks, slices, radius=1.0):
    pos = []
    uv = []

    for i in range(stacks+1):
        v = i / stacks
        phi = v * math.pi
        for j in range(slices+1):
            u = j / slices
            theta = u * 2*math.pi

            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)

            pos += [radius*x, radius*y, radius*z]
            uv += [u, v]

    pos = np.array(pos, dtype=np.float32)
    uv = np.array(uv, dtype=np.float32)

    idx = []
    for i in range(stacks):
        for j in range(slices):
            a = i*(slices+1) + j
            b = a + slices + 1
            idx += [a, b, a+1, b, b+1, a+1]

    idx = np.array(idx, dtype=np.uint32)
    return pos, uv, idx


# ============================================================
# GL WIDGET
# ============================================================

@dataclass
class RenderSettings:
    fov: float = math.radians(60)
    shadow_size: int = 2048
    near: float = 0.05
    far: float = 100.0
    light_dir: np.ndarray = field(default_factory=lambda: np.array([-1.0, 0.3, -0.4], dtype=np.float32))
    light_intensity: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.98, 0.92], dtype=np.float32))


class GLMeshWidget(QOpenGLWidget):

    def __init__(self, planet, orbit, camera, settings, parent=None):
        super().__init__(parent)
        self.setMinimumSize(960, 540)

        self.planet = planet
        self.orbit = orbit
        self.camera = camera
        self.settings = settings

        self._last = time.time()
        self._t0 = time.time()

        t = QtCore.QTimer(self)
        t.timeout.connect(self.update)
        t.start(16)

        self._prog_sky = None
        self._prog_depth = None
        self._prog_lit = None
        self._prog_axes = None

    # ============================================================
    # INITIALIZE GL
    # ============================================================

    def initializeGL(self):
        print("GL_VERSION =", GL.glGetString(GL.GL_VERSION))
        print("GL_VENDOR  =", GL.glGetString(GL.GL_VENDOR))
        print("GL_RENDERER=", GL.glGetString(GL.GL_RENDERER))

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)

        # compile shaders
        self._prog_sky = self._build_program(VS_SCREEN, FS_SKY)
        self._prog_depth = self._build_program(VS_DEPTH, FS_DEPTH)
        self._prog_lit = self._build_program(VS_LIT, FS_LIT)
        self._prog_axes = self._build_program(VS_SCREEN, FS_AXES)

        # fullscreen quad
        self._vao_quad = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao_quad)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)

        quad = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1
        ], dtype=np.float32)

        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0); GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 16, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1); GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, 16, ctypes.c_void_p(8))

        # planet sphere
        p_pos, p_uv, p_idx = build_uv_sphere(160, 320, 1.0)
        self._vao_planet, _, _ = self._build_mesh(p_pos, p_uv, p_idx)
        self._planet_index_count = p_idx.size

        # satellite sphere
        s_pos, s_uv, s_idx = build_uv_sphere(64, 128, 0.35)
        self._vao_sat, _, _ = self._build_mesh(s_pos, s_uv, s_idx)
        self._sat_index_count = s_idx.size

        # height texture
        self._planet_height_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._planet_height_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        self._upload_height()

        # shadow map
        self._shadow_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._shadow_tex)
        size = self.settings.shadow_size
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT24, size, size, 0,
                        GL.GL_DEPTH_COMPONENT, GL.GL_UNSIGNED_INT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        self._shadow_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._shadow_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                  GL.GL_TEXTURE_2D, self._shadow_tex, 0)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # ============================================================
    # HELPER MESH + PROGRAM BUILDERS
    # ============================================================

    def _build_mesh(self, pos, uv, idx):
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)

        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)

        inter = np.concatenate([pos.reshape(-1,3),
                                uv.reshape(-1,2)], axis=1).astype(np.float32).ravel()

        GL.glBufferData(GL.GL_ARRAY_BUFFER, inter.nbytes, inter, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0); GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 20, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1); GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, 20, ctypes.c_void_p(12))

        ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)

        return vao, vbo, ebo

    def _compile(self, kind, src):
        s = GL.glCreateShader(kind)
        GL.glShaderSource(s, src)
        GL.glCompileShader(s)

        if GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
            raise RuntimeError(GL.glGetShaderInfoLog(s).decode())

        return s

    def _build_program(self, vs, fs):
        v = self._compile(GL.GL_VERTEX_SHADER, vs)
        f = self._compile(GL.GL_FRAGMENT_SHADER, fs)
        p = GL.glCreateProgram()
        GL.glAttachShader(p, v)
        GL.glAttachShader(p, f)
        GL.glLinkProgram(p)

        if GL.glGetProgramiv(p, GL.GL_LINK_STATUS) != GL.GL_TRUE:
            raise RuntimeError(GL.glGetProgramInfoLog(p).decode())

        GL.glDeleteShader(v)
        GL.glDeleteShader(f)
        return p

    def _upload_height(self):
        h = self.planet.heights.astype(np.float32)
        h01 = np.clip(h * 0.5 + 0.5, 0.0, 1.0)
        data = (h01 * 255).astype(np.uint8)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._planet_height_tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R8,
                        data.shape[1], data.shape[0], 0,
                        GL.GL_RED, GL.GL_UNSIGNED_BYTE, data)

    # ============================================================
    # RENDER FRAME
    # ============================================================

    def paintGL(self):
        now = time.time()
        dt = min(max(now - self._last, 0.0), 0.05)
        self._last = now
        self.orbit.update(dt)

        pr = max(1.0, float(self.devicePixelRatioF()))
        w = max(1, int(self.width() * pr))
        h = max(1, int(self.height() * pr))

        # compute matrices
        eye = self.camera.position()
        center = self.camera.target
        up = self.camera.up()

        V = look_at(eye, center, up)
        P = perspective(self.settings.fov, w/h, 0.05, 100.0)
        VP = mul(P, V)

        # shadow matrices
        light_dir = self.settings.light_dir / (np.linalg.norm(self.settings.light_dir)+1e-8)

        L_view = look_at(-light_dir*8, np.array([0,0,0], np.float32), np.array([0,1,0], np.float32))
        L_proj = ortho(-10, 10, -10, 10, 0.1, 30)
        LVP = mul(L_proj, L_view)

        sat_center = self.orbit.position()

        # ======================================================
        # 1) SKY
        # ======================================================
        GL.glViewport(0, 0, w, h)
        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glUseProgram(self._prog_sky)
        self._set1f(self._prog_sky, "uTime", now - self._t0)
        self._set2f(self._prog_sky, "uResolution", float(w), float(h))
        self._set3f(self._prog_sky, "uCamFwd", *self.camera.forward())
        self._set3f(self._prog_sky, "uCamRight", *self.camera.right())
        self._set3f(self._prog_sky, "uCamUp", *self.camera.up())

        GL.glBindVertexArray(self._vao_quad)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        # ======================================================
        # 2) SHADOW MAP
        # ======================================================
        GL.glViewport(0, 0, self.settings.shadow_size, self.settings.shadow_size)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._shadow_fbo)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)

        GL.glUseProgram(self._prog_depth)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._planet_height_tex)
        self._seti(self._prog_depth, "uHeightTex", 0)

        # planet
        self._set1f(self._prog_depth, "uBaseRadius", 2.0)
        self._set1f(self._prog_depth, "uHeightScale", 0.20)
        self._setMat4(self._prog_depth, "uM", mat4_identity())
        self._setMat4(self._prog_depth, "uVP", LVP)

        GL.glBindVertexArray(self._vao_planet)
        GL.glDrawElements(GL.GL_TRIANGLES, self._planet_index_count, GL.GL_UNSIGNED_INT, None)

        # satellite
        M_sat = mat4_identity()
        M_sat[:3,3] = sat_center

        self._setMat4(self._prog_depth, "uM", M_sat)
        self._set1f(self._prog_depth, "uBaseRadius", 0.0)
        self._set1f(self._prog_depth, "uHeightScale", 0.0)

        GL.glBindVertexArray(self._vao_sat)
        GL.glDrawElements(GL.GL_TRIANGLES, self._sat_index_count, GL.GL_UNSIGNED_INT, None)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)

        # ======================================================
        # 3) LIT PASS
        # ======================================================
        GL.glViewport(0, 0, w, h)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        GL.glUseProgram(self._prog_lit)

        self._setMat4(self._prog_lit, "uVP", VP)
        self._setMat4(self._prog_lit, "uLVP", LVP)

        self._set3f(self._prog_lit, "uLightDir", *light_dir)
        self._set3f(self._prog_lit, "uLightI", *self.settings.light_intensity)
        self._set3f(self._prog_lit, "uCamPos", *eye)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._shadow_tex)
        self._seti(self._prog_lit, "uShadowMap", 1)

        # planet
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._planet_height_tex)
        self._seti(self._prog_lit, "uHeightTex", 0)
        self._set1f(self._prog_lit, "uBaseRadius", 2.0)
        self._set1f(self._prog_lit, "uHeightScale", 0.20)
        self._set1f(self._prog_lit, "uIsSatellite", 0.0)
        self._set3f(self._prog_lit, "uBaseColor", 0.22,0.65,0.28)
        self._set1f(self._prog_lit, "uShininess", 32.0)
        self._setMat4(self._prog_lit, "uM", mat4_identity())

        GL.glBindVertexArray(self._vao_planet)
        GL.glDrawElements(GL.GL_TRIANGLES, self._planet_index_count, GL.GL_UNSIGNED_INT, None)

        # satellite
        self._set1f(self._prog_lit, "uIsSatellite", 1.0)
        self._set1f(self._prog_lit, "uBaseRadius", 0.0)
        self._set1f(self._prog_lit, "uHeightScale", 0.0)
        self._set3f(self._prog_lit, "uBaseColor", 0.9,0.9,1.0)
        self._set1f(self._prog_lit, "uShininess", 64.0)
        self._setMat4(self._prog_lit, "uM", M_sat)

        GL.glBindVertexArray(self._vao_sat)
        GL.glDrawElements(GL.GL_TRIANGLES, self._sat_index_count, GL.GL_UNSIGNED_INT, None)

        # ======================================================
        # 4) AXES
        # ======================================================
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)

        GL.glUseProgram(self._prog_axes)
        self._set2f(self._prog_axes, "uResolution", float(w), float(h))
        self._set3f(self._prog_axes, "uCamRight", *self.camera.right())
        self._set3f(self._prog_axes, "uCamUp", *self.camera.up())
        self._set3f(self._prog_axes, "uWorldX", 1,0,0)
        self._set3f(self._prog_axes, "uWorldY", 0,1,0)
        self._set3f(self._prog_axes, "uWorldZ", 0,0,1)

        GL.glBindVertexArray(self._vao_quad)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        GL.glDisable(GL.GL_BLEND)

    # ============================================================
    # UNIFORM HELPERS
    # ============================================================

    def _loc(self, prog, name):
        return GL.glGetUniformLocation(prog, name)

    def _seti(self, prog, name, v):
        GL.glUniform1i(self._loc(prog, name), int(v))

    def _set1f(self, prog, name, v):
        GL.glUniform1f(self._loc(prog, name), float(v))

    def _set2f(self, prog, name, x, y):
        GL.glUniform2f(self._loc(prog, name), float(x), float(y))

    def _set3f(self, prog, name, x, y=None, z=None):
        if y is None:
            GL.glUniform3f(self._loc(prog,name), x[0], x[1], x[2])
        else:
            GL.glUniform3f(self._loc(prog,name), float(x), float(y), float(z))

    def _setMat4(self, prog, name, M):
        GL.glUniformMatrix4fv(self._loc(prog, name), 1, GL.GL_FALSE, M.astype(np.float32))


# ============================================================
# CONTROL PANEL UI
# ============================================================

@dataclass
class UiState:
    seed: int = 1


class ControlPanel(QtWidgets.QWidget):
    parameters_changed = QtCore.pyqtSignal()
    seed_changed = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)

        v.addWidget(self._orbit_group())
        v.addWidget(self._camera_group())
        v.addWidget(self._seed_group())
        v.addStretch(1)

    def _spin(self, value=0.0, minimum=-10.0, maximum=10.0, step=0.1):
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(3)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def _orbit_group(self):
        g = QtWidgets.QGroupBox("Орбита")
        grid = QtWidgets.QGridLayout(g)

        self.orbit_radius = self._spin(4.0, 1.2, 16.0, 0.1)
        self.orbit_speed  = self._spin(1.2, 0.0, 10.0, 0.1)

        apply = QtWidgets.QPushButton("Применить")
        apply.clicked.connect(self.parameters_changed.emit)

        grid.addWidget(QtWidgets.QLabel("Радиус"), 0, 0)
        grid.addWidget(self.orbit_radius, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Скорость (рад/с)"), 1, 0)
        grid.addWidget(self.orbit_speed, 1, 1)
        grid.addWidget(apply, 2, 0, 1, 2)

        return g

    def _camera_group(self):
        g = QtWidgets.QGroupBox("Камера")
        grid = QtWidgets.QGridLayout(g)

        self.cam_yaw = self._spin(90.0, -360, 360, 1.0)
        self.cam_pitch = self._spin(0.0, -89, 89, 1.0)
        self.cam_zoom = self._spin(8.0, 1.5, 50.0, 0.1)

        apply = QtWidgets.QPushButton("Применить")
        apply.clicked.connect(self.parameters_changed.emit)

        grid.addWidget(QtWidgets.QLabel("Yaw (°)"), 0, 0)
        grid.addWidget(self.cam_yaw, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Pitch (°)"), 1, 0)
        grid.addWidget(self.cam_pitch, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Distance"), 2, 0)
        grid.addWidget(self.cam_zoom, 2, 1)

        grid.addWidget(apply, 3, 0, 1, 2)

        return g

    def _seed_group(self):
        g = QtWidgets.QGroupBox("Генерация планеты (seed)")
        grid = QtWidgets.QGridLayout(g)

        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 100000)
        self.seed.setValue(1)

        apply = QtWidgets.QPushButton("Перегенерировать")
        apply.clicked.connect(self._emit_seed)

        grid.addWidget(QtWidgets.QLabel("Seed"), 0, 0)
        grid.addWidget(self.seed, 0, 1)
        grid.addWidget(apply, 1, 0, 1, 2)
        return g

    def _emit_seed(self):
        self.seed_changed.emit(self.seed.value())


# ============================================================
# MAIN WINDOW
# ============================================================

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Visualiser — FIXED")
        self.resize(1200, 750)

        self.ui_state = UiState()
        self.planet = Planet(PlanetConfig())
        self.planet.generate(self.ui_state.seed)

        self.camera = Camera()
        self.camera.target[:] = np.array([0,0,0], dtype=np.float32)

        self.orbit = OrbitSimulator(OrbitParameters(4.0, 1.2))
        self.settings = RenderSettings()

        self.gl = GLMeshWidget(self.planet, self.orbit, self.camera, self.settings)
        self.panel = ControlPanel()

        self.panel.parameters_changed.connect(self._apply_controls)
        self.panel.seed_changed.connect(self._regenerate_planet)

        sp = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        sp.addWidget(self.panel)
        sp.addWidget(self.gl)
        sp.setStretchFactor(1, 1)
        self.setCentralWidget(sp)

    def _apply_controls(self):
        self.orbit.set_radius(float(self.panel.orbit_radius.value()))
        self.orbit.set_speed(float(self.panel.orbit_speed.value()))
        self.camera.yaw = math.radians(float(self.panel.cam_yaw.value()))
        self.camera.pitch = math.radians(float(self.panel.cam_pitch.value()))
        self.camera.distance = float(self.panel.cam_zoom.value())

    def _regenerate_planet(self, seed: int):
        self.planet.generate(seed)
        self.gl.makeCurrent()
        self.gl._upload_height()
        self.gl.doneCurrent()


# ============================================================
# SHADERS (UNCHANGED)
# ============================================================

VS_SCREEN = """
#version 330 core
layout(location=0) in vec2 in_pos;
layout(location=1) in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FS_SKY = """
#version 330 core
out vec4 fragColor;
in vec2 v_uv;
uniform vec2 uResolution;
uniform float uTime;
uniform vec3 uCamFwd;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
float hash21(vec2 p){ p = fract(p*vec2(123.34, 345.45)); p += dot(p, p+34.345); return fract(p.x*p.y); }
vec2  hash22(vec2 p){ p = fract(p*vec2(443.8975, 441.423)); p += dot(p, p+19.19); return fract(vec2(p.x*p.y, p.x+p.y)); }
float starDot(vec2 f, vec2 c, float r){ float d = length(f-c); return smoothstep(r, 0.0, d); }
vec3 starsLayer(vec2 suv, vec2 scale, float existThresh, vec2 rrange, float t){
    vec2 p = suv*scale; vec2 i = floor(p); vec2 f = fract(p); vec3 col = vec3(0.0);
    for (int oy=-1; oy<=1; ++oy){
        for (int ox=-1; ox<=1; ++ox){
            vec2 cell = i + vec2(ox, oy);
            float rnd = hash21(cell);
            float exist = step(existThresh, rnd);
            vec2 center = hash22(cell);
            float r = mix(rrange.x, rrange.y, hash21(cell+2.7));
            float tw = 0.6 + 0.4 * sin(rnd*123.0 + t*(2.0 + 3.0*hash21(cell+5.31)));
            vec3 sc = vec3(0.85,0.90,1.0);
            col += sc * starDot(f, center, r) * tw * exist;
        }
    }
    return col;
}
void main(){
    vec2 ndc = vec2(v_uv.x*2.0-1.0, 1.0 - v_uv.y*2.0);
    vec3 dir = normalize(uCamFwd + ndc.x*uCamRight + ndc.y*uCamUp);
    float h = clamp(dir.y*0.5 + 0.5, 0.0, 1.0);
    vec3 bottom = vec3(0.000, 0.002, 0.008);
    vec3 top    = vec3(0.003, 0.012, 0.030);
    vec3 base = mix(bottom, top, h);
    float phi = atan(dir.y, dir.x); if (phi < 0.0) phi += 6.28318530718;
    float theta = acos(clamp(dir.z, -1.0, 1.0));
    vec2 suv = vec2(phi / 6.28318530718, theta / 3.1415926535);
    vec3 s1 = starsLayer(suv, vec2(1200.0, 700.0), 0.940, vec2(0.0035, 0.0068), uTime);
    vec3 s2 = starsLayer(suv, vec2(2600.0, 1600.0), 0.965, vec2(0.0024, 0.0048), uTime);
    vec3 s3 = starsLayer(suv, vec2(4200.0, 2500.0), 0.985, vec2(0.0018, 0.0032), uTime);
    vec3 s4 = starsLayer(suv, vec2(7000.0, 4200.0), 0.992, vec2(0.0012, 0.0020), uTime);
    vec3 g1 = starsLayer(suv, vec2(1000.0, 600.0), 0.9970, vec2(0.0100, 0.0220), uTime);
    vec3 g2 = starsLayer(suv, vec2( 800.0, 480.0), 0.9985, vec2(0.0180, 0.0350), uTime);
    vec3 col = clamp(base + s1 + s2 + s3 + s4 + g1 + g2, 0.0, 1.0);
    col = pow(col, vec3(1.0/2.2));
    fragColor = vec4(col, 1.0);
}
"""

VS_DEPTH = """
#version 330 core
layout(location=0) in vec3 in_pos;
layout(location=1) in vec2 in_uv;
uniform mat4 uM;
uniform mat4 uVP;
uniform sampler2D uHeightTex;
uniform float uBaseRadius;
uniform float uHeightScale;
out vec3 vWorldPos;
float heightTex(vec2 uv){ return texture(uHeightTex, uv).r * 2.0 - 1.0; }
void main(){
    float h = heightTex(in_uv);
    float r = uBaseRadius * (1.0 + uHeightScale * h);
    vec3 world = (uM * vec4(in_pos * r, 1.0)).xyz;
    vWorldPos = world;
    gl_Position = uVP * vec4(world, 1.0);
}
"""

FS_DEPTH = "#version 330 core\nvoid main(){}\n"

VS_LIT = """
#version 330 core
layout(location=0) in vec3 in_pos;
layout(location=1) in vec2 in_uv;
uniform mat4 uM;
uniform mat4 uVP;
uniform mat4 uLVP;
uniform sampler2D uHeightTex;
uniform float uBaseRadius;
uniform float uHeightScale;
out vec3 vWorldPos;
out vec2 vUV;
out vec4 vShadowPos;
float heightTex(vec2 uv){ return texture(uHeightTex, uv).r * 2.0 - 1.0; }
void main(){
    float h = heightTex(in_uv);
    float r = uBaseRadius * (1.0 + uHeightScale * h);
    vec3 world = (uM * vec4(in_pos * r, 1.0)).xyz;
    vWorldPos = world;
    vUV = in_uv;
    vShadowPos = uLVP * vec4(world, 1.0);
    gl_Position = uVP * vec4(world, 1.0);
}
"""

FS_LIT = """
#version 330 core
out vec4 fragColor;
in vec3 vWorldPos;
in vec2 vUV;
in vec4 vShadowPos;
uniform sampler2D uHeightTex;
uniform sampler2D uShadowMap;
uniform vec3 uCamPos;
uniform vec3 uLightDir;
uniform vec3 uLightI;
uniform vec3 uBaseColor;
uniform float uShininess;
uniform float uIsSatellite;
uniform float uBaseRadius;
uniform float uHeightScale;
float heightTex(vec2 uv){ return texture(uHeightTex, uv).r * 2.0 - 1.0; }

vec3 planetNormal(vec2 uv){
    float du = 1.0/2048.0;
    float dv = 1.0/1024.0;

    float h  = heightTex(uv);
    float hU = heightTex(uv + vec2(du,0));
    float hV = heightTex(uv + vec2(0,dv));

    float r  = uBaseRadius * (1.0 + uHeightScale * h);
    float rU = uBaseRadius * (1.0 + uHeightScale * hU);
    float rV = uBaseRadius * (1.0 + uHeightScale * hV);

    float theta = uv.x * 6.28318530718;
    float phi   = uv.y * 3.1415926535;

    vec3 dir  = vec3(sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta));
    vec3 dirU = vec3(sin(phi)*cos(theta+du*6.2831), cos(phi), sin(phi)*sin(theta+du*6.2831));
    vec3 dirV = vec3(sin(phi+dv*3.1415)*cos(theta), cos(phi+dv*3.1415), sin(phi+dv*3

    vec3 dirV = vec3(sin(phi+dv*3.1415)*cos(theta),
                     cos(phi+dv*3.1415),
                     sin(phi+dv*3.1415)*sin(theta));

    vec3 p  = dir  * r;
    vec3 pU = dirU * rU;
    vec3 pV = dirV * rV;

    vec3 tU = normalize(pU - p);
    vec3 tV = normalize(pV - p);
    vec3 n = normalize(cross(tU, tV));
    return n;
}

float shadowPCF(vec4 shadowPos){
    vec3 sp = shadowPos.xyz / shadowPos.w;
    vec3 uvz = sp * 0.5 + 0.5;

    if (uvz.x < 0.0 || uvz.x > 1.0 ||
        uvz.y < 0.0 || uvz.y > 1.0)
        return 1.0;

    float current = uvz.z;
    float bias = 0.0015;

    vec2 texel = 1.0 / vec2(textureSize(uShadowMap, 0));
    float occl = 0.0;

    for (int oy=-1; oy<=1; ++oy){
        for (int ox=-1; ox<=1; ++ox){
            float p = texture(uShadowMap, uvz.xy + vec2(ox,oy)*texel).r;
            occl += (current - bias > p) ? 1.0 : 0.0;
        }
    }
    return 1.0 - occl/9.0;
}

void main(){
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 L = normalize(-uLightDir);
    vec3 N;

    if (uIsSatellite > 0.5)
    {
        N = normalize(vWorldPos);
    }
    else
    {
        N = planetNormal(vUV);
    }

    float diff = max(dot(N, L), 0.0);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(R, V), 0.0), uShininess);
    float amb = 0.08;

    float sh = shadowPCF(vShadowPos);

    float lightTerm = amb + sh * diff;
    vec3 col = uBaseColor * lightTerm + sh * 0.25 * spec * vec3(1.0);

    col *= uLightI;
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0/2.2));

    fragColor = vec4(col, 1.0);
}
"""

FS_AXES = """
#version 330 core
out vec4 fragColor;
in vec2 v_uv;

uniform vec2 uResolution;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uWorldX;
uniform vec3 uWorldY;
uniform vec3 uWorldZ;

float sdSegment(vec2 p, vec2 a, vec2 b){
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba*h);
}

vec2 proj(vec3 w){
    return vec2(dot(w, uCamRight), dot(w, uCamUp));
}

vec3 drawAxis(vec2 uv, vec2 base, vec2 d, vec3 col, float len, float thick){
    if (length(d) < 1e-6)
        return vec3(0.0);

    vec2 nd = normalize(d);
    vec2 tip = base + nd * len;

    float aLine = 1.0 - smoothstep(thick, thick+0.002, sdSegment(uv, base, tip));

    vec2 perp = vec2(-nd.y, nd.x);
    vec2 a1 = tip - nd*0.020 + perp*0.012;
    vec2 a2 = tip - nd*0.020 - perp*0.012;

    float aHead = 1.0 - smoothstep(thick, thick+0.002,
        min(sdSegment(uv, tip, a1), sdSegment(uv, tip, a2)));

    return col * (aLine + aHead);
}

void main(){
    vec2 uv = v_uv;

    vec2 base = vec2(0.080, 0.150);
    float len = 0.110;
    float thick = 0.0018;

    vec2 ex = proj(uWorldX);
    vec2 ey = proj(uWorldY);
    vec2 ez = proj(uWorldZ);

    vec3 col = vec3(0.0);
    col += drawAxis(uv, base, ex, vec3(1.0, 0.10, 0.10), len, thick);
    col += drawAxis(uv, base, ey, vec3(0.10, 1.0, 0.10), len, thick);
    col += drawAxis(uv, base, ez, vec3(0.10, 0.40, 1.0), len, thick);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

# ============================================================
# RUN APPLICATION
# ============================================================

def run():
    import sys
    from PyQt6.QtGui import QSurfaceFormat

    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setVersion(3, 3)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(4)

    QSurfaceFormat.setDefaultFormat(fmt)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
