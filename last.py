# satellite_sim_gl.py 
# Полная OpenGL-версия: PyQt6 + QOpenGLWidget + PyOpenGL.
# Правки:
#  - очень тёмное небо (почти чёрно-синий градиент);
#  - много звёзд, хаотичное распределение, мерцание;
#  - часть звёзд заметно крупнее (giants) + редкие superGiants;
#  - снизу слева индикатор осей X/Y/Z с направлениями (стрелки);
#  - фикс IndentationError в _apply_controls;
#  - безопасный ctypes.c_void_p для указателей атрибутов.

from __future__ import annotations
import math
import os
import time
import ctypes
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL import GL

# -------------- camera.py --------------
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
        r = np.cross(self.WORLD_UP, fwd)
        n = float(np.linalg.norm(r)) or 1.0
        return (r / n).astype(np.float32)

    def up(self) -> np.ndarray:
        r = self.right(); fwd = self.forward()
        u = np.cross(fwd, r)
        n = float(np.linalg.norm(u)) or 1.0
        return (u / n).astype(np.float32)

    def position(self) -> np.ndarray:
        return self.target - self.forward() * self.distance

# -------------- orbit.py --------------
@dataclass
class OrbitParameters:
    radius: float
    angular_velocity: float  # рад/с

class OrbitSimulator:
    def __init__(self, params: OrbitParameters):
        self.params = params
        self._angle = 0.0

    def update(self, dt: float) -> None:
        self._angle = (self._angle + self.params.angular_velocity * dt) % (2 * math.pi)

    def set_radius(self, radius: float) -> None:
        self.params.radius = max(radius, 1.5)

    def set_speed(self, angular_velocity: float) -> None:
        self.params.angular_velocity = angular_velocity

    def set_angle(self, angle: float) -> None:
        self._angle = angle % (2 * math.pi)

    def position(self) -> np.ndarray:
        x = self.params.radius * math.cos(self._angle)
        z = self.params.radius * math.sin(self._angle)
        return np.array([x, 0.0, z], dtype=np.float32)

    def velocity(self) -> np.ndarray:
        vx = -self.params.radius * math.sin(self._angle) * self.params.angular_velocity
        vz =  self.params.radius * math.cos(self._angle) * self.params.angular_velocity
        return np.array([vx, 0.0, vz], dtype=np.float32)

# -------------- planet.py --------------
@dataclass
class PlanetConfig:
    base_radius: float = 2.0
    lat_steps: int = 128
    lon_steps: int = 256
    height_scale: float = 0.20

class Planet:
    def __init__(self, config: PlanetConfig | None = None):
        self.config = config or PlanetConfig()
        self.heights = np.zeros((self.config.lat_steps, self.config.lon_steps), dtype=np.float32)

    def generate(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        lat = np.linspace(-math.pi/2, math.pi/2, self.config.lat_steps, dtype=np.float32)
        lon = np.linspace(0.0, 2*math.pi, self.config.lon_steps, endpoint=False, dtype=np.float32)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        base_noise = self._fbm_noise(lat_grid, lon_grid, rng)
        crater_noise = self._crater_field(lat_grid, lon_grid, rng)
        h = (base_noise * 0.7 + crater_noise * 0.3).astype(np.float32)
        self.heights = np.clip(h, -1.0, 1.0)

    def _fbm_noise(self, lat_grid, lon_grid, rng) -> np.ndarray:
        freq, amp = 1.0, 1.0
        total = np.zeros_like(lat_grid, dtype=np.float32)
        norm = 0.0
        for _ in range(4):
            phase_lat = rng.uniform(0, 2*math.pi)
            phase_lon = rng.uniform(0, 2*math.pi)
            s = np.sin(lat_grid*freq + phase_lat) * np.cos(lon_grid*freq + phase_lon)
            total += (amp * s).astype(np.float32)
            norm += amp
            freq *= 2.0; amp *= 0.5
        total /= max(norm, 1e-6)
        return total.astype(np.float32)

    def _crater_field(self, lat_grid, lon_grid, rng) -> np.ndarray:
        cr = np.zeros_like(lat_grid, dtype=np.float32)
        count = int(rng.integers(12, 24))
        for _ in range(count):
            cl = float(rng.uniform(-math.pi/2, math.pi/2))
            co = float(rng.uniform(0, 2*math.pi))
            r = float(rng.uniform(0.05, 0.18))
            d = float(rng.uniform(-0.5, -0.1))
            dist = self._angular_distance(lat_grid, lon_grid, cl, co)
            crater = d * np.exp(-(dist*dist)/(2.0*r*r))
            cr = np.minimum(cr, crater.astype(np.float32))
        return cr.astype(np.float32)

    def _angular_distance(self, lat_grid, lon_grid, cl, co) -> np.ndarray:
        dlat = lat_grid - cl
        dlon = lon_grid - co
        a = (np.sin(dlat*0.5)**2).astype(np.float32) + np.cos(lat_grid)*math.cos(cl)*(np.sin(dlon*0.5)**2)
        a = np.clip(a.astype(np.float32), 0.0, 1.0)
        return (2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a) + 1e-6)).astype(np.float32)

# -------------- renderer (OpenGL) --------------
@dataclass
class RenderSettings:
    width: int = 256
    height: int = 144
    fov: float = math.radians(60)
    max_distance: float = 60.0
    planet_tolerance: float = 0.02
    planet_steps: int = 120

@dataclass
class Light:
    direction: np.ndarray
    intensity: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.98, 0.92], dtype=np.float32))

class GLRaymarchWidget(QOpenGLWidget):
    frame_ready = QtCore.pyqtSignal(object)  # np.uint8 RGB image when recording

    def __init__(self, planet: Planet, orbit: OrbitSimulator, light: Light, camera: Camera, settings: RenderSettings, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.planet = planet
        self.orbit = orbit
        self.light = light
        self.camera = camera
        self.settings = settings

        # Сцена: поворот/перенос
        self._scene_rot = np.identity(3, dtype=np.float32)
        self._scene_trans = np.zeros(3, dtype=np.float32)

        # Таймер обновления ~83 Гц
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(12)
        self._last_time = time.time()
        self._t0 = time.time()  # для uTime

        # Будут заполнены в initializeGL
        self._prog = None
        self._vao = None
        self._planet_tex = None

    # --- Scene controls ---
    def rotate_scene_euler_deg(self, dx: float, dy: float, dz: float) -> None:
        def Rx(a):
            c,s = math.cos(a), math.sin(a)
            return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)
        def Ry(a):
            c,s = math.cos(a), math.sin(a)
            return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)
        def Rz(a):
            c,s = math.cos(a), math.sin(a)
            return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
        R = Rz(math.radians(dz)) @ Ry(math.radians(dy)) @ Rx(math.radians(dx))
        self._scene_rot = (R @ self._scene_rot).astype(np.float32)

    def translate_scene(self, dx: float, dy: float, dz: float) -> None:
        self._scene_trans += np.array([dx, dy, dz], dtype=np.float32)

    # --- OpenGL lifecycle ---
    def initializeGL(self):
        GL.glDisable(GL.GL_DEPTH_TEST)
        # Шейдеры
        vs = self._compile_shader(GL.GL_VERTEX_SHADER, VERT_SRC)
        fs = self._compile_shader(GL.GL_FRAGMENT_SHADER, FRAG_SRC)
        self._prog = GL.glCreateProgram()
        GL.glAttachShader(self._prog, vs)
        GL.glAttachShader(self._prog, fs)
        GL.glLinkProgram(self._prog)
        if GL.glGetProgramiv(self._prog, GL.GL_LINK_STATUS) != GL.GL_TRUE:
            raise RuntimeError(GL.glGetProgramInfoLog(self._prog).decode())
        GL.glDeleteShader(vs); GL.glDeleteShader(fs)

        # Прямоугольник на весь экран
        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        quad = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype=np.float32)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
        # pos(2) + uv(2)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 16, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, 16, ctypes.c_void_p(8))

        # Текстура высот планеты
        self._planet_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._planet_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        self._upload_planet_heightmap()

    def _compile_shader(self, kind, src: str):
        s = GL.glCreateShader(kind)
        GL.glShaderSource(s, src)
        GL.glCompileShader(s)
        if GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
            raise RuntimeError(GL.glGetShaderInfoLog(s).decode())
        return s

    def _upload_planet_heightmap(self):
        # высоты в диапазоне [-1,1] → R8 (нормируем в [0,1])
        h = self.planet.heights.astype(np.float32)
        h01 = np.clip((h * 0.5 + 0.5), 0.0, 1.0)
        data = (h01 * 255).astype(np.uint8)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._planet_tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R8, data.shape[1], data.shape[0], 0, GL.GL_RED, GL.GL_UNSIGNED_BYTE, data)

    def paintGL(self):
        now = time.time()
        dt = min(max(now - self._last_time, 0.0), 0.050)
        self._last_time = now
        self.orbit.update(dt)

        # HiDPI viewport
        pr = float(self.devicePixelRatioF())
        w_pix = max(int(round(self.width() * pr)), 1)
        h_pix = max(int(round(self.height() * pr)), 1)
        GL.glViewport(0, 0, w_pix, h_pix)
        GL.glUseProgram(self._prog)

        # Юниформы
        def set3(name, v):
            loc = GL.glGetUniformLocation(self._prog, name)
            GL.glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))
        def set1(name, x):
            loc = GL.glGetUniformLocation(self._prog, name)
            GL.glUniform1f(loc, float(x))
        def seti(name, x):
            loc = GL.glGetUniformLocation(self._prog, name)
            GL.glUniform1i(loc, int(x))
        def setm3(name, m):
            loc = GL.glGetUniformLocation(self._prog, name)
            GL.glUniformMatrix3fv(loc, 1, GL.GL_TRUE, m.astype(np.float32))

        aspect = w_pix / h_pix
        tan_half_fov = math.tan(self.settings.fov / 2)

        cam_pos_w = self.camera.position().astype(np.float32)
        cam_fwd_w = self.camera.forward().astype(np.float32)
        cam_right_w = self.camera.right().astype(np.float32)
        cam_up_w = self.camera.up().astype(np.float32)

        sat_center = self.orbit.position().astype(np.float32)

        set3("uCamPos", cam_pos_w)
        set3("uCamFwd", cam_fwd_w)
        set3("uCamRight", cam_right_w)
        set3("uCamUp", cam_up_w)
        set1("uTanHalfFov", tan_half_fov)
        set1("uAspect", aspect)
        setm3("uSceneRot", self._scene_rot)
        set3("uSceneTrans", self._scene_trans)

        set3("uLightDir", self.light.direction.astype(np.float32))
        set3("uLightI", self.light.intensity.astype(np.float32))
        set3("uSatCenter", sat_center)
        set1("uSatRadius", 0.35)
        set1("uMaxDist", self.settings.max_distance)
        set1("uPlanetTol", self.settings.planet_tolerance)
        seti("uPlanetSteps", self.settings.planet_steps)
        set1("uBaseRadius", self.planet.config.base_radius)
        set1("uHeightScale", self.planet.config.height_scale)
        set1("uTime", now - self._t0)  # время для мерцания звёзд

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._planet_tex)
        seti("uHeightTex", 0)
        set1("uLatSteps", float(self.planet.config.lat_steps))
        set1("uLonSteps", float(self.planet.config.lon_steps))

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        if self._record_callback is not None:
            img = GL.glReadPixels(0, 0, w_pix, h_pix, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            frame = np.frombuffer(img, dtype=np.uint8).reshape(h_pix, w_pix, 3)
            frame = np.flipud(frame).copy()
            self._record_callback(frame)

    _record_callback = None
    def set_record_callback(self, fn):
        self._record_callback = fn

# ---- GLSL ----
VERT_SRC = """
#version 330 core
layout(location=0) in vec2 in_pos;
layout(location=1) in vec2 in_uv;
out vec2 v_uv;
void main(){
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAG_SRC = """
#version 330 core
out vec4 fragColor;
in vec2 v_uv;

uniform vec3 uCamPos, uCamFwd, uCamRight, uCamUp;
uniform float uTanHalfFov, uAspect;
uniform mat3 uSceneRot;
uniform vec3 uSceneTrans;

uniform sampler2D uHeightTex; // RED8 -> [0,1]
uniform float uLatSteps, uLonSteps;
uniform float uBaseRadius, uHeightScale;

uniform vec3 uLightDir; // направление, КУДА летит свет
uniform vec3 uLightI;
uniform vec3 uSatCenter;
uniform float uSatRadius;
uniform float uMaxDist;
uniform float uPlanetTol;
uniform int   uPlanetSteps;

uniform float uTime; // для мерцания

// ---------- helpers ----------
vec3 normalizeSafe(vec3 v){ float n = length(v); return n>0.0? v/n : vec3(0.0,0.0,1.0); }
vec2 normalizeSafe2(vec2 v){ float n = length(v); return n>0.0? v/n : vec2(1.0,0.0); }

float heightTex(vec2 ij){ // ij: lat,lon индексы в [0..N)
    vec2 uv = vec2(ij.y / uLonSteps, ij.x / uLatSteps);
    return texture(uHeightTex, uv).r * 2.0 - 1.0; // [-1,1]
}

float planetRadiusAtDir(vec3 dir){
    dir = normalizeSafe(dir);
    float theta = acos(clamp(dir.z, -1.0, 1.0));
    float phi = atan(dir.y, dir.x); if (phi < 0.0) phi += 6.28318530718;
    float lat = theta / 3.1415926535 * (uLatSteps - 1.0);
    float lon = phi   / 6.28318530718 * (uLonSteps);
    float lat0 = floor(lat); float lon0 = floor(lon);
    int ilat = int(clamp(lat0, 0.0, uLatSteps-1.0));
    int ilon = int(mod(lon0, uLonSteps));
    float h = heightTex(vec2(float(ilat), float(ilon)));
    return uBaseRadius * (1.0 + uHeightScale * h);
}

vec3 planetNormalApprox(vec3 dir){
    dir = normalizeSafe(dir);
    float theta = acos(clamp(dir.z, -1.0, 1.0));
    float phi = atan(dir.y, dir.x); if (phi < 0.0) phi += 6.28318530718;
    float lat = theta / 3.1415926535 * (uLatSteps - 1.0);
    float lon = phi   / 6.28318530718 * (uLonSteps);
    float lat0 = floor(lat); float lon0 = floor(lon);
    float lat1 = min(lat0 + 1.0, uLatSteps - 1.0);
    float lon1 = mod(lon0 + 1.0, uLonSteps);
    float u = lat - lat0; float v = lon - lon0;
    float h00 = heightTex(vec2(lat0, lon0));
    float h10 = heightTex(vec2(lat1, lon0));
    float h01 = heightTex(vec2(lat0, lon1));
    float h11 = heightTex(vec2(lat1, lon1));
    float dh_dlat = (1.0 - v) * (h10 - h00) + v * (h11 - h01);
    float dh_dlon = (1.0 - u) * (h01 - h00) + u * (h11 - h10);
    vec3 n = dir;
    vec3 tangent_lat = normalizeSafe(vec3(n.x, n.y, 0.0));
    if (length(tangent_lat) < 1e-6) tangent_lat = vec3(1.0, 0.0, 0.0);
    vec3 tangent_lon = normalizeSafe(cross(n, tangent_lat));
    n = n - tangent_lat * (dh_dlat * uHeightScale);
    n = n - tangent_lon * (dh_dlon * uHeightScale);
    return normalizeSafe(n);
}

bool intersectSphere(vec3 ro, vec3 rd, vec3 c, float r, out float t, out vec3 p){
    vec3 oc = ro - c; float b = dot(oc, rd); float c2 = dot(oc, oc) - r*r; float disc = b*b - c2;
    if (disc < 0.0) return false; float s = sqrt(max(disc, 0.0));
    t = -b - s; if (t <= 1e-4) t = -b + s; if (t <= 1e-4) return false; p = ro + rd*t; return true;
}

bool marchPlanet(vec3 ro, vec3 rd, out float t, out vec3 p){
    t = 0.0; float minStep = uPlanetTol * 0.25; float maxStep = uMaxDist / float(max(uPlanetSteps,1));
    for (int i=0;i<uPlanetSteps;i++){
        p = ro + rd*t; float r = length(p); float sr = planetRadiusAtDir(p);
        float dist = r - sr; if (dist < uPlanetTol) return true;
        float step = clamp(dist * 0.7, minStep, maxStep); t += step; if (t > uMaxDist) break;
    }
    return false;
}

bool shadowedBySatellite(vec3 point, vec3 L){
    float t; vec3 hp; return intersectSphere(point + L*0.02, L, uSatCenter, uSatRadius, t, hp);
}

bool shadowedByPlanet(vec3 point, vec3 L){
    float t; vec3 hp; return marchPlanet(point + L*0.05, L, t, hp);
}

// -------- Звёздное небо: много, хаотично, мерцают --------
float hash21(vec2 p){
    p = fract(p*vec2(123.34, 345.45));
    p += dot(p, p+34.345);
    return fract(p.x*p.y);
}
vec2 hash22(vec2 p){
    p = fract(p*vec2(443.8975, 441.423));
    p += dot(p, p+19.19);
    return fract(vec2(p.x*p.y, p.x+p.y));
}

float starDot(vec2 f, vec2 center, float r){
    float d = length(f - center);
    return smoothstep(r, 0.0, d);
}

vec3 starsDotsLayer(vec2 suv, vec2 scale, float existThresh, vec2 rRange){
    vec2 p = suv * scale;
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec3 col = vec3(0.0);

    for (int oy=-1; oy<=1; ++oy){
        for (int ox=-1; ox<=1; ++ox){
            vec2 cell = i + vec2(ox, oy);
            float rnd = hash21(cell);
            float exist = step(existThresh, rnd);
            vec2 center = hash22(cell);                   // случайный центр в клетке
            float r = mix(rRange.x, rRange.y, hash21(cell+2.7)); // размер

            float twPhase = rnd * 123.0;
            float twFreq  = 2.0 + 3.0 * hash21(cell+5.31);
            float twinkle = 0.6 + 0.4 * sin(twPhase + uTime * twFreq);

            vec3 starCol = vec3(0.85, 0.90, 1.0);
            col += starCol * starDot(f, center, r) * twinkle * exist;
        }
    }
    return col;
}

vec3 skyColor(vec3 dir){
    dir = normalizeSafe(dir);
    // Очень тёмный синий градиент
    float h = clamp(dir.y*0.5 + 0.5, 0.0, 1.0);
    vec3 bottom = vec3(0.000, 0.002, 0.008);
    vec3 top    = vec3(0.003, 0.012, 0.030);
    vec3 base = mix(bottom, top, h);

    // сфера → UV
    float phi = atan(dir.y, dir.x); if (phi < 0.0) phi += 6.28318530718;
    float theta = acos(clamp(dir.z, -1.0, 1.0));
    vec2 suv = vec2(phi / 6.28318530718, theta / 3.1415926535);

    // слои звёзд
    vec3 stars1 = starsDotsLayer(suv, vec2(1200.0, 700.0), 0.940, vec2(0.0035, 0.0068)); // крупные
    vec3 stars2 = starsDotsLayer(suv, vec2(2600.0, 1600.0), 0.965, vec2(0.0024, 0.0048)); // средние
    vec3 stars3 = starsDotsLayer(suv, vec2(4200.0, 2500.0), 0.985, vec2(0.0018, 0.0032)); // мелкие
    vec3 stars4 = starsDotsLayer(suv, vec2(7000.0, 4200.0), 0.992, vec2(0.0012, 0.0020)); // фон

    // редкие очень крупные
    vec3 giants      = starsDotsLayer(suv, vec2(1000.0, 600.0), 0.9970, vec2(0.0100, 0.0220));
    vec3 superGiants = starsDotsLayer(suv, vec2( 800.0, 480.0), 0.9985, vec2(0.0180, 0.0350));

    return clamp(base + stars1 + stars2 + stars3 + stars4 + giants + superGiants, 0.0, 1.0);
}

// -------- Индикатор осей внизу слева --------
float sdSegment(vec2 p, vec2 a, vec2 b){
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

vec3 drawAxis(vec2 uv, vec2 base, vec2 dir2d, vec3 col, float len, float thick){
    if (length(dir2d) < 1e-6) return vec3(0.0);
    vec2 d = normalizeSafe2(dir2d);
    vec2 tip = base + d * len;

    // стержень
    float dLine = sdSegment(uv, base, tip);
    float aLine = 1.0 - smoothstep(thick, thick + 0.002, dLine);

    // стрелка (две «усы»)
    vec2 perp = vec2(-d.y, d.x);
    vec2 a1 = tip - d * 0.020 + perp * 0.012;
    vec2 a2 = tip - d * 0.020 - perp * 0.012;
    float dA = min(sdSegment(uv, tip, a1), sdSegment(uv, tip, a2));
    float aHead = 1.0 - smoothstep(thick, thick + 0.002, dA);

    return col * (aLine + aHead);
}

vec3 axesOverlay(vec2 uv){
    // область внизу слева
    vec2 base = vec2(0.080, 0.150);
    float len = 0.110;      // длина осей в UV
    float thick = 0.0018;   // толщина линий

    // направления осей сцены в мировых координатах камеры
    vec3 ex_w = normalizeSafe(uSceneRot * vec3(1.0, 0.0, 0.0));
    vec3 ey_w = normalizeSafe(uSceneRot * vec3(0.0, 1.0, 0.0));
    vec3 ez_w = normalizeSafe(uSceneRot * vec3(0.0, 0.0, 1.0));

    // проекция на экранные оси (Right/Up камеры)
    vec2 ex = vec2(dot(ex_w, uCamRight), dot(ex_w, uCamUp));
    vec2 ey = vec2(dot(ey_w, uCamRight), dot(ey_w, uCamUp));
    vec2 ez = vec2(dot(ez_w, uCamRight), dot(ez_w, uCamUp));

    vec3 col = vec3(0.0);
    col += drawAxis(uv, base, ex, vec3(1.0, 0.10, 0.10), len, thick); // X — красный
    col += drawAxis(uv, base, ey, vec3(0.10, 1.0, 0.10), len, thick); // Y — зелёный
    col += drawAxis(uv, base, ez, vec3(0.10, 0.40, 1.0), len, thick); // Z — синий
    return clamp(col, 0.0, 1.0);
}

// -------- Шейдинг тел --------
vec3 shadePlanet(vec3 ro, vec3 rd, vec3 pos){
    vec3 normal = planetNormalApprox(pos);
    vec3 view_dir = -rd;
    vec3 L = normalizeSafe(-uLightDir);
    bool shadow_self = dot(normal, L) < 0.0; // терминатор
    bool shadow_sat = shadowedBySatellite(pos, L);
    float diffuse = 0.0; float specular = 0.0;
    if (!(shadow_self || shadow_sat)){
        diffuse = max(dot(normal, L), 0.0);
        vec3 reflectDir = reflect(-L, normal);
        specular = pow(max(dot(reflectDir, view_dir), 0.0), 32.0);
    }
    float ambient = 0.08;
    vec3 base = vec3(0.22, 0.65, 0.28);
    vec3 color = base * (ambient + diffuse) + 0.22 * specular;
    color *= uLightI;
    float altitude = length(pos) - uBaseRadius;
    float haze = clamp(altitude / (uBaseRadius * 0.6), 0.0, 1.0);
    vec3 sky = vec3(0.96, 0.96, 0.98);
    color = color * (1.0 - 0.10*haze) + sky * (0.06*haze);
    return clamp(color, 0.0, 1.0);
}

vec3 shadeSatellite(vec3 ro, vec3 rd, vec3 pos){
    vec3 normal = normalizeSafe(pos - uSatCenter);
    vec3 L = normalizeSafe(-uLightDir);
    vec3 view_dir = -rd;

    bool shadow_planet = shadowedByPlanet(pos, L);

    float diffuse = 0.0;
    float specular = 0.0;
    if (!shadow_planet) {
        diffuse = max(dot(normal, L), 0.0);
        vec3 reflectDir = reflect(-L, normal);
        specular = pow(max(dot(reflectDir, view_dir), 0.0), 64.0);
    }

    float rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 3.0) * 0.12;
    vec3 base = vec3(0.85, 0.85, 0.90);
    vec3 color = base * (0.08 + diffuse) + 0.4 * specular + rim;
    color *= uLightI;
    return clamp(color, 0.0, 1.0);
}

void main(){
    float ndc_x = (v_uv.x * 2.0 - 1.0) * uTanHalfFov * uAspect;
    float ndc_y = (1.0 - v_uv.y * 2.0) * uTanHalfFov;
    vec3 dirW = normalizeSafe(uCamFwd + ndc_x * uCamRight + ndc_y * uCamUp);

    vec3 origin = (transpose(uSceneRot) * (uCamPos)) - uSceneTrans;
    vec3 dir    = normalizeSafe(transpose(uSceneRot) * dirW);

    float tSat; vec3 pSat; bool hitSat = intersectSphere(origin, dir, uSatCenter, uSatRadius, tSat, pSat);
    float tPl; vec3 pPl;  bool hitPl  = marchPlanet(origin, dir, tPl, pPl);

    vec3 space = skyColor(dir);
    vec3 col;
    if (hitSat && hitPl){ col = (tSat < tPl) ? shadeSatellite(origin, dir, pSat) : shadePlanet(origin, dir, pPl); }
    else if (hitSat){ col = shadeSatellite(origin, dir, pSat); }
    else if (hitPl){ col = shadePlanet(origin, dir, pPl); }
    else { col = space; }

    // наложим оси поверх (аддитивно)
    col = clamp(col + axesOverlay(v_uv), 0.0, 1.0);

    // гамма 2.2
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0/2.2));
    fragColor = vec4(col, 1.0);
}
"""

# -------------- UI --------------
@dataclass
class UiState:
    seed: int = 1
    recording: bool = False
    frame_index: int = 0

class RenderWidget(QtWidgets.QWidget):
    def __init__(self, gl_widget: GLRaymarchWidget, parent=None):
        super().__init__(parent)
        self.gl_widget = gl_widget
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(gl_widget)

class ControlPanel(QtWidgets.QWidget):
    parameters_changed = QtCore.pyqtSignal()
    seed_changed = QtCore.pyqtSignal(int)
    scene_rotated = QtCore.pyqtSignal(float, float, float)
    scene_translated = QtCore.pyqtSignal(float, float, float)
    record_toggled = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._create_translation_group())
        layout.addWidget(self._create_rotation_group())
        layout.addWidget(self._create_scaling_group())
        layout.addWidget(self._create_orbit_group())
        layout.addStretch(1)

    def _spin(self, value=0.0, minimum=-10.0, maximum=10.0, step=0.1) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox(); box.setRange(minimum, maximum); box.setDecimals(2); box.setSingleStep(step); box.setValue(value); return box

    def _create_translation_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Перенос СЦЕНЫ")
        form = QtWidgets.QGridLayout(group)
        self.dx, self.dy, self.dz = [self._spin() for _ in range(3)]
        form.addWidget(QtWidgets.QLabel("dX"), 0, 0); form.addWidget(self.dx, 0, 1)
        form.addWidget(QtWidgets.QLabel("dY"), 1, 0); form.addWidget(self.dy, 1, 1)
        form.addWidget(QtWidgets.QLabel("dZ"), 2, 0); form.addWidget(self.dz, 2, 1)
        btn = QtWidgets.QPushButton("Перенести"); btn.clicked.connect(self._emit_translate); form.addWidget(btn, 3, 0, 1, 2)
        return group

    def _emit_translate(self):
        self.scene_translated.emit(self.dx.value(), self.dy.value(), self.dz.value())
        self.dx.setValue(0.0); self.dy.setValue(0.0); self.dz.setValue(0.0)

    def _create_rotation_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Поворот СЦЕНЫ (Δуглы, градусы)")
        form = QtWidgets.QGridLayout(group)
        self.angle_x = self._spin(maximum=360)
        self.angle_y = self._spin(maximum=360)
        self.angle_z = self._spin(maximum=360)
        form.addWidget(QtWidgets.QLabel("X"), 0, 0); form.addWidget(self.angle_x, 0, 1)
        form.addWidget(QtWidgets.QLabel("Y"), 1, 0); form.addWidget(self.angle_y, 1, 1)
        form.addWidget(QtWidgets.QLabel("Z"), 2, 0); form.addWidget(self.angle_z, 2, 1)
        btn = QtWidgets.QPushButton("Повернуть сцену"); btn.clicked.connect(self._emit_rotate); form.addWidget(btn, 3, 0, 1, 2)
        return group

    def _emit_rotate(self):
        dx, dy, dz = self.angle_x.value(), self.angle_y.value(), self.angle_z.value()
        self.scene_rotated.emit(dx, dy, dz)
        self.angle_x.setValue(0.0); self.angle_y.setValue(0.0); self.angle_z.setValue(0.0)

    def _create_scaling_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Масштабирование (равномерно)")
        form = QtWidgets.QGridLayout(group)
        self.scale_box = self._spin(value=1.0, minimum=0.1, maximum=10.0, step=0.1)
        form.addWidget(QtWidgets.QLabel("scale"), 0, 0); form.addWidget(self.scale_box, 0, 1)
        btn = QtWidgets.QPushButton("Масштабировать"); btn.clicked.connect(self.parameters_changed.emit)
        form.addWidget(btn, 1, 0, 1, 2)
        return group

    def _create_orbit_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Параметры орбиты и Seed")
        form = QtWidgets.QGridLayout(group)
        self.orbit_radius = self._spin(value=4.0, minimum=1.5, maximum=12.0)
        self.orbit_speed = self._spin(value=1.2, minimum=0.0, maximum=6.0)
        self.seed_box = QtWidgets.QSpinBox(); self.seed_box.setRange(0, 10000); self.seed_box.setValue(1)
        form.addWidget(QtWidgets.QLabel("Радиус"), 0, 0); form.addWidget(self.orbit_radius, 0, 1)
        form.addWidget(QtWidgets.QLabel("Скорость (рад/с)"), 1, 0); form.addWidget(self.orbit_speed, 1, 1)
        form.addWidget(QtWidgets.QLabel("Seed"), 2, 0); form.addWidget(self.seed_box, 2, 1)
        btn_apply = QtWidgets.QPushButton("Применить"); btn_apply.clicked.connect(self._apply_clicked); form.addWidget(btn_apply, 3, 0, 1, 2)
        self.record_button = QtWidgets.QPushButton("Начать запись"); self.record_button.setCheckable(True); self.record_button.toggled.connect(self.record_toggled); form.addWidget(self.record_button, 4, 0, 1, 2)
        return group

    def _apply_clicked(self):
        self.parameters_changed.emit(); self.seed_changed.emit(self.seed_box.value())

class MainWindow(QtWidgets.QMainWindow):
    BASE_WIDTH = 1200
    BASE_HEIGHT = 720

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Visualiser (OpenGL, scene transform)")
        self.setMinimumSize(self.BASE_WIDTH, self.BASE_HEIGHT)
        self.resize(self.BASE_WIDTH, self.BASE_HEIGHT)

        self.ui_state = UiState()

        # Камера фиксирована: смотрим «внутрь» по +Z (yaw=+90°), Y — вверх
        self.camera = Camera(distance=8.0, yaw=math.radians(90.0), pitch=0.0)
        self.camera.target[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.planet = Planet(PlanetConfig())
        self.planet.generate(self.ui_state.seed)
        self.orbit = OrbitSimulator(OrbitParameters(radius=4.0, angular_velocity=1.2))
        light_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        self.light = Light(light_dir, np.array([1.0, 0.98, 0.92], dtype=np.float32))
        self.settings = RenderSettings(width=256, height=144, planet_steps=48, planet_tolerance=0.05, max_distance=32.0)

        self.gl = GLRaymarchWidget(self.planet, self.orbit, self.light, self.camera, self.settings)

        self.panel = ControlPanel()
        self.render_widget = RenderWidget(self.gl)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.panel)
        splitter.addWidget(self.render_widget)
        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1)
        self.panel.setMinimumWidth(280)
        splitter.setSizes([280, self.BASE_WIDTH - 280])
        self.setCentralWidget(splitter)

        # Хоткеи качества
        QtGui.QShortcut(QtGui.QKeySequence("0"), self, activated=lambda: self._set_quality("ULTRA"))
        QtGui.QShortcut(QtGui.QKeySequence("1"), self, activated=lambda: self._set_quality("FAST"))
        QtGui.QShortcut(QtGui.QKeySequence("2"), self, activated=lambda: self._set_quality("BALANCED"))
        QtGui.QShortcut(QtGui.QKeySequence("3"), self, activated=lambda: self._set_quality("HIGH"))

        # Сигналы
        self.panel.parameters_changed.connect(self._apply_controls)
        self.panel.seed_changed.connect(self._regenerate_planet)
        self.panel.scene_rotated.connect(self._rotate_scene)
        self.panel.scene_translated.connect(self._translate_scene)
        self.panel.record_toggled.connect(self._toggle_recording)

        self.statusBar().showMessage(
            "Готово (OpenGL). Качество: 0=ULTRA, 1=FAST (по умолчанию), 2=BALANCED, 3=HIGH. Поворот/перенос — у СЦЕНЫ."
        )

        # Центрирование
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
            self.settings.planet_steps, self.settings.planet_tolerance, self.settings.max_distance = 40, 0.06, 28.0
        elif preset == "FAST":
            self.settings.width, self.settings.height = 256, 144
            self.settings.planet_steps, self.settings.planet_tolerance, self.settings.max_distance = 48, 0.05, 32.0
        elif preset == "HIGH":
            self.settings.width, self.settings.height = 800, 450
            self.settings.planet_steps, self.settings.planet_tolerance, self.settings.max_distance = 200, 0.01, 80.0
        else:
            self.settings.width, self.settings.height = 480, 270
            self.settings.planet_steps, self.settings.planet_tolerance, self.settings.max_distance = 120, 0.02, 60.0
        self.statusBar().showMessage(f"Качество: {preset}. Поворот/перенос — у СЦЕНЫ.")

    def _apply_controls(self) -> None:
        # Масштабирование как зум
        scale = max(float(self.panel.scale_box.value()), 0.1)
        self.camera.distance = float(np.clip(self.camera.distance / scale, 1.5, 50.0))
        self.panel.scale_box.setValue(1.0)
        # Орбита
        self.orbit.set_radius(self.panel.orbit_radius.value())
        self.orbit.set_speed(self.panel.orbit_speed.value())  # <-- без лишнего отступа
        # Перегенерируем высоты → обновим текстуру
        self._upload_planet_tex()

    def _upload_planet_tex(self):
        if self.gl._planet_tex is not None:
            self.gl._upload_planet_heightmap()

    def _rotate_scene(self, dx: float, dy: float, dz: float) -> None:
        self.gl.rotate_scene_euler_deg(dx, dy, dz)

    def _translate_scene(self, dx: float, dy: float, dz: float) -> None:
        self.gl.translate_scene(dx, dy, dz)

    def _regenerate_planet(self, seed: int) -> None:
        self.planet.generate(seed); self._upload_planet_tex()

    def _toggle_recording(self, active: bool) -> None:
        self.ui_state.recording = active
        self.panel.record_button.setText("Остановить запись" if active else "Начать запись")
        if active:
            self.ui_state.frame_index = 0
            Path("frames").mkdir(exist_ok=True)
            self.gl.set_record_callback(self._save_frame)
        else:
            self.gl.set_record_callback(None)

    def _save_frame(self, image: np.ndarray) -> None:
        out = Path("frames"); out.mkdir(exist_ok=True)
        fname = out / f"frame_{self.ui_state.frame_index:05d}.png"
        QtGui.QImage(image.data, image.shape[1], image.shape[0], 3*image.shape[1], QtGui.QImage.Format.Format_RGB888).save(str(fname))
        self.ui_state.frame_index += 1

# -------------- Entry point --------------
def run() -> None:
    import sys
    os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run()
