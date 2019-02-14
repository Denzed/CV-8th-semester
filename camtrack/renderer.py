#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from typing import List, Tuple

import cv2
import numpy as np
from pyquaternion import Quaternion
from OpenGL import GL
from OpenGL import GLUT
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

import data3d


def uses_program(prog_name, setup_mvp, **used_buffer_info):
    def _uses_program(f):
        def wrapper(self, *args, **kwargs):
            program = getattr(self, prog_name)
            shaders.glUseProgram(program)

            if setup_mvp:
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(program, 'mvp'), 1, True,
                    kwargs['mvp'] if 'mvp' in kwargs else args[0]
                )

            buffers = []

            for (name, (buff_name, cnt, typ)) in used_buffer_info.items():
                buffers.append((
                    GL.glGetAttribLocation(program, name),
                    getattr(self, buff_name),
                    cnt,
                    typ
                ))

            for (loc, buffer, cnt, typ) in buffers:
                buffer.bind()
                GL.glEnableVertexAttribArray(loc)
                GL.glVertexAttribPointer(loc, cnt, typ, False, 0, buffer)

            result = f(self, *args, **kwargs)

            for (loc, buffer, _, _) in reversed(buffers):
                GL.glDisableVertexAttribArray(loc)
                buffer.unbind()

            shaders.glUseProgram(0)

            return result

        return wrapper
    return _uses_program


_opencv_to_opengl = np.mat([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)


def _interpolate_cam_pose(t: float, p0: data3d.Pose, p1: data3d.Pose):
    def to_quat(m):
        qr = np.sqrt(1 + m[0][0] + m[1][1] + m[2][2]) / 2
        return Quaternion(np.array([
            qr,
            (m[2][1] - m[1][2]) / 4 / qr,
            (m[0][2] - m[2][0]) / 4 / qr,
            (m[1][0] - m[0][1]) / 4 / qr
        ]))

    return data3d.Pose(
        Quaternion.slerp(to_quat(p0.r_mat), to_quat(p1.r_mat), t).rotation_matrix,
        p0.t_vec * (1 - t) + p1.t_vec * t
    )


def _setup_projection(fov_y, aspect_ratio, z_near, z_far):
    y_max = z_near * np.tan(fov_y)
    x_max = y_max * aspect_ratio

    return np.mat([
        [z_near / x_max, 0, 0, 0],
        [0, z_near / y_max, 0, 0],
        [0, 0, -(z_far + z_near) / (z_far - z_near), -2 * z_far * z_near / (z_far - z_near)],
        [0, 0, -1, 0],
    ], dtype=np.float32)


def _setup_view(translation, rotation):
    return np.mat(np.block([
        [rotation, translation[np.newaxis].transpose()],
        [0, 0, 0, 1]
    ]), dtype=np.float32)


def _build_general_program(*defines):
    defines = ''.join(map(lambda d: f'#define {d}\n', defines))

    vertex_shader = shaders.compileShader(  # 1.40 not supported on my PC ;(
        f"""
        #version 130
        {defines}

        uniform mat4 mvp;

        in vec3 point_position;

        #if defined(COLORED_POINTS)
        in vec3 point_color_in;
        out vec3 point_color;
        #elif defined(WITH_TEXTURE)
        in vec2 point_texcoords;
        out vec2 uv;
        #endif

        void main() {{
            gl_Position = mvp * vec4(point_position, 1.0);
            
            #if defined(COLORED_POINTS)
            point_color = point_color_in;
            #elif defined(WITH_TEXTURE)
            uv = point_texcoords;
            #endif
        }}""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        f"""
        #version 130
        
        {defines}
        
        #if defined(COLORED_POINTS)
        in vec3 point_color;
        #elif defined(FIXED_COLOR)
        uniform vec3 point_color;        
        #elif defined(WITH_TEXTURE)
        uniform sampler2D tex;
        in vec2 uv;
        #define point_color texture(tex, uv).rgb        
        #endif

        out vec3 out_color;

        void main() {{
            out_color = point_color;
        }}""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        vertex_shader, fragment_shader
    )


class CameraTrackRenderer:

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.

        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustrum
        :param point_cloud: colored point cloud
        """
        self._load_camera_model(cam_model_files)

        self._fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D,
            self._camera_model_tex, 0
        )
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        self._camera_track = tracked_cam_track
        self._camera_params = tracked_cam_parameters

        poses = [pose.t_vec for pose in tracked_cam_track]
        self._camera_track_buffer = vbo.VBO(np.array(list(zip(poses, poses[1:])), dtype=np.float32))

        pyramid = [
            [[-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]],
            [[-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]]
        ]

        self._camera_pyramid_buffer = vbo.VBO(
            np.array([
                pyramid[0][0], pyramid[0][1],
                pyramid[0][1], pyramid[0][2],
                pyramid[0][2], pyramid[0][3],
                pyramid[0][3], pyramid[0][0],
                pyramid[1][0], pyramid[1][1],
                pyramid[1][1], pyramid[1][2],
                pyramid[1][2], pyramid[1][3],
                pyramid[1][3], pyramid[1][0],
                pyramid[0][0], pyramid[1][0],
                pyramid[0][1], pyramid[1][1],
                pyramid[0][2], pyramid[1][2],
                pyramid[0][3], pyramid[1][3],
            ], dtype=np.float32)
        )

        self._point_positions_buffer = vbo.VBO(np.array(point_cloud.points, dtype=np.float32))
        self._point_colors_buffer = vbo.VBO(np.array(point_cloud.colors, dtype=np.float32))

        self._colored_program = _build_general_program('COLORED_POINTS')
        self._uncolored_program = _build_general_program('FIXED_COLOR')
        self._textured_program = _build_general_program('WITH_TEXTURE')

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def _load_camera_model(self, cam_model_files: Tuple[str, str]):
        vertices = []
        texcoords = []
        faces = []

        for line in open(cam_model_files[0]):
            t, *coords = line.split()
            if t == 'v':
                vertices.append([-float(coords[0]), float(coords[1]), float(coords[2])])
            elif t == 'vt':
                texcoords.append(list(map(float, coords)))
            elif t == 'f':
                faces.append(list(map(int, coords)))

        self._camera_model_vertices_buffer = vbo.VBO(
            np.array(sum((vertices[i - 1] for i in sum(faces, [])), []), dtype=np.float32)
        )
        self._camera_model_texcoords_buffer = vbo.VBO(
            np.array(sum((texcoords[i - 1] for i in sum(faces, [])), []), dtype=np.float32)
        )
        self._camera_model_tex = GL.glGenTextures(1)

        image_data = cv2.imread(cam_model_files[1])
        height, width, _ = image_data.shape

        self._camera_model_texture_size = (width, height)
        self._camera_screen_coords = np.array([
            (0.113281 * width, 0.0313 * height), (0.28125 * width, 0.1485 * height)
        ], dtype=np.int)

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._camera_model_tex)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGB, width, height, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
            np.flipud(image_data)
        )
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and 
        tracked camera position

        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustrum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """
        tracked_cam_track_pos = int(tracked_cam_track_pos_float)
        tracked_cam_t = tracked_cam_track_pos_float - tracked_cam_track_pos
        self._camera_track.append(self._camera_track[-1])
        tracked_cam_pose = _interpolate_cam_pose(
            tracked_cam_t,
            *self._camera_track[tracked_cam_track_pos:tracked_cam_track_pos + 2]
        )

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        aspect_ratio = GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH) / GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)

        mvp = _setup_projection(camera_fov_y, aspect_ratio, 0.1, 100) * \
            _setup_view(-camera_tr_vec, camera_rot_mat) * \
            _opencv_to_opengl

        tracked_cam_mv = _setup_view(tracked_cam_pose.t_vec, -tracked_cam_pose.r_mat)
        tracked_cam_p = _setup_projection(
            self._camera_params.fov_y, self._camera_params.aspect_ratio,
            0.1, 50
        )

        self._render_cam_frustum(mvp * tracked_cam_mv * np.linalg.inv(tracked_cam_p))
        self._render_points(mvp)
        self._render_cam_track(mvp)

        screen_mvp = _setup_projection(
            self._camera_params.fov_y, self._camera_params.aspect_ratio, 0.1, 50
        ) * _setup_view(-tracked_cam_pose.t_vec, tracked_cam_pose.r_mat) * _opencv_to_opengl
        self._render_cam(mvp * tracked_cam_mv, screen_mvp)

        GLUT.glutSwapBuffers()

    @uses_program(
        '_colored_program',
        True,
        point_position=('_point_positions_buffer', 3, GL.GL_FLOAT),
        point_color_in=('_point_colors_buffer', 3, GL.GL_FLOAT)
    )
    def _render_points(self, mvp):
        GL.glDrawArrays(GL.GL_POINTS, 0, self._point_colors_buffer.size // 4)

    @uses_program(
        '_uncolored_program',
        True,
        point_position=('_camera_track_buffer', 3, GL.GL_FLOAT)
    )
    def _render_cam_track(self, mvp):
        GL.glUniform3f(
            GL.glGetUniformLocation(self._uncolored_program, 'point_color'),
            1.0, 1.0, 1.0
        )

        GL.glDrawArrays(GL.GL_LINES, 0, self._camera_track_buffer.size // 4)

    @uses_program(
        '_uncolored_program',
        True,
        point_position=('_camera_pyramid_buffer', 3, GL.GL_FLOAT)
    )
    def _render_cam_frustum(self, mvp):
        GL.glUniform3f(
            GL.glGetUniformLocation(self._uncolored_program, 'point_color'),
            1.0, 1.0, 0.0
        )

        GL.glDrawArrays(GL.GL_LINES, 0, self._camera_pyramid_buffer.size // 4)

    @uses_program(
        '_textured_program',
        True,
        point_position=('_camera_model_vertices_buffer', 3, GL.GL_FLOAT),
        point_texcoords=('_camera_model_texcoords_buffer', 2, GL.GL_FLOAT)
    )
    def _render_cam_model(self, mvp):
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._camera_model_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self._textured_program, 'tex'), 0)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._camera_model_vertices_buffer.size // 4)

    def _render_cam(self, mvp, screen_mvp):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        screen_x0, screen_y0, screen_w, screen_h = (
            self._camera_screen_coords[0][0],
            self._camera_screen_coords[0][1],
            self._camera_screen_coords[1][0] - self._camera_screen_coords[0][0],
            self._camera_screen_coords[1][1] - self._camera_screen_coords[0][1]
        )
        GL.glViewport(screen_x0, screen_y0, screen_w, screen_h)

        GL.glEnable(GL.GL_SCISSOR_TEST)  # clearing camera screen from previous drawings
        GL.glScissor(screen_x0, screen_y0, screen_w, screen_h)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glDisable(GL.GL_SCISSOR_TEST)

        self._render_points(screen_mvp)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH), GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT))

        self._render_cam_model(mvp)
