#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np
from OpenGL import GL
from OpenGL import GLUT
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

import data3d

_opencv_to_opengl = np.mat([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)


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


def _build_colored_program():
    vertex_shader = shaders.compileShader(  # 1.40 not supported on my PC ;(
        """
        #version 130
        uniform mat4 mvp;

        in vec3 point_position;
        in vec3 point_color_in;

        out vec3 point_color;

        void main() {
            gl_Position = mvp * vec4(point_position, 1.0);

            point_color = point_color_in;
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 130
        in vec3 point_color;

        out vec3 out_color;

        void main() {
            out_color = point_color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        vertex_shader, fragment_shader
    )


def _build_uncolored_program():
    vertex_shader = shaders.compileShader(
        """
        #version 130
        uniform mat4 mvp;

        in vec3 position;
        
        void main() {
            vec4 camera_space_position = mvp * vec4(position, 1.0);
            gl_Position = camera_space_position;
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 130
        uniform vec3 fixed_color;
        
        out vec3 out_color;

        void main() {
            out_color = fixed_color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        vertex_shader, fragment_shader
    )


def _build_textured_program():
    vertex_shader = shaders.compileShader(
        """
        #version 130
        uniform mat4 mvp;

        in vec3 point_position;
        in vec2 point_texcoords;
        
        out vec2 uv;
        
        void main() {
            vec4 camera_space_position = mvp * vec4(point_position, 1.0);
            gl_Position = camera_space_position;
            uv = point_texcoords;
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 130
        
        uniform sampler2D tex;
        
        in vec2 uv;
        
        out vec3 out_color;

        void main() {
            out_color = texture(tex, uv).rgb;
        }""",
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

        self._camera_track = tracked_cam_track
        self._camera_params = tracked_cam_parameters

        self._camera_track_buffer = vbo.VBO(
            np.array([pose.t_vec for pose in tracked_cam_track], dtype=np.float32)
        )

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

        self._colored_program = _build_colored_program()
        self._uncolored_program = _build_uncolored_program()
        self._textured_program = _build_textured_program()

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
        # a frame in which a tracked camera model and frustrum should be drawn
        # without interpolation
        tracked_cam_track_pos = int(tracked_cam_track_pos_float)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        aspect_ratio = GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH) / GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)

        mvp = _setup_projection(camera_fov_y, aspect_ratio, 0.1, 100) * \
              _setup_view(-camera_tr_vec, camera_rot_mat) * \
              _opencv_to_opengl

        tracked_cam_pose = self._camera_track[tracked_cam_track_pos]

        tracked_cam_mv = _setup_view(tracked_cam_pose.t_vec, -tracked_cam_pose.r_mat)
        tracked_cam_p = _setup_projection(
            self._camera_params.fov_y, self._camera_params.aspect_ratio,
            0.1, 50
        )

        self._render_cam_frustum(mvp * tracked_cam_mv * np.linalg.inv(tracked_cam_p))
        self._render_points(mvp)
        self._render_cam_track(mvp)
        self._render_cam_model(mvp * tracked_cam_mv)

        GLUT.glutSwapBuffers()

    def _render_points(self, mvp):
        shaders.glUseProgram(self._colored_program)

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._colored_program, 'mvp'),
            1, True, mvp)

        self._point_positions_buffer.bind()
        loc_pos = GL.glGetAttribLocation(self._colored_program, 'point_position')
        GL.glEnableVertexAttribArray(loc_pos)
        GL.glVertexAttribPointer(loc_pos, 3, GL.GL_FLOAT, False, 0, self._point_positions_buffer)

        self._point_colors_buffer.bind()
        loc_color = GL.glGetAttribLocation(self._colored_program, 'point_color_in')
        GL.glEnableVertexAttribArray(loc_color)
        GL.glVertexAttribPointer(loc_color, 3, GL.GL_FLOAT, False, 0, self._point_colors_buffer)

        GL.glDrawArrays(GL.GL_POINTS, 0, self._point_colors_buffer.size)

        GL.glDisableVertexAttribArray(loc_color)
        GL.glDisableVertexAttribArray(loc_pos)

        self._point_colors_buffer.unbind()
        self._point_positions_buffer.unbind()

        shaders.glUseProgram(0)

    def _render_cam_track(self, mvp):
        shaders.glUseProgram(self._uncolored_program)

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._uncolored_program, 'mvp'),
            1, True, mvp)

        GL.glUniform3f(
            GL.glGetUniformLocation(self._uncolored_program, 'fixed_color'),
            1.0, 1.0, 1.0
        )

        self._camera_track_buffer.bind()
        loc_pos = GL.glGetAttribLocation(self._uncolored_program, 'position')
        GL.glEnableVertexAttribArray(loc_pos)
        GL.glVertexAttribPointer(loc_pos, 3, GL.GL_FLOAT, False, 0, self._camera_track_buffer)

        GL.glDrawArrays(GL.GL_LINE_STRIP, 0, self._camera_track_buffer.size)

        GL.glDisableVertexAttribArray(loc_pos)

        self._camera_track_buffer.unbind()

        shaders.glUseProgram(0)

    def _render_cam_frustum(self, mvp):
        shaders.glUseProgram(self._uncolored_program)

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._uncolored_program, 'mvp'),
            1, True, mvp
        )

        GL.glUniform3f(
            GL.glGetUniformLocation(self._uncolored_program, 'fixed_color'),
            1.0, 1.0, 0.0
        )

        self._camera_pyramid_buffer.bind()
        loc_pos = GL.glGetAttribLocation(self._uncolored_program, 'position')
        GL.glEnableVertexAttribArray(loc_pos)
        GL.glVertexAttribPointer(loc_pos, 3, GL.GL_FLOAT, False, 0, self._camera_pyramid_buffer)

        GL.glDrawArrays(GL.GL_LINES, 0, self._camera_pyramid_buffer.size)

        GL.glDisableVertexAttribArray(loc_pos)

        self._camera_pyramid_buffer.unbind()

        shaders.glUseProgram(0)

    def _render_cam_model(self, mvp):
        shaders.glUseProgram(self._textured_program)

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._textured_program, 'mvp'),
            1, True, mvp)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._camera_model_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self._textured_program, 'tex'), 0)

        self._camera_model_vertices_buffer.bind()
        loc_pos = GL.glGetAttribLocation(self._textured_program, 'point_position')
        GL.glEnableVertexAttribArray(loc_pos)
        GL.glVertexAttribPointer(loc_pos, 3, GL.GL_FLOAT, False, 0, self._camera_model_vertices_buffer)

        self._camera_model_texcoords_buffer.bind()
        loc_color = GL.glGetAttribLocation(self._textured_program, 'point_texcoords')
        GL.glEnableVertexAttribArray(loc_color)
        GL.glVertexAttribPointer(loc_color, 2, GL.GL_FLOAT, False, 0, self._camera_model_texcoords_buffer)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._camera_model_vertices_buffer.size)

        GL.glDisableVertexAttribArray(loc_color)
        GL.glDisableVertexAttribArray(loc_pos)

        self._camera_model_texcoords_buffer.unbind()
        self._camera_model_vertices_buffer.unbind()

        shaders.glUseProgram(0)
