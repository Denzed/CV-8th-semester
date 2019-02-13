#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from typing import List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL import GLUT
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

import data3d


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
        self._camera_model_coords_buffer = vbo.VBO(
            np.array([0, 0, 0, 0, 0.1, 0, 0.1, 0, 0], dtype=np.float32)
        )
        self._camera_model_colors_buffer = vbo.VBO(
            np.array([1, 0, 0] * 3, dtype=np.float32)
        )

        self._camera_track = tracked_cam_track
        self._camera_params = tracked_cam_parameters

        self._camera_track_buffer = vbo.VBO(
            np.array([pose.t_vec for pose in tracked_cam_track], dtype=np.float32)
        )

        self._point_positions_buffer = vbo.VBO(np.array(point_cloud.points, dtype=np.float32))
        self._point_colors_buffer = vbo.VBO(np.array(point_cloud.colors, dtype=np.float32))

        self._colored_program = _build_colored_program()
        self._uncolored_program = _build_uncolored_program()

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

    @staticmethod
    def _setup_projection(fov_y, aspect_ratio, z_near, z_far):
        y_max = z_near * np.tan(fov_y)
        x_max = y_max * aspect_ratio

        return np.mat([
            [z_near / x_max, 0, 0, 0],
            [0, z_near / y_max, 0, 0],
            [0, 0, -(z_far + z_near) / (z_far - z_near), -2 * z_far * z_near / (z_far - z_near)],
            [0, 0, -1, 0],
        ], dtype=np.float32)

    @staticmethod
    def _setup_view(translation, rotation):
        return np.mat(np.block([
            [rotation, translation[np.newaxis].transpose()],
            [0, 0, 0, 1]
        ]), dtype=np.float32)

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

        mvp = self._setup_projection(camera_fov_y, aspect_ratio, 0.1, 100) * \
            self._setup_view(-camera_tr_vec, -camera_rot_mat)

        self._render_screen(mvp)
        self._render_tracked_cam(mvp, tracked_cam_track_pos)

        GLUT.glutSwapBuffers()

    def _render_screen(self, mvp):
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

    def _render_tracked_cam(self, mvp, frame):
        pose = self._camera_track[frame]

        tracked_cam_mv = self._setup_view(pose.t_vec, pose.r_mat)
        
        # TODO: pyramid

        shaders.glUseProgram(self._colored_program)

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._colored_program, 'mvp'),
            1, True, mvp * tracked_cam_mv)

        self._camera_model_coords_buffer.bind()
        loc_pos = GL.glGetAttribLocation(self._colored_program, 'point_position')
        GL.glEnableVertexAttribArray(loc_pos)
        GL.glVertexAttribPointer(loc_pos, 3, GL.GL_FLOAT, False, 0, self._camera_model_coords_buffer)

        self._camera_model_colors_buffer.bind()
        loc_color = GL.glGetAttribLocation(self._colored_program, 'point_color_in')
        GL.glEnableVertexAttribArray(loc_color)
        GL.glVertexAttribPointer(loc_color, 3, GL.GL_FLOAT, False, 0, self._camera_model_colors_buffer)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._camera_model_coords_buffer.size)

        GL.glDisableVertexAttribArray(loc_color)
        GL.glDisableVertexAttribArray(loc_pos)

        self._camera_model_colors_buffer.unbind()
        self._camera_model_coords_buffer.unbind()

        shaders.glUseProgram(0)