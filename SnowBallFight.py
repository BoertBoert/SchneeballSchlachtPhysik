
import time

import glfw
import numpy as np
from OpenGL.GL import *
import tkinter as tk
from tkinter import messagebox



# GLFW
if not glfw.init():
    raise Exception("GLFW cannot be initialized!")

monitor = glfw.get_primary_monitor()
video_mode = glfw.get_video_mode(monitor)
screen_width, screen_height = video_mode.size.width, video_mode.size.height

#Voll-Screen modus
game_window = glfw.create_window(screen_width, screen_height, "Snowball Fight", monitor, None)
if not game_window:
    glfw.terminate()
    raise Exception("GLFW window cannot be created!")

glfw.make_context_current(game_window)


# Globale Variabeln
keys = set()  #Knopfdruck
frame_count = 0
fps_timer = time.time()
frame_index = 0
current_frame_rate = 60
TARGET_FPS = 60
FRAME_TIME = 1.0 / TARGET_FPS
STAGE_TILE_SIZE = 36
game_seconds = 0


# Key callback funktion
def key_callback(*args):
    global keys
    key = args[1]
    action = args[3]
    if action == glfw.PRESS:
        keys.add(key)
    elif action == glfw.RELEASE:
        keys.discard(key)

glfw.set_key_callback(game_window, key_callback)



class Ui:
    def __init__(self):
        self.stage = stage
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.bar_size = 500  # Grösse Prozentanzeige einfärbung
        self.percentages = np.array([])
        self.game_finished = False

        #Koordinaten
        self.top_left_xy = np.array([screen_width/2 - self.bar_size, screen_height - 50])  # Top-left corner near the top of the screen
        self.bottom_right_xy = np.array([screen_width/2 + self.bar_size, screen_height - 100])  # Bottom-right corner, slightly lower

        self.calculate_colour_percentage()

    def calculate_colour_percentage(self):
        #Prozentzahl der Farben berechnen
        empty_amount = 0
        w_amount = 0
        b_amount = 0

        for colour in self.stage.colour_vertices_info:
            if colour[1] == "w":
                w_amount += 1
            elif colour[1] == "b":
                b_amount += 1
            else:
                empty_amount += 1

        total = empty_amount + w_amount + b_amount
        if total > 0:
            self.percentages = np.array([w_amount, empty_amount, b_amount]) / total
        else:
            self.percentages = np.array([0, 1, 0])  # Default to 100% empty if total is zero

    def render_rectangle(self, top_left_xy, bottom_right_xy):
        #2d Rechteck
        glBegin(GL_POLYGON)
        glVertex2i(int(top_left_xy[0]), int(top_left_xy[1]))
        glVertex2i(int(top_left_xy[0]), int(bottom_right_xy[1]))
        glVertex2i(int(bottom_right_xy[0]), int(bottom_right_xy[1]))
        glVertex2i(int(bottom_right_xy[0]), int(top_left_xy[1]))
        glEnd()

    def render_percentage_bar(self):
        #Prozentbalken zeichnen
        self.calculate_colour_percentage()

        # orthographic projection for 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.screen_width, 0, self.screen_height, -1, 1)  # Use dynamic screen dimensions
        glDisable(GL_DEPTH_TEST)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        frame = np.array([-10, 10])
        #Rahmen Schwarz
        glColor3f(0, 0, 0)
        self.render_rectangle(self.top_left_xy + frame, self.bottom_right_xy - frame)

        #Balkenbreite
        bar_width = np.abs(self.top_left_xy[0] - self.bottom_right_xy[0])
        scaled_percentages = self.percentages * bar_width

        #Farbe der Segmente
        colors = [
            (1, 1, 1),  # Light blue for 'w'
            (0.3, 0.3, 0.3),  # Dark gray for 'empty'
            (173 / 255, 216 / 255, 230 / 255),  # Black for 'b'
        ]

        start_x = self.top_left_xy[0]
        bottom_y = self.bottom_right_xy[1]

        for i, color in enumerate(colors):
            glColor3f(*color)
            end_x = start_x + scaled_percentages[i]
            self.render_rectangle([start_x, self.top_left_xy[1]], [end_x, bottom_y])
            start_x = end_x  # Update start_x for the next segment

        #Isometrische Projektion zurücksetzten
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glEnable(GL_DEPTH_TEST)

    def render_clock_circle(self):
        """Render a clock circle filled proportionally to the elapsed time. (ganze Funktion mit Chatgpt, zeit sparen)"""
        max_seconds = 120  # Assuming the clock runs for 60 seconds
        angle = (game_seconds / max_seconds) * 360  # Calculate the angle of the filled portion

        # Set up orthographic projection for 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.screen_width, 0, self.screen_height, -1, 1)
        glDisable(GL_DEPTH_TEST)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        center = (self.top_left_xy + self.bottom_right_xy)/2
        center += np.array([800, 0])
        # Circle center and radius
        center_x = center[0]
        center_y = center[1]
        radius = 50

        # Render the filled portion of the circle
        if game_seconds >= max_seconds:
            self.game_finished = True
        elif max_seconds - game_seconds <= max_seconds//5:
            glColor3f(1.0, 0.0, 0.0)  # Red for last 20 seconds
        else:
            glColor3f(1.0, 1.0, 1.0)  # White otherwise

        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(center_x, center_y)  # Center of the circle
        for i in range(0, int(angle) + 1):
            theta = np.radians(i)
            x = center_x + radius * np.cos(np.pi/2 - theta)
            y = center_y + radius * np.sin(np.pi/2 - theta)
            glVertex2f(x, y)
        glEnd()

        # Render the circle outline
        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(5)
        glBegin(GL_LINE_LOOP)
        for i in range(360):
            theta = np.radians(i)
            x = center_x + radius * np.cos(theta)
            y = center_y + radius * np.sin(theta)
            glVertex2f(x, y)
        glEnd()

        # Restore previous projection and model-view matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glEnable(GL_DEPTH_TEST)


class Player:
    def __init__(self, xyz, size, facing_angle, team):
        self.player_and_vertex_distance = size / 2 + STAGE_TILE_SIZE / 2
        self.vertex_minus_player_distance = STAGE_TILE_SIZE / np.sqrt(2) - size / 2
        self.speed = 5
        self.direction_change = 0
        self.movement_vector = np.array([0, 0, 0])
        self.spawn_xyz = np.array(xyz, dtype=np.float32)
        self.xyz = np.copy(self.spawn_xyz)
        self.size = size
        self.health = 3
        self.facing_angle = facing_angle
        self.respawn_tick = 0
        self.rotation_matrix = None
        self.movement_permission = True
        self.team = team
        self.next_allowed_shot_frame = 0
        self.corners = np.array([
            [+self.size / 2, +self.size, +self.size / 2],  # Top-front-right
            [+self.size / 2, +self.size, -self.size / 2],  # Top-back-right
            [-self.size / 2, +self.size, -self.size / 2],  # Top-back-left
            [-self.size / 2, +self.size, +self.size / 2],  # Top-front-left
            [+self.size / 2, 0, +self.size / 2],  # Bottom-front-right
            [+self.size / 2, 0, -self.size / 2],  # Bottom-back-right
            [-self.size / 2, 0, -self.size / 2],  # Bottom-back-left
            [-self.size / 2, 0, +self.size / 2]  # Bottom-front-left
        ])
        self.faces = [
            [0, 1, 2, 3],  # Top face
            [4, 5, 6, 7],  # Bottom face
            [0, 4, 7, 3],  # Front face
            [1, 5, 6, 2],  # Back face
            [0, 1, 5, 4],  # Right face
            [3, 2, 6, 7]  # Left face
        ]
        self.colours = [
            (0.6, 0, 0),
            (0.6, 0.6, 0),
            (0.6, 0, 0.6),
            (0, 0.6, 0),
            (0, 0.6, 0.6),
            (0, 0, 0.6),
            (0.6, 0.6, 0.6),
            (0, 0, 0)
        ]

    def move_cube(self):
        #Rotation berechnen
        angle = np.radians(self.facing_angle)
        self.rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]])

        #Bewegung
        if self.movement_permission:
            self.movement_vector = np.dot([0, self.movement_vector[1], -self.speed],
                                        self.rotation_matrix.T)  # Forward in local space
        else:
            self.movement_vector = np.array([0, 0, 0])
        #Neue Position
        self.xyz += self.movement_vector
        self.collision()
        self.free_fall()
        self.respawn()
        self.render_player()

    def render_player(self):
        #Koordinaten der Ecken berechnen
        rotated_corners = np.dot(self.corners, self.rotation_matrix.T) + self.xyz

        colour_counter = 0
        for face in self.faces:
            glBegin(GL_POLYGON)
            glColor3f(*self.colours[colour_counter % len(self.colours)])
            for vertex in face:
                glVertex3fv(rotated_corners[vertex])
            glEnd()
            colour_counter += 1

    def collision(self):
        collision = False
        for coordinate in stage.wall_vertices:
            xyz_length = coordinate - self.xyz
            length_player_block = np.sqrt(xyz_length[0] ** 2 + xyz_length[1] ** 2 + xyz_length[2] ** 2)
            if length_player_block < self.player_and_vertex_distance:
                collision = True
        if collision:
            self.xyz -= self.movement_vector

    def free_fall(self):
        free_fall = False
        gravity_constant = 9.81
        delta_t = 1 / current_frame_rate * 10  # Time step for the current frame

        for coordinate in stage.air_vertices:
            xyz_length = coordinate - self.xyz
            length_player_air = np.sqrt(xyz_length[0] ** 2 + xyz_length[1] ** 2 + xyz_length[2] ** 2)
            if length_player_air < self.vertex_minus_player_distance:
                free_fall = True

        if free_fall:
            # Schwerkraft
            self.movement_vector[1] -= gravity_constant * delta_t

    def respawn(self):
        if (self.xyz[1] < -300) or (self.health == 0):
            self.respawn_tick = frame_index + 120
            self.xyz = np.copy(self.spawn_xyz)
            self.movement_permission = False
            self.health = 5
        if frame_index > self.respawn_tick:
            self.movement_permission = True


class Stage:
    def __init__(self):
        self.stage = None
        self.air_vertices = None
        self.wall_vertices = None
        self.colour_vertices = None
        self.colour_vertices_info = None
        self.vertices = None
        self.points = None
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        self.define_stage()
        self.createvertices(360, 360, STAGE_TILE_SIZE)
        self.render_polygon()
        glEndList()
        self.render_colour()

    def create_points(self, size_x, size_z, step_amount_x, step_amount_z):
        x_range = np.linspace(-size_x, size_x, step_amount_x)
        z_range = np.linspace(-size_z, size_z, step_amount_z)
        x_p, z_p = np.meshgrid(x_range, z_range)
        y_p = np.zeros_like(x_p)
        points = np.stack((x_p, y_p, z_p), axis=-1)
        self.points = points.reshape((step_amount_x, step_amount_z, 3))

    def createvertices(self, size_x, size_z, step_size):
        step_amount_x = int((size_x * 2) / step_size) + 1
        step_amount_z = int((size_z * 2) / step_size) + 1
        self.create_points(size_x, size_z, step_amount_x, step_amount_z)
        self.vertices = []
        for x_i in range(step_amount_x - 1):
            for z_i in range(step_amount_z - 1):
                quad = [
                    self.points[x_i, z_i], self.points[x_i, z_i + 1],
                    self.points[x_i + 1, z_i + 1], self.points[x_i + 1, z_i]
                ]
                self.vertices.append(quad)
        self.vertices = np.array(self.vertices)
        self.create_colourvertices(size_x, size_z, step_size / 3)

    def create_colourvertices(self, size_x, size_z, step_size):
        step_amount_x = int((size_x * 2) / step_size) + 1
        step_amount_z = int((size_z * 2) / step_size) + 1
        self.create_points(size_x, size_z, step_amount_x, step_amount_z)
        self.points[:, :, 1] += 1
        self.colour_vertices = []
        self.colour_vertices_info = []
        t = 0
        for x_i in range(step_amount_x - 1):
            for z_i in range(step_amount_z - 1):
                quad = [
                    self.points[x_i, z_i], self.points[x_i, z_i + 1],
                    self.points[x_i + 1, z_i + 1], self.points[x_i + 1, z_i]]
                information = [(self.points[x_i, z_i] + self.points[x_i + 1, z_i + 1]) / 2, None]
                if self.stage[(x_i // 3) * (step_amount_x // 3) + z_i // 3] == 1:
                    self.colour_vertices.append(quad)
                    self.colour_vertices_info.append(information)
                t += 1
        self.colour_vertices = np.array(self.colour_vertices)

    def render_polygon(self):
        self.wall_vertices = []
        self.air_vertices = []
        for i, quad in enumerate(self.vertices):
            change = 5 * np.sin(np.pi / 2 * (i + i // (len(self.vertices) ** 0.5)))
            if self.stage[i] == 1:
                colour = (80 + change, 80 + change, 80 + change)
                draw_polygon(quad, colour)
            elif self.stage[i] == 2:
                middle_coordinate = (quad[0] + quad[2]) / 2
                self.wall_vertices.append(middle_coordinate)
                colour = (50 + change, 50 + change, 50 + change)
                draw_wall(quad, colour)
            elif self.stage[i] == 0:
                middle_coordinate = (quad[0] + quad[2]) / 2
                self.air_vertices.append(middle_coordinate)

    def render_colour(self):
        for i, quad in enumerate(self.colour_vertices):
            if self.colour_vertices_info[i][1] == "w":
                draw_polygon(quad, (255, 255, 255))
            elif self.colour_vertices_info[i][1] == "b":
                draw_polygon(quad, (173, 216, 230))

    def append_to_colour_vertices_info(self, index, team):
        self.colour_vertices_info[index][1] = team

    def render(self):
        glCallList(self.display_list)
        self.render_colour()

    def define_stage(self):
        self.stage = [
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2,
            2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2,
            2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2,
            2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2,
            2, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2,
            2, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2,
            2, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2,
            2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2,
            2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
            2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
            2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
            2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 2,
            2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 2,
            2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 2,
            2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
            2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
            2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
            2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
        ]


class SnowBall:
    snowballs = []

    def __init__(self, xyz, start_movement_force, mass, team):
        self.xyz = np.array(xyz)
        self.mass = mass
        self.movement_vector = np.array(start_movement_force)
        self.gravity_vector = np.array([0, -self.mass * 9.81, 0])
        self.t = 0
        self.is_destroyed = False
        self.team = team
        self.explosion_size = 48
        self.explosion = True
        self.__class__.snowballs.append(self)

    def movement(self):
        delta_t = 1 / current_frame_rate * 10  # Variabel t für Bewegung
        self.t += delta_t

        acceleration = self.gravity_vector / self.mass  # Only gravity affects the acceleration here

        self.movement_vector += acceleration * delta_t

        self.xyz += self.movement_vector * delta_t

    def render(self):
        if self.team == "w":
            draw_dot(self.xyz, (255-30, 255-30, 255-30), 20)
        else:
            draw_dot(self.xyz, (173-30, 216-30, 230-30), 20)

    def destruction_detection(self):
        if self.xyz[1] < 0:
            for coordinate in stage.air_vertices:
                xyz_length = coordinate - self.xyz
                length_player_block = np.linalg.norm(xyz_length)
                if length_player_block < STAGE_TILE_SIZE / 2:
                    self.explosion = False
                    if self.xyz[1] < -200:
                        self.is_destroyed = True
                else:
                    self.is_destroyed = True
        for coordinate in stage.wall_vertices:
            xyz_length = coordinate - self.xyz
            length_player_block = np.linalg.norm(xyz_length)
            if length_player_block < STAGE_TILE_SIZE / np.sqrt(3):
                self.is_destroyed = True
                self.explosion = False

        xyz_length = Player1.xyz - self.xyz
        length_player_block = np.linalg.norm(xyz_length)
        if length_player_block < STAGE_TILE_SIZE * np.sqrt(2) and self.team == "b":
            self.is_destroyed = True
            self.explosion = False
            Player1.health -= 1
        xyz_length = Player2.xyz - self.xyz
        length_player_block = np.linalg.norm(xyz_length)
        if length_player_block < STAGE_TILE_SIZE * np.sqrt(2) and self.team == "w":
            self.is_destroyed = True
            self.explosion = False
            Player2.health -= 1

        if np.linalg.norm(self.xyz) > 1000:
            self.is_destroyed = True

    def destruction_drawing_explosion(self):
        if self.explosion:
            for index, element in enumerate(stage.colour_vertices_info):
                xyz_length = element[0] - self.xyz
                xyz_length = np.sqrt(xyz_length[0] ** 2 + xyz_length[1] ** 2 + xyz_length[2] ** 2)
                if xyz_length < self.explosion_size:
                    stage.append_to_colour_vertices_info(index, self.team)


    def destruction(self):
        if self.is_destroyed:
            self.destruction_drawing_explosion()
            self.__class__.snowballs.remove(self)

    @classmethod
    def perform_actions(cls):
        for snowball in cls.snowballs[:]:
            snowball.movement()
            snowball.render()
            snowball.destruction_detection()
            if snowball.is_destroyed:
                snowball.destruction()


def draw_polygon(vertices, color):
    glColor3f(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    glBegin(GL_QUADS)
    for vertex in vertices:
        glVertex3fv(vertex)
    glEnd()


def draw_dot(position, color, size):
    #Punkt zeichen
    glColor3f(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    glPointSize(size)
    glBegin(GL_POINTS)
    glVertex3fv(position)
    glEnd()


def draw_wall(base_vertices, color):
    #wand mit chatgpt zeit sparen
    """
    Draws a block with the given base vertices and color.
    The block will extend 1 unit up from the base.

    Parameters:
        base_vertices (list of lists): List of 4 vertices defining the base of the block.
        color (tuple): RGB color for the block.
    """
    glColor3f(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    # Define the top face by adding 1 to the Y-coordinate of each base vertex
    top_vertices = [[v[0], v[1] + 36, v[2]] for v in base_vertices]

    # Draw the bottom face
    glBegin(GL_QUADS)
    for vertex in base_vertices:
        glVertex3fv(vertex)
    glEnd()

    # Draw the top face
    glBegin(GL_QUADS)
    for vertex in top_vertices:
        glVertex3fv(vertex)
    glEnd()

    # Draw the vertical sides
    glBegin(GL_QUADS)
    for i in range(4):  # Loop through each edge of the base
        glVertex3fv(base_vertices[i])
        glVertex3fv(base_vertices[(i + 1) % 4])  # Next vertex (wrap around with % 4)
        glVertex3fv(top_vertices[(i + 1) % 4])  # Corresponding top vertex
        glVertex3fv(top_vertices[i])
    glEnd()

def create_snowball(player):
    #Schneeball erstellen mit eine Kraft und ;asse
    if player.next_allowed_shot_frame <= frame_index:
        relative_xyz = np.array([-10, 20, 0])
        movement_direction = player.movement_vector / np.linalg.norm(player.movement_vector)
        tilted_direction = adjust_direction_up(movement_direction, 45)
        force_strength = 35
        mass = 1
        start_movement_force = tilted_direction * force_strength
        SnowBall(player.xyz + relative_xyz, start_movement_force, mass, player.team)
        player.next_allowed_shot_frame = frame_index + 15


def adjust_direction_up(vector, angle_degrees):

    angle_radians = np.radians(angle_degrees)

    # Horizontale Komponente
    horizontal_length = np.linalg.norm(vector[[0, 2]])  # x and z components

    # y Komponente
    new_y = vector[1] + horizontal_length * np.tan(angle_radians)

    # Neue Vektor
    tilted_vector = np.array([vector[0], new_y, vector[2]])

    #Normalisierung
    return tilted_vector / np.linalg.norm(tilted_vector)


def isometric_projection_conversion():
    # Orthographische Projektierung
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-400, 400, -300, 300, -1000, 1000)  # Orthographic projection

    #Rotation
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glRotatef(35.264, 1, 0, 0)  # Tilt down 35.264 degrees
    glRotatef(45, 0, 1, 0)  # Rotate 45 degrees for isometric effect
    glScalef(1, 1, 1)


def key_input():
    # Player 1 (WASD)
    if glfw.get_key(game_window, glfw.KEY_W) == glfw.PRESS and Player1.xyz[1] >= 0:  # Move forward
        Player1.speed = 5
    else:
        Player1.speed *= 0.8

    if glfw.get_key(game_window, glfw.KEY_A) == glfw.PRESS:  # Turn left
        Player1.facing_angle += 6
    if glfw.get_key(game_window, glfw.KEY_D) == glfw.PRESS:  # Turn right
        Player1.facing_angle -= 6
    if glfw.get_key(game_window, glfw.KEY_S) == glfw.PRESS:  # Schneeball
        create_snowball(Player1)

    # Player 2 (Arrow keys)
    if glfw.get_key(game_window, glfw.KEY_UP) == glfw.PRESS  and Player2.xyz[1] >= 0:  # Move forward
        Player2.speed = 5
    else:
        Player2.speed *= 0.8

    if glfw.get_key(game_window, glfw.KEY_LEFT) == glfw.PRESS:  # Turn left
        Player2.facing_angle += 6
    if glfw.get_key(game_window, glfw.KEY_RIGHT) == glfw.PRESS:  # Turn right
        Player2.facing_angle -= 6
    if glfw.get_key(game_window, glfw.KEY_DOWN) == glfw.PRESS:  #Schneeball
        create_snowball(Player2)

def show_results_in_popup():
    winner = "Draw!"
    if player1_score > player2_score:
        winner = "Player 1 Wins!"
    elif player1_score < player2_score:
        winner = "Player 2 Wins!"

    result_message = (
        f"Game Finished!\n\n"
        f"Score Player 1: {player1_score}%\n"
        f"Score Player 2: {player2_score}%\n\n"
        f"{winner}"
    )

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Game Results", result_message)


# OpenGL Setup
glEnable(GL_DEPTH_TEST)
glClearColor(0.1, 0.1, 0.1, 1)
glfw.swap_interval(0)

isometric_projection_conversion()

# Klassen generieren
stage = Stage()
Player1 = Player([-200, 0, -200], 24, 0, "w")
Player2 = Player([200, 0, 200], 24, 0, "b")
ui = Ui()

# Mainloop
while not glfw.window_should_close(game_window):
    start_time = time.time()
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    movement_vector = np.array([0, 0, 0])

    key_input()

    stage.render()
    Player1.move_cube()
    Player2.move_cube()
    SnowBall.perform_actions()
    ui.render_percentage_bar()
    ui.render_clock_circle()
    glfw.swap_buffers(game_window)

    # FPS Measurement, Regulierung und Monitorisierung (Chatgpt)
    frame_count += 1
    if time.time() - fps_timer >= 1.0:  # Every second
        current_frame_rate = frame_count
        print(f"FPS: {frame_count}")
        frame_count = 0
        fps_timer = time.time()
        game_seconds += 1

    #Framezeit
    elapsed_time = time.time() - start_time

    #Für regelmässiger Framerate
    if elapsed_time < FRAME_TIME:
        time.sleep(FRAME_TIME - elapsed_time)
    frame_index += 1

    if ui.game_finished:
        break


player1_score = int(ui.percentages[0] * 100)
player2_score = int(ui.percentages[2] * 100)
show_results_in_popup()
glfw.terminate()
