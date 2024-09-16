import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2
import cv2.aruco as aruco

def init_gl():
    glEnable(GL_DEPTH_TEST)  # Habilita o Z-buffer
    glDepthFunc(GL_LESS)     # Define a função de teste de profundidade
    glClearColor(0, 0, 0, 1) # Define a cor de fundo (preto)

# Inicializar o Pygame e criar uma janela OpenGL
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)
init_gl()

# Configurações da câmera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, display[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display[1])
camera_matrix = np.array([[800, 0, display[0] // 2], [0, 800, display[1] // 2], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

def draw_cube():
    glBegin(GL_QUADS)
    glColor3f(1, 0, 0)
    glVertex3f(0.5, 0.5, -1)
    glVertex3f(-0.5, 0.5, -1)
    glVertex3f(-0.5, -0.5, -1)
    glVertex3f(0.5, -0.5, -1)
    glEnd()

def draw_triangle():
    glBegin(GL_TRIANGLES)
    glColor3f(0, 1, 0)  # Cor verde para o triângulo
    glVertex3f(0, 0.5, -1)  # Ponto superior
    glVertex3f(-0.5, -0.5, -1) # Ponto inferior esquerdo
    glVertex3f(0.5, -0.5, -1) # Ponto inferior direito
    glEnd()

def draw_axes(frame, rvec, tvec):
    # Define os pontos de eixo (em metros)
    axis_points = np.float32([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]).reshape(-1, 3)

    # Projeta os pontos 3D para 2D
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

    # Converte os pontos projetados para a imagem
    imgpts = np.int32(imgpts).reshape(-1, 2)  # Converte para inteiros

    # Desenha os eixos
    if len(imgpts) >= 4:
        frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)  # Eixo X em vermelho
        frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3)  # Eixo Y em verde
        frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3)  # Eixo Z em azul

    return frame

def render_scene():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Captura o frame da câmera
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Exibir o frame da câmera na janela OpenGL
    frame = cv2.flip(frame, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_data = frame.tobytes()  # Atualizado para tobytes()

    # Ajusta a visualização para que o vídeo ocupe toda a tela
    glRasterPos2f(-1, -1)
    glDrawPixels(display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE, frame_data)

    if ids is not None:
        for i in range(len(ids)):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)

            # Desenhar os eixos manualmente
            frame = draw_axes(frame, rvec[0][0], tvec[0][0])

            aruco.drawDetectedMarkers(frame, corners)

            # Atualizar o frame da câmera com eixos desenhados (opcional)
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_data = frame.tobytes()

            # Desenha o frame com eixos desenhados
            glRasterPos2f(-1, -1)
            glDrawPixels(display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE, frame_data)

            # Desenho dos objetos 3D
            glPushMatrix()
            glTranslatef(tvec[0][0][0], tvec[0][0][1], tvec[0][0][2] - 2)  # Ajustar a posição Z para frente da câmera
            glRotatef(rvec[0][0][0] * 180 / np.pi, 1, 0, 0)
            glRotatef(rvec[0][0][1] * 180 / np.pi, 0, 1, 0)
            glRotatef(rvec[0][0][2] * 180 / np.pi, 0, 0, 1)

            if ids[i][0] == 0:
                draw_cube()
            elif ids[i][0] == 1:
                draw_triangle()

            glPopMatrix()

    pygame.display.flip()

# Loop principal
clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            quit()

    render_scene()
    clock.tick(30)  # Limita a 30 FPS
