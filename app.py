import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2
import cv2.aruco as aruco

# Função para inicializar as configurações do OpenGL
def init_gl():
    glEnable(GL_DEPTH_TEST)  # Habilita o Z-buffer para controlar a profundidade dos objetos
    glDepthFunc(GL_LESS)     # Define a função de teste de profundidade para descartar pixels mais distantes
    glClearColor(0, 0, 0, 1) # Define a cor de fundo como preto

# Inicializar o Pygame e criar uma janela OpenGL
pygame.init()
display = (800, 600)  # Define a resolução da janela
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)  # Cria uma janela dupla para renderização com OpenGL
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)  # Define a perspectiva da câmera
glTranslatef(0.0, 0.0, -5)  # Move a cena para trás no eixo Z
init_gl()  # Inicializa as configurações do OpenGL

# Configurações da câmera
cap = cv2.VideoCapture(0)  # Captura de vídeo da webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, display[0])  # Define a largura do vídeo
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display[1])  # Define a altura do vídeo
camera_matrix = np.array([[800, 0, display[0] // 2], [0, 800, display[1] // 2], [0, 0, 1]], dtype=np.float32)  # Matriz intrínseca da câmera
dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # Coeficientes de distorção (nenhuma distorção assumida)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # Dicionário ArUco com marcadores 4x4 (50 tipos)
parameters = aruco.DetectorParameters()  # Parâmetros do detector de marcadores

# Função para desenhar um cubo simples em OpenGL
def draw_cube():
    glBegin(GL_QUADS)  # Inicia o desenho de quadrados
    glColor3f(1, 0, 0)  # Define a cor do cubo (vermelho)
    glVertex3f(0.5, 0.5, -1)  # Vértices do cubo
    glVertex3f(-0.5, 0.5, -1)
    glVertex3f(-0.5, -0.5, -1)
    glVertex3f(0.5, -0.5, -1)
    glEnd()

# Função para desenhar um triângulo simples em OpenGL
def draw_triangle():
    glBegin(GL_TRIANGLES)  # Inicia o desenho de triângulos
    glColor3f(0, 1, 0)  # Define a cor do triângulo (verde)
    glVertex3f(0, 0.5, -1)  # Vértice superior do triângulo
    glVertex3f(-0.5, -0.5, -1)  # Vértice inferior esquerdo
    glVertex3f(0.5, -0.5, -1)  # Vértice inferior direito
    glEnd()

# Função para desenhar eixos X, Y, Z na imagem do vídeo
def draw_axes(frame, rvec, tvec):
    # Define os pontos de eixo (em metros) para X, Y e Z
    axis_points = np.float32([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]).reshape(-1, 3)

    # Projeta os pontos 3D no plano 2D da imagem
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

    # Converte os pontos projetados para inteiros para desenhar na imagem
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Desenha os eixos na imagem
    if len(imgpts) >= 4:
        frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)  # Eixo X em vermelho
        frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3)  # Eixo Y em verde
        frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3)  # Eixo Z em azul

    return frame

# Função principal de renderização da cena
def render_scene():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Limpa o buffer de cor e profundidade
    glLoadIdentity()  # Reseta a matriz de transformação

    # Captura um frame da câmera
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converte a imagem para tons de cinza
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)  # Detecta os marcadores ArUco

    # Exibe o frame da câmera na janela OpenGL
    frame = cv2.flip(frame, 0)  # Inverte o frame verticalmente
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converte de BGR (OpenCV) para RGB (OpenGL)
    frame_data = frame.tobytes()  # Converte a imagem para bytes

    # Ajusta a visualização para que o vídeo ocupe toda a tela
    glRasterPos2f(-1, -1)  # Define a posição do canto inferior esquerdo
    glDrawPixels(display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE, frame_data)  # Desenha o vídeo na tela

    if ids is not None:
        for i in range(len(ids)):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)  # Estima a pose do marcador

            # Desenha os eixos na imagem do vídeo
            frame = draw_axes(frame, rvec[0][0], tvec[0][0])

            # Atualiza o frame da câmera com os eixos desenhados (opcional)
            frame = cv2.flip(frame, 0)  # Inverte o frame de volta
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converte novamente para RGB
            frame_data = frame.tobytes()

            # Exibe o frame atualizado com eixos desenhados
            glRasterPos2f(-1, -1)
            glDrawPixels(display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE, frame_data)

            # Desenha os objetos 3D (cubo ou triângulo) com base no ID do marcador ArUco detectado
            glPushMatrix()
            glTranslatef(tvec[0][0][0], tvec[0][0][1], tvec[0][0][2] - 2)  # Ajusta a posição do objeto
            glRotatef(rvec[0][0][0] * 180 / np.pi, 1, 0, 0)  # Rotaciona em torno do eixo X
            glRotatef(rvec[0][0][1] * 180 / np.pi, 0, 1, 0)  # Rotaciona em torno do eixo Y
            glRotatef(rvec[0][0][2] * 180 / np.pi, 0, 0, 1)  # Rotaciona em torno do eixo Z

            # Escolhe o objeto 3D para desenhar com base no ID do marcador
            if ids[i][0] == 0:
                draw_cube()  # Desenha o cubo se o ID for 0
            elif ids[i][0] == 1:
                draw_triangle()  # Desenha o triângulo se o ID for 1

            glPopMatrix()  # Reseta a matriz para evitar acúmulo de transformações

    pygame.display.flip()  # Atualiza a janela com as novas renderizações

# Loop principal do programa
clock = pygame.time.Clock()  # Controle de tempo para limitar FPS
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Se o usuário fechar a janela
            pygame.quit()  # Fecha o Pygame
            cap.release()  # Libera a captura de vídeo
            cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV
            quit()  # Encerra o programa

    render_scene()  # Render
