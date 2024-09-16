import cv2
import numpy as np
import cv2.aruco as aruco

# Inicializar a captura de vídeo (substitua o índice da câmera, se necessário)
cap = cv2.VideoCapture(0)

# Definir os parâmetros da câmera (calibração ou valores aproximados)
camera_matrix = np.array([[800, 0, 320], 
                          [0, 800, 240], 
                          [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # Supondo que a câmera não tenha distorção significativa

# Carregar os dicionários de marcadores ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Função para desenhar eixos manualmente
def draw_axis(frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.05):
    axis = np.float32([[length, 0, 0], [0, length, 0], [0, 0, -length]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

    # Converter os pontos projetados para inteiros
    imgpts = np.int32(imgpts)

    # Desenhar os eixos em relação ao ponto de origem (tvec)
    corner = tuple(imgpts[0].ravel())  # Origem
    frame = cv2.line(frame, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 5)  # Eixo X em vermelho
    frame = cv2.line(frame, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)  # Eixo Y em verde
    frame = cv2.line(frame, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 5)  # Eixo Z em azul
    return frame

# Função principal para detectar os marcadores
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar os marcadores no frame
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # Se pelo menos um marcador for detectado
    if ids is not None:
        # Desenhar os marcadores detectados no frame
        aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Para cada marcador detectado
        for i in range(len(ids)):
            # Estimar a pose do marcador
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)
            
            # Desenhar os eixos no marcador detectado
            frame = draw_axis(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.1)
            
            # Exibir as informações do marcador detectado
            print(f"ID do marcador: {ids[i][0]}")
            print(f"Vetor de rotação (rvec): {rvec}")
            print(f"Vetor de translação (tvec): {tvec}")
            print("-------------------------------")
    
    # Exibir o frame com os marcadores e eixos desenhados
    cv2.imshow('Detecção de Marcadores ArUco', frame)
    
    # Sair do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
