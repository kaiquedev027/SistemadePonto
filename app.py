import os
import cv2
import numpy as np
import uuid
import sqlite3
from flask import Flask, render_template, request, redirect, url_for,send_file
from datetime import datetime
from fpdf import FPDF

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUNCIONARIOS_DIR = os.path.join(BASE_DIR, "funcionarios")

if not os.path.exists(FUNCIONARIOS_DIR):
    os.makedirs(FUNCIONARIOS_DIR)

def gerar_relatorio_todos(data=None):
    conn = sqlite3.connect('ponto.db')
    cursor = conn.cursor()
    
    if data:  # Se uma data for fornecida, filtre os registros por essa data
        cursor.execute('''
        SELECT nome, horario FROM registros
        WHERE date(horario) = ?
        ORDER BY horario DESC
        ''', (data,))
    else:  # Caso contrário, obtenha todos os registros
        cursor.execute('''
        SELECT nome, horario FROM registros
        ORDER BY horario DESC
        ''')

    registros = cursor.fetchall()
    conn.close()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Adiciona título
    pdf.set_font("Arial", 'B', 16)
    if data:
        pdf.cell(0, 10, f"Relatório de Pontos - Data: {data}", 0, 1, 'C')
    else:
        pdf.cell(0, 10, "Relatório de Pontos de Todos os Funcionários", 0, 1, 'C')

    # Adiciona cabeçalho da tabela
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Nome", 1)
    pdf.cell(60, 10, "Data e Hora", 1)
    pdf.ln()

    # Adiciona linhas de histórico
    pdf.set_font("Arial", '', 12)
    for registro in registros:
        pdf.cell(40, 10, registro[0], 1)
        pdf.cell(60, 10, registro[1], 1)
        pdf.ln()

    # Salva o PDF
    pdf_file = f"relatorio_pontos_{data if data else 'todos_funcionarios'}.pdf"
    pdf.output(pdf_file)

    return pdf_file


# Função para inicializar o banco de dados SQLite
def init_db():
    conn = sqlite3.connect('ponto.db')
    cursor = conn.cursor()
    
    # Criar tabela para registro de ponto
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS registros (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT NOT NULL,
        horario TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

# Função para inserir um registro de ponto no banco de dados
def registrar_ponto(nome, horario):
    conn = sqlite3.connect('ponto.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO registros (nome, horario)
    VALUES (?, ?)
    ''', (nome, horario))
    
    conn.commit()
    conn.close()

def obter_historico(nome):
    conn = sqlite3.connect('ponto.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT nome, horario FROM registros
    WHERE nome = ?
    ORDER BY horario DESC
    ''', (nome,))
    
    registros = cursor.fetchall()
    conn.close()
    return registros

# Inicializar o banco de dados
init_db()

# Carregar o classificador de rosto pré-treinado do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Função para treinar o modelo LBPH
def train_model():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    images, labels, label_names = [], [], []
    current_label = 0

    # Treinar o modelo com todas as imagens de todos os funcionários
    for nome_funcionario in os.listdir(FUNCIONARIOS_DIR):
        pasta_funcionario = os.path.join(FUNCIONARIOS_DIR, nome_funcionario)
        if not os.path.isdir(pasta_funcionario):
            continue
        
        label_names.append(nome_funcionario)
        for filename in os.listdir(pasta_funcionario):
            img_path = os.path.join(pasta_funcionario, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(current_label)
        
        current_label += 1
    
    if len(images) == 0:
        return None, None

    face_recognizer.train(images, np.array(labels))
    return face_recognizer, label_names

# Função para autenticar o funcionário ao bater o ponto
def authenticate_employee(face_recognizer, label_names):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return "Erro ao abrir a câmera."

    CONFIDENCE_THRESHOLD = 50  # Ajuste a confiança conforme necessário

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar o quadro da câmera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            id_, confidence = face_recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < CONFIDENCE_THRESHOLD:  # Menor valor indica maior confiança
                nome_funcionario = label_names[id_]
                cap.release()
                cv2.destroyAllWindows()
                horario = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
                registrar_ponto(nome_funcionario, horario)  # Registra no banco de dados
                return nome_funcionario, horario

        cv2.imshow("Autenticando", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None, None

# Página principal com botões
@app.route('/')
def index():
    return render_template('index.html')

# Rota para cadastrar funcionário
@app.route('/cadastrar', methods=['GET', 'POST'])
def cadastrar():
    if request.method == 'POST':
        nome = request.form['nome']
        pasta_funcionario = os.path.join(FUNCIONARIOS_DIR, nome)
        if not os.path.exists(pasta_funcionario):
            os.makedirs(pasta_funcionario)
        
        # Abrir câmera e capturar 50 fotos em cores
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Erro ao abrir a câmera.")
            return "Erro ao abrir a câmera."

        count = 0
        while count < 50:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar o quadro da câmera.")
                break
            
            gray_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray_face, 1.05, 3, minSize=(30, 30))

            for (x, y, w, h) in detected_faces:
                filename = f"{uuid.uuid4()}.jpg"
                cv2.imwrite(os.path.join(pasta_funcionario, filename), gray_face[y:y+h, x:x+w])  # Salva as imagens em escala de cinza
                count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Capturando Rostos", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return redirect(url_for('index'))

    return render_template('cadastrar.html')

# Rota para bater ponto
@app.route('/bater_ponto', methods=['GET'])
def bater_ponto():
    face_recognizer, label_names = train_model()

    if face_recognizer is None or label_names is None:
        return "Nenhuma imagem foi encontrada para treinar o modelo. Por favor, cadastre funcionários primeiro."

    nome_funcionario, horario = authenticate_employee(face_recognizer, label_names)

    if nome_funcionario is not None:
        historico = obter_historico(nome_funcionario)  # Obter histórico de pontos
        return render_template('sucesso.html', nome=nome_funcionario, horario=horario, historico=historico)
    else:
        return "Falha na autenticação. Tente novamente."


@app.route('/historico', methods=['GET'])
def historicos():
    data = request.args.get('data')  # Obtém a data do filtro

    conn = sqlite3.connect('ponto.db')
    cursor = conn.cursor()

    if data:  # Se uma data foi fornecida, filtrar os registros por essa data
        cursor.execute('''
        SELECT nome, horario FROM registros
        WHERE date(horario) = ?
        ORDER BY horario DESC
        ''', (data,))
    else:  # Caso contrário, buscar todos os registros
        cursor.execute('''
        SELECT nome, horario FROM registros
        ORDER BY horario DESC
        ''')

    registros = cursor.fetchall()
    conn.close()

    return render_template('historico.html', registros=registros)


@app.route('/gerar_relatorio_todos', methods=['GET'])
def gerar_relatorio_todos_funcionarios():
    data = request.args.get('data')  # Obtém a data da query string (se fornecida)
    pdf_file = gerar_relatorio_todos(data)
    return send_file(pdf_file, as_attachment=True)


@app.route('/login')
def login():
    return render_template('login.html')


if __name__ == '__main__':
    app.run(debug=True)
