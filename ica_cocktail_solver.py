import streamlit as st
from scipy.io.wavfile import read, write
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA



# menu
def menu():
    st.sidebar.header('Home')
    page = st.sidebar.radio("", ('Introdução',
                                'Exemplo', 'Testar', 'Contato'))


    #hide the menu
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    if page == 'Introdução':
        intro()

    if page == 'Exemplo':
        exemplo()

    if page == 'Testar':
        testar()

    if page == 'Contato':
        contato()



def intro():
    st.title('Cocktail Party Solver')

    '''
    O Cocktail Party Solver é uma aplicação destinada a resolver o problema da separação de áudio capturado por diferentes captores.
    Para deixar o app mais intutivo, foi limitado a utilização de apenas 2 fontes e 2 captores.


    Na literatura esse problema é chamado de Blind Source Separation. Para resolvermos o problema, nos utilizamos de algortimos de ICA (Independent Component Analysis)
    , mais especificamente do FastICA com estimação paralela das componentes vizando maximizar a entropia negativa do conjunto.


    Para que seja simulada a captação de 2 fontes, é requisitado a utilização de 2 arquivos wav que tenham sido gravados de maneira
    independete, uma vez que o presuposto é que as fontes sejam independentes.

    Após isso, os dados passam por uma transformação dada pela matriz de misturas, no nosso caso uma matriz 2x2, para que o sistema linear resultante
    faça a simulação da captação do áudio misturado por 2 captores separados.

    Por fim, o resultado do ICA é escalado de maneira que possamos escutá-los através do formato wav. Podem ser utilizados arquivos wav mono ou estéreo.

    Abaixo um vídeo ensinando como utilizar o app!
    '''

    st.video('https://www.youtube.com/watch?v=Sii_ZtoYMys')


def exemplo():

    st.title('Exemplo da utilização do Solver')

    '''

    Abaixo, podemos escutar 2 jornais, 1 em português e 1 em inglês.

    '''

    st.subheader('Primeiro áudio')
    st.audio('audio_samples/g1.wav')
    st.image('figure_samples/fig_audio1.png')

    st.subheader('Segundo áudio')
    st.audio('audio_samples/news.wav')
    st.image('figure_samples/fig_audio2.png')

    '''

    Agora, precisamos definir uma matriz de misturas, que vai simular a captação dos áudios de formas diferentes por
    2 captores separadamente.

    '''


    st.subheader('Matriz de Misturas')
    st.latex(r''' A = 
        \begin{bmatrix}
            0.8 & 0.3\\ 
            0.2 & 0.7
        \end{bmatrix}
            ''')



    '''

    Os resultados da aplicação dessa transformação linear é mostrado abaixo:

    '''

    st.subheader('Áudio Captado pela primeira fonte')
    st.audio('audio_samples/audio_mistura_teste1.wav')
    st.image('figure_samples/fig_mistura1.png')

    st.subheader('Áudio Captado pela segunda fonte')
    st.audio('audio_samples/audio_mistura_teste2.wav')
    st.image('figure_samples/fig_mistura2.png')

    '''

    Observamos que com essa matriz de misturas, o efeito nas 2 misturas foi deixar um som um pouco mais alto do que o outro, simulando o caso
    em que o captador de som está mais perto de uma fonte do que de outra.

    '''

    st.header('Aplicação do ICA')


    '''

    Abaixo o resultado obtido pela utilzação do FastICA (ICA com ponto fixo). O ICA em si não consegue devolver o escalonamento correto, e isso é fácil de ver
    pois qualquer multiplicação dentro da matriz de misturas pode ser compensada com uma 'divisão' nas sources,
     o que faz com que tenhamos essa dúvida com relação a escala.

    '''

    st.subheader('Resultados sem correção nas amplitudes')
    st.image('figure_samples/fig_estimacao_sem.png')



    st.header('Resultados com correção de amplitude')
    '''

    Agora, podemos corrigir as amplitudes fazendo o nosso range de valores ficar nos valores padrões de um arquivo wav.

    '''


    st.subheader('Estimativa da primeira source')
    st.audio('audio_samples/audio1.wav')

    st.subheader('Estimativa da segunda source')
    st.audio('audio_samples/audio2.wav')



def testar():

    st.title('Vamos Testar!')

    st.subheader('Upload de Arquivos!')
    '''

    Abaixo você precisa fazer o upload de 2 arquivos wav. Não precisa se preocupar com o tamanho deles
    ou se eles são mono ou estéreo. O app vai tomar o tempo do arquivo mais curto e vai analisar a mistura apenas ali,
    uma vez que não faz sentido misturar um áudio com outro já terminado.

    '''

    file1 = st.file_uploader('Escolha um arquivo wav para o áudio1', type = 'wav')

    file2 = st.file_uploader('Escolha um arquivo wav para o áudio2', type = 'wav')



    st.subheader('Definição da matriz de misturas')
    '''

    Além disso, caso você queira, você também pode modificar a matriz de misturas. A matriz utilizada para o nosso exemplo fica como standard, mas você pode
    modificá-la abaixo.

    '''


    select = st.selectbox('Escolha uma opção', ('Utilizar a matriz de misturas standard','Eu mesmo quero criar a matriz de misturas'))


    if select == 'Utilizar a matriz de misturas standard':

        A = np.array([[0.8, 0.3],[0.2,0.7]])


    if select == 'Eu mesmo quero criar a matriz de misturas':
        st.markdown('Por favor escolha os elementos da matriz. A mesma se encontra no seguinte formato:')
        st.latex(r'''
        \begin{bmatrix}
            a11 & a12\\ 
            a21 & a22 
        \end{bmatrix}
        ''')
        st.markdown('Lembrando que o range dos elementos vai de 0 até 1 e que as colunas devem ser diferentes, senão temos apenas 1 captura.')

        a11 = st.slider(
        'a11',
        0.0, 1.0)

        a12 = st.slider(
        'a12',
        0.0, 1.0)

        a21 = st.slider(
        'a21',
        0.0, 1.0)

        a22 = st.slider(
        'a22',
        0.0, 1.0)


        A = np.array([[a11, a12],[a21, a22]])


    if st.button('Vamos lá!'):

        if file1 and file2 is not None:

            fast_ica(file1, file2, A)

        else:

            st.markdown('Por favor coloque os 2 arquivos wav!')


def contato():

    st.title('Informações para contato')

    '''
    Caso haja alguma dúvida ou sugestão,meu email e linkedin.

    Email: romulopaiva@poli.ufrj.br

    Linkedin: www.linkedin.com/in/rômulo-pinto-paiva-amaral-826738111

    Caso deseje conhecer alguns de meus outros trabalhos:

    Github: RomuloPaiva01


    '''
       

def fast_ica(file1, file2, A):


    #pre-processing the files
    audio1 = read(file1)
    sample_rate = audio1[0]
    audio1 = np.array(audio1[1])

    audio2 = read(file2)
    audio2 = np.array(audio2[1])


    #checking if we have a mono or stereo type
    if isinstance(audio1[0], np.ndarray) == True: 
        audio1 = audio1[:,0]

    if isinstance(audio2[0], np.ndarray) == True:
        audio2 = audio2[:,0]


    audio1 = np.array(audio1)
    audio2 = np.array(audio2)


    #I want the audios to have the same lenght, so I will take the small len
    short = min(len(audio1), len(audio2))
    audio1 = audio1[:short]
    audio2 = audio2[:short]

    #number of components
    n_components = 2


    #creating the mixture
    s  = np.array([audio1, audio2])
    x = A @ s

    
    #if I want to see one mixture after
    mixture1 = ((x[0] + x[0].min()) * (2 ** 15) / x[0].ptp()).astype(np.int16)
    mixture2 = ((x[1] + x[1].min()) * (2 ** 15) / x[1].ptp()).astype(np.int16)

    #ica
    transformer = FastICA(n_components= n_components, max_iter= 1000, random_state=0, whiten = True, algorithm='parallel', fun = 'cube')
    s_estimation = transformer.fit_transform(x.T)


    #correct the scale
    s1 = ((s_estimation[:, 0] + s_estimation[:, 0].min()) * (2 ** 15) / s_estimation[:, 0].ptp()).astype(np.int16)
    s2 = ((s_estimation[:, 1] + s_estimation[:, 1].min()) * (2 ** 15) / s_estimation[:, 1].ptp()).astype(np.int16)

    #writing the wav files
    write('audiofinal1.wav', sample_rate, s1)
    write('audiofinal2.wav', sample_rate, s2)
    write('audio_mistura1.wav', sample_rate, mixture1)
    write('audio_mistura2.wav', sample_rate, mixture2)

    #writing the outputs
    def creating_plot(data, title):

        fig = plt.figure(figsize=(10, 7))
        plt.title(title)
        plt.xlabel('Tempo x Sample Rate')
        plt.ylabel('Amplitude')
        plt.plot(data)
        st.pyplot()


    st.markdown('Primeira Captação')
    creating_plot(mixture1, 'Mistura 1')
    st.audio('audio_mistura1.wav')

    st.markdown('Segunda Captação')
    creating_plot(mixture2, 'Mistura 2')
    st.audio('audio_mistura2.wav')

    st.markdown('Estimativa da primeira source')
    creating_plot(s1, 'Estimativa da primeira source')
    st.audio('audiofinal1.wav')

    st.markdown('Estimativa da segunda source')
    creating_plot(s2, 'Estimativa da segunda source')
    st.audio('audiofinal2.wav')


menu()