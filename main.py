from MusicGenerator import MusicGenerator
from keras.models import load_model
import numpy as np

# 1. MusicGenerator 생성, generator.h5파일이 같은 폴더안에 있어야함
musicGenerator = MusicGenerator()
# 2. 악보 생성
score = musicGenerator.Generate()
# 3. 악보 변환및 저장( 첫번째 매개변수는 저장됄 파일 위치, 두번째 매개변수는 악보, 세번째 매개변수는 파일 이름
musicGenerator.notes_to_midi('', score, 'sample')