from MusicGenerator import MusicGenerator
from keras.models import load_model
import numpy as np

# 1. MusicGenerator 생성
musicGenerator = MusicGenerator()
# 2. 악보 생성
score = musicGenerator.Generate()
# 3. 악보 변환및 저장
musicGenerator.notes_to_midi('', score, 'sample')