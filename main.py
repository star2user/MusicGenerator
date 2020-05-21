from MusicGenerator import MusicGenerator
from music21 import *
from music21.converter.subConverters import ConverterLilypond

#1. MusicGenerator 생성, weights-g.h5파일이 같은 폴더안에 있어야함
musicGenerator = MusicGenerator()
# 2. 악보 생성
score = musicGenerator.Generate()
# 3. 악보 변환및 저장( 첫번째 매개변수는 저장됄 파일 위치, 두번째 매개변수는 악보, 세번째 매개변수는 파일 이름
# samples 폴더 있어야 합니다.

run_folder = 'samples'
musicGenerator.notes_to_midi(run_folder, score, 'classicSample')
musicGenerator.notes_to_png(run_folder, score, 'classicSample')



