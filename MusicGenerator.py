import os
import numpy as np
from music21 import *
from music21.converter.subConverters import ConverterMusicXML
import io
from music21 import midi
from music21 import note, stream, duration, tempo
from music21 import converter
from pypianoroll import Multitrack, Track
from keras.models import load_model

# MusicGenerator 클래스 호출시 generator.h5파일이 같은 폴더안에 있어어ㅑ함

class MusicGenerator():
    def __init__(self):
        self.z_dim = 32
        self.n_tracks = 1
        self.n_bars = 4
        self.n_steps_per_bar = 16
        self.n_pitches = 84
        self.generator = load_model('generator.h5', compile=False)

    def Generate(self):
        n = 2
        chords_noise = np.random.normal(0, 1, (n, self.z_dim))
        style_noise = np.random.normal(0, 1, (n, self.z_dim))
        melody_noise = np.random.normal(0, 1, (n, self.n_tracks, self.z_dim))
        groove_noise = np.random.normal(0, 1, (n, self.n_tracks, self.z_dim))


        score = self.generator.predict([chords_noise, style_noise, melody_noise, groove_noise])

        return score

    # run_foler = 저장 위치(현재 폴더에 저장시 ''로 입력)
    # score = 변환전 악보
    # filname
    def notes_to_midi(self, run_folder, output, filename):
        score = self.TrimScore(output, 16)

        # 배치를 미디로 변환
        binarized = score > -0.5  # 1. 이진화
        # 1. reshpae (마디 부분합치기)
        score = binarized.reshape(-1, binarized.shape[1] * binarized.shape[2], binarized.shape[3],
                                  binarized.shape[4])  # 2. 마디 합치기
        # 2. pad설정
        pad_width = ((0, 0), (0, 0), (24, 20), (0, 0))
        # 3. pad적용
        score = np.pad(score, pad_width, 'constant')  # 3. 패드 적용
        # 4. reshape(-1, pitches, Track)
        score = score.reshape(-1, score.shape[2], score.shape[3])  # 4. reshape ( -1, 피치 수, 트랙)
        # 5. multitrack 생성
        multitrack = Multitrack(beat_resolution=24, tempo=120)  # 4.4. Multitrack 생성
        track = Track(score[..., 0])
        multitrack.append_track(track)

        if run_folder != None:
            multitrack.write('{}/{}.mid'.format(run_folder, filename))
        else:
            multitrack.write('{}.mid'.format('sample'))


    # notes_to_png
    # run_folder : 파일 경로
    # score : MusicGenerator의 Generate함수 출력 값
    # filename : 파일 이름

    # 필수 사전처리 항목 : lilypond  C:\에 설치 되어있어야함

    def notes_to_png(self, run_folder, output, filename):
        environment.set("lilypondPath", r"C:\LilyPond\usr\bin\lilypond.exe")
        # 1. midi 생성
        self.notes_to_midi(run_folder, output, filename)
        #self.notes_to_midi(run_folder, score, filename)
        # 2. midi 를 stream형태의 score로
        #score = converter.parse(os.path.join(run_folder, "{}.mid".format(filename)))
        mf= midi.MidiFile()
        mf.open('samples/sample.mid', attrib='rb')
        mf.read()
        mf.close()
        print(len(mf.tracks))

        streamScore = midi.translate.midiFileToStream(mf)
        # 3. score를 lilypond를 통해 이미지로 변경
        streamScore.write('lily.png', fp=os.path.join(run_folder, "{}.png".format(filename)))
        mf.close()

    def TrimScore(self, score, leastNoteBeat):
        # score: (batchSize, 4, 96, 84 1) 형태의 배열
        # leastNoteBeat : 악보에서 나올 수 있는 음표의 최소박자
        output = np.array(score)

        batchSize = score.shape[0]
        count = score.shape[2] // leastNoteBeat;
        barCount = score.shape[1]
        pitchCount = score.shape[3]
        trackCount = score.shape[4]

        for dataNumber in range(batchSize):
            for trackNumber in range(trackCount):
                for barNumber in range(barCount):
                    for i in range(leastNoteBeat):
                        for pitchNumber in range(pitchCount):
                            output[dataNumber, barNumber, i * count:(i + 1) * count, pitchNumber, trackNumber] = score[
                                dataNumber, barNumber, i * count, pitchNumber, trackNumber]

        return output




