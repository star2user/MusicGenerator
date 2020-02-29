import os
import numpy as np
from music21 import *
from music21.converter.subConverters import ConverterMusicXML
import io
from music21 import midi
from music21 import note, stream, duration, tempo
from music21 import converter

from keras.models import load_model

# MusicGenerator 클래스 호출시 generator.h5파일이 같은 폴더안에 있어어ㅑ함

class MusicGenerator():
    def __init__(self):
        self.z_dim = 32
        self.n_tracks = 4
        self.n_bars = 2
        self.n_steps_per_bar = 16
        self.n_pitches = 84
        self.generator = load_model('generator.h5', compile=False)

    def Generate(self):
        chords_noise = np.random.normal(0, 1, (1, self.z_dim))
        style_noise = np.random.normal(0, 1, (1, self.z_dim))
        melody_noise = np.random.normal(0, 1, (1, self.n_tracks, self.z_dim))
        groove_noise = np.random.normal(0, 1, (1, self.n_tracks, self.z_dim))


        score = self.generator.predict([chords_noise, style_noise, melody_noise, groove_noise])
        np.argmax(score[0, 0, 0:4, :, 3], axis=1)
        score[0, 0, 0:4, 60, 3] = 0.02347812

        return score

    # run_foler = 저장 위치(현재 폴더에 저장시 ''로 입력)
    # score = 변환전 악보
    # filname
    def notes_to_midi(self, run_folder, score, filename):

        for score_num in range(len(score)):

            max_pitches = np.argmax(score, axis = 3)

            midi_note_score = max_pitches[score_num].reshape([self.n_bars * self.n_steps_per_bar, self.n_tracks])
            parts = stream.Score()
            parts.append(tempo.MetronomeMark(number=66))

            for i in range(self.n_tracks):
                last_x = int(midi_note_score[:, i][0])
                s = stream.Part()
                dur = 0

                for idx, x in enumerate(midi_note_score[:, i]):
                    x = int(x)

                    if (x != last_x or idx % 4 == 0) and idx > 0:
                        n = note.Note(last_x)
                        n.duration = duration.Duration(dur)
                        s.append(n)
                        dur = 0

                    last_x = x
                    dur = dur + 0.25

                n = note.Note(last_x)
                n.duration = duration.Duration(dur)
                s.append(n)

                parts.append(s)
                parts.write('midi', fp=os.path.join(run_folder, "{}.midi".format(filename)))


    # notes_to_png
    # run_folder : 파일 경로
    # score : MusicGenerator의 Generate함수 출력 값
    # filename : 파일 이름

    # 필수 사전처리 항목 : lilypond  C:\에 설치 되어있어야함

    def notes_to_png(self, run_folder, score, filename):
        environment.set("lilypondPath", r"C:\LilyPond\usr\bin\lilypond.exe")

        for score_num in range(len(score)):

            max_pitches = np.argmax(score, axis = 3)

            midi_note_score = max_pitches[score_num].reshape([self.n_bars * self.n_steps_per_bar, self.n_tracks])
            parts = stream.Score()
            parts.append(tempo.MetronomeMark(number=66))

            for i in range(self.n_tracks):
                last_x = int(midi_note_score[:, i][0])
                s = stream.Part()
                dur = 0

                for idx, x in enumerate(midi_note_score[:, i]):
                    x = int(x)

                    if (x != last_x or idx % 4 == 0) and idx > 0:
                        n = note.Note(last_x)
                        n.duration = duration.Duration(dur)
                        s.append(n)
                        dur = 0

                    last_x = x
                    dur = dur + 0.25

                n = note.Note(last_x)
                n.duration = duration.Duration(dur)
                s.append(n)

                parts.append(s)
                parts.write('lily.png', fp=os.path.join(run_folder, "{}.png".format(filename)))




