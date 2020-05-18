from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D, Reshape, Permute, RepeatVector, Concatenate, Conv3D
from keras.layers.merge import _Merge
from keras.initializers import RandomNormal

from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam, RMSprop
import os
import numpy as np
from music21 import *
from music21.converter.subConverters import ConverterMusicXML
import io
from music21 import midi
from music21 import note, stream, duration, tempo
from music21 import converter
from pypianoroll import Multitrack, Track

# MusicGenerator 클래스 호출시 generator.h5파일이 같은 폴더안에 있어어ㅑ함

class MusicGenerator():
    def __init__(self):
        self.z_dim = 32
        self.n_tracks = 1
        self.n_bars = 4
        self.n_steps_per_bar = 96
        self.n_pitches = 84

        self.weight_init = RandomNormal(mean=0., stddev=0.02)  #  'he_normal' #RandomNormal(mean=0., stddev=0.02)
        self._build_generator()
        self.generator.load_weights('weights-g.h5')

    def conv_t(self, x, f, k, s, a, p, bn):
        x = Conv2DTranspose(
            filters=f
            , kernel_size=k
            , padding=p
            , strides=s
            , kernel_initializer=self.weight_init
        )(x)

        if bn:
            x = BatchNormalization(momentum=0.9)(x)

        if a == 'relu':
            x = Activation(a)(x)
        elif a == 'lrelu':
            x = LeakyReLU()(x)

        return x

    def TemporalNetwork(self):

        input_layer = Input(shape=(self.z_dim,), name='temporal_input')

        x = Reshape([1, 1, self.z_dim])(input_layer)
        x = self.conv_t(x, f=1024, k=(2, 1), s=(1, 1), a='relu', p='valid', bn=True)
        x = self.conv_t(x, f=self.z_dim, k=(self.n_bars - 1, 1), s=(1, 1), a='relu', p='valid', bn=True)

        output_layer = Reshape([self.n_bars, self.z_dim])(x)

        return Model(input_layer, output_layer)

    def BarGenerator(self):

        input_layer = Input(shape=(self.z_dim * 4,), name='bar_generator_input')

        x = Dense(1024)(input_layer)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = Reshape([2, 1, 512])(x)
        x = self.conv_t(x, f=512, k=(2, 1), s=(2, 1), a='relu', p='same', bn=True) # (4, 1, 512)
        x = self.conv_t(x, f=256, k=(2, 1), s=(2, 1), a='relu', p='same', bn=True) # (8, 1, 256)
        x = self.conv_t(x, f=256, k=(2, 1), s=(2, 1), a='relu', p='same', bn=True) # (16, 1, 256)
        x = self.conv_t(x, f=256, k=(6, 1), s=(6, 1), a='relu', p='same', bn=True) # (96, 1, 256)
        x = self.conv_t(x, f=256, k=(1, 7), s=(1, 7), a='relu', p='same', bn=True) # (96, 7, 256)
        x = self.conv_t(x, f=1, k=(1, 12), s=(1, 12), a='tanh', p='same', bn=False) # (96, 84, 1)

        output_layer = Reshape([1, self.n_steps_per_bar, self.n_pitches, 1])(x)

        return Model(input_layer, output_layer)

    def _build_generator(self):

        chords_input = Input(shape=(self.z_dim,), name='chords_input')
        style_input = Input(shape=(self.z_dim,), name='style_input')
        melody_input = Input(shape=(self.n_tracks, self.z_dim), name='melody_input') # 우리 모델에서 self.n_tracks는 항상 1
        groove_input = Input(shape=(self.n_tracks, self.z_dim), name='groove_input')

        # CHORDS -> TEMPORAL NETWORK
        self.chords_tempNetwork = self.TemporalNetwork()
        self.chords_tempNetwork.name = 'temporal_network'
        chords_over_time = self.chords_tempNetwork(chords_input)  # [n_bars, z_dim]

        # MELODY -> TEMPORAL NETWORK
        melody_over_time = [None] * self.n_tracks  # list of n_tracks [n_bars, z_dim] tensors
        self.melody_tempNetwork = [None] * self.n_tracks
        for track in range(self.n_tracks):
            self.melody_tempNetwork[track] = self.TemporalNetwork()
            melody_track = Lambda(lambda x: x[:, track, :])(melody_input)
            melody_over_time[track] = self.melody_tempNetwork[track](melody_track)

        # CREATE BAR GENERATOR FOR EACH TRACK
        self.barGen = [None] * self.n_tracks
        for track in range(self.n_tracks):
            self.barGen[track] = self.BarGenerator()

        # CREATE OUTPUT FOR EVERY TRACK AND BAR
        bars_output = [None] * self.n_bars
        for bar in range(self.n_bars):
            track_output = [None] * self.n_tracks

            c = Lambda(lambda x: x[:, bar, :], name='chords_input_bar_' + str(bar))(chords_over_time)  # [z_dim]
            s = style_input  # [z_dim]

            for track in range(self.n_tracks):
                m = Lambda(lambda x: x[:, bar, :])(melody_over_time[track])  # [z_dim]
                g = Lambda(lambda x: x[:, track, :])(groove_input)  # [z_dim]

                z_input = Concatenate(axis=1, name='total_input_bar_{}_track_{}'.format(bar, track))([c, s, m, g])

                track_output[track] = self.barGen[track](z_input)

            bars_output[bar] = track_output[0]

        generator_output = Concatenate(axis=1, name='concat_bars')(bars_output)

        self.generator = Model([chords_input, style_input, melody_input, groove_input], generator_output)

    def Generate(self):
        n = 1
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
        mf.open('samples/Hanon1.mid', attrib='rb')
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




