from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D, Reshape, Permute, RepeatVector, Concatenate, Conv3D
from keras.layers.merge import _Merge

from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.initializers import RandomNormal

import os
import matplotlib.pyplot as plt
import numpy as np

import io
from music21 import midi
from music21 import note, stream, duration, tempo
from music21 import converter

from keras.models import load_model

class MusicGenerator():
    def __init__(self):
        self.z_dim = 32
        self.n_tracks = 4
        self.n_bars = 2
        self.n_steps_per_bar = 16
        self.n_pitches = 84

        self.weight_init = RandomNormal(mean=0., stddev=0.02)  #  'he_normal' #RandomNormal(mean=0., stddev=0.02)
        self._build_generator()

    def load_weights(self, filename):
        self.generator.load_weights(os.path.join('filename'))

    def Generate(self):
        chord_noise = np.random.normal[0, 1, (1, self.z_dim)]
        style_noise = np.random.normal[0, 1, (1, self.z_dim)]
        melody_noise = np.random.normal[0, 1, (1, self.z_dim)]
        groove_noise = np.random.normal[0, 1, (1, self.z_dim)]


        score = self.generator.predict([chord_noise, style_noise, melody_noise, groove_noise])
        np.argmax(score[0, 0, 0:4, :, 3], axis=1)
        score[0, 0, 0:4, 60, 3] = 0.02347812

        return score

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
                parts.write('midi', fp=os.path.join(run_folder, "samples/{}.midi".format(filename)))

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

    def _build_generator(self):

        chords_input = Input(shape=(self.z_dim,), name='chords_input')
        style_input = Input(shape=(self.z_dim,), name='style_input')
        melody_input = Input(shape=(self.n_tracks, self.z_dim), name='melody_input')
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

            bars_output[bar] = Concatenate(axis=-1)(track_output)

        generator_output = Concatenate(axis=1, name='concat_bars')(bars_output)

        self.generator = Model([chords_input, style_input, melody_input, groove_input], generator_output)






