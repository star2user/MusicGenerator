from MusicGenerator import MusicGenerator
from keras.models import load_model
import numpy as np

critic = load_model('critic.h5', compile=False)
generator = load_model('generator.h5', compile=False)
z_dim = 32
n_tracks = 4


chords_noise = np.random.normal(0, 1, (1, z_dim))
style_noise = np.random.normal(0, 1, (1, z_dim))
melody_noise = np.random.normal(0, 1, (1, n_tracks, z_dim))
groove_noise = np.random.normal(0, 1, (1, n_tracks, z_dim))


score = generator.predict([chords_noise, style_noise, melody_noise, groove_noise])
np.argmax(score[0, 0, 0:4, :, 3], axis=1)
score[0, 0, 0:4, 60, 3] = 0.02347812