from MusicGenerator import MusicGenerator

musicGenerator = MusicGenerator()
musicGenerator.load_weights('weights/weights-g.h5')
score = musicGenerator.Generate()
musicGenerator.notes_to_midi('', score, 'sample')