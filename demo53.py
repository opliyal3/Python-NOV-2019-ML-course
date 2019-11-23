import keras.utils as utils

origs = [4, 7, 9, 0]
NUM_CATEGORY = 46

for orig in origs:
    converted = utils.to_categorical(orig, NUM_CATEGORY)
    print(f'after conversion, {orig} will become {converted}')
