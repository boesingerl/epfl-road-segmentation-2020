from imgaug import augmenters as iaa
import numpy as np

def apply_augments(original):
  # desired rotations
  rotations = [90,180,270]

  # apply rotations
  rotaters = [iaa.Sequential([iaa.Affine(rotate=r)]) for r in rotations]
  rotated = [seq.augment_images(original) for seq in rotaters]
  rotated.append(original)

  # apply flips
  flippers = [iaa.Sequential([iaa.flip.Fliplr(1.0)]), iaa.Sequential([iaa.flip.Flipud(1.0)])]
  flipped = [seq.augment_images(x) for seq in flippers for x in rotated]

  flipped.extend(rotated)
  return np.concatenate(flipped)