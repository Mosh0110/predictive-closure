# Predictive Closure

This repository contains the code for the paper *Predictive Closure: A Set-Level Characterization of JEPA's Feature Selection*.

The results in this repository show that what a JEPA keeps in its representation depends on how features relate to each other, not only on how predictable each one is.
A feature whose next value is random can still be represented when its current value helps predict other features, and changing how one feature relates to another can change how well other features are represented.
Steering one feature, with an auxiliary loss or by giving it to the predictor, can also strengthen another feature whose next values it helps predict.

## Experiments

Each experiment trains small JEPA systems on sequences of CIFAR-10 image collages, with a representation too small to hold every feature.
How well a feature is represented is measured by how accurately a probe can read it from the trained encoder.

- **Cross-subsidization** (`traffic`): Each frame shows two images and a traffic light. The light's color is random at every step, so it can never be predicted, but it decides which of the two images is replaced in the next frame. The more often the light decides this, the better the light is represented.
- **Set switching** (`predictability`): Each frame shows four images in two pairs, A and B, C and D. Within each pair, an image is sometimes copied to the other position in the next frame. Making the copy from C to D more likely shifts the representation from A and B to C and D, although nothing about A and B changes.
- **Feasibility dependence** (`subset`): The pair A and B competes with a cycle of images that are copied into one another. As the cycle grows from 2 to 16 images, with the same copying probabilities, it becomes less represented, until A and B are the better-represented images.
- **Auxiliary reconstruction** (`reconstruction`): In the four-image setup, an extra loss asks the system to reconstruct image C. This can also strengthen the representation of D, whose next value C helps predict but which is never reconstructed, and weaken that of A and B.
- **Predictor side information** (`side_input`): In the four-image setup, image C is given directly to the predictor. This can strengthen the representation of D and change how well the other images are represented.

## Running

```sh
bash reproduce.sh
```

This trains all the systems and saves the paper's figures in `results/figures/`.

## License

[MIT](LICENSE).
