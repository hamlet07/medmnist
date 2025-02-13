# Model Card 

## Model Description

**Input:** Images in size 32x32 with 3 channels (original images need to be resized; grayscale images require 2 addtional channels)

**Output:** Classification of images with defined number of classes

**Model Architecture:** Model's architecture takes advantage of VGG16; Layers included in the top (dense, dropout) are optimised.

## Performance

Details in README.md (Section: Analysis)
Models trained and evaluated on Apple M1 Pro 16 GB RAM.
Time of preparation depends strongly on size of datasets.

## Limitations

Solution is not prepared to work with CUDA (requires additional adjustemnt of code)

## Trade-offs

Noted trade-off between model preparation and required time for optimisation (reduced number of trials for some of datasets) and training/evaluation (different batch sizes) - details in README.md. 