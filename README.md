# Realtime style transfer
My implementation of [Perceptual Losses for Real-Time Style Transfer
and Super-Resolution](https://cs.stanford.edu/people/jcjohns/eccv16/)

## Train

1. Place the style image in `img` directory (Default is `starry_night.jpg`)
2. make `data/0` directory
3. Place image dataset in `data/0` directory
4. Run `python src/train.py`

## Test
`python src/test.py [WEIGHT_FILE] [CONTENT_IMAGE]`

## Requirement
- Python ($\ge$ 3.6)
- Pytorch (1.5.0)
- Pillow