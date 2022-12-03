export semi_setting='pascal/1_8/split_0'
python -W ignore main.py \
  --dataset pascal --data-root E:\note\ssl\data\voc_aug_3\VOCdevkit\VOC2012 \
  --batch-size 4 --backbone resnet50 --model deeplabv3plus \
  --labeled-id-path dataset/splits/$semi_setting/labeled.txt \
  --unlabeled-id-path dataset/splits/$semi_setting/unlabeled.txt \
  --pseudo-mask-path outdir/pseudo_masks/$semi_setting \
  --save-path outdir/models/$semi_setting