# SpectFormer Model for Image Classification

Created by [Badri N. Patro](https://badripatro.github.io/), [Vinay P. Namboodiri](https://vinaypn.github.io/), [Vijay Srinivas Agneeswaran](https://in.linkedin.com/in/vijaysrinivasagneeswaran)


### Requirements

- python =3.8 (conda create -y --name spectformer python=3.8)
- torch>=1.10.0 (conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch)
- timm (pip install timm)


**Data preparation**: download and extract ImageNet images from http://image-net.org/. The directory structure should be

```
│ILSVRC2012/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Evaluation

To evaluate a pre-trained spectformer model on the ImageNet validation set with a single GPU, run:

```
python infer.py --data-path /path/to/ILSVRC2012/ --arch arch_name --model-path /path/to/model
```


### Training

#### ImageNet

To train spectformer models on ImageNet from scratch, run:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_spectformer.py  --output_dir logs/spectformer-xs --arch spectformer-xs --batch-size 128 --data-path /path/to/ILSVRC2012/
```

To finetune a pre-trained model at higher resolution, run:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_spectformer.py  --output_dir logs/spectformer-xs-img384 --arch spectformer-xs --input-size 384 --batch-size 64 --data-path /path/to/ILSVRC2012/ --lr 5e-6 --weight-decay 1e-8 --min-lr 5e-6 --epochs 30 --finetune /path/to/model
```

#### Transfer Learning Datasets

To finetune a pre-trained model on a transfer learning dataset, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_spectformer_transfer.py  --output_dir logs/spectformer-xs-cars --arch spectformer-xs --batch-size 64 --data-set CARS --data-path /path/to/stanford_cars --epochs 1000 --lr 0.0001 --weight-decay 1e-4 --clip-grad 1 --warmup-epochs 5 --finetune /path/to/model 
```
### The model weights will be released after acceptance
