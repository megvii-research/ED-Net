PyTorch implementation of A Lightweight Encoder-Decoder Path for Deep Residual Networks.


Requirements

    numpy
    
    torch-1.1.0
    
    torchvision-0.3.0
    

step
    
    1.download your dataset by yourself, such as ImageNet-1k
    
    2.create a list for your dataset,such as 
        imagename label
        xxx.jpg 1
        xxx.jpg 3
        xxx.jpg 999
    
    3.python3 imagenet_train.py --test_data_path your_path --train_data_path  your_path -a ED50 --epochs 100 --schedule 30 -b 256 --lr 0.1
