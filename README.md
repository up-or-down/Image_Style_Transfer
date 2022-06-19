# Image_Style_Transfer 
realize image style transfer using basic components of neural network 
According to UCAS course:Intelligent Computing Systems 

# Basic NN Components [layers1](https://github.com/up-or-down/Image_Style_Transfer/blob/main/layers_1.py)
## FullyConnectedLayer：

<p align = 'center'>
<img src = 'examples/style/udnie.jpg' height = '246px'>
</p>

### forward
$$ \boldsymbol{y}=\boldsymbol{W}^{T}\boldsymbol{x}+\boldsymbol{b} $$
### backward
Define $\nabla_\boldsymbol{y} L $ as partial derivative of Loss function  $L$ to  $y$ 

$$ \nabla_\boldsymbol{W} L=x\nabla_\boldsymbol{y} L^T $$

$$ \nabla_\boldsymbol{b} L=\nabla_\boldsymbol{y} L $$

$$ \nabla_\boldsymbol{x} L=\boldsymbol{W}^T\nabla_\boldsymbol{y} L $$

## ReLULayer：
### forward

### backward

## SoftmaxLossLayer：



2、
