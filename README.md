# ART 

# Attractive & Repulsive Trainig 
Code implementation of paper, ART: Class-Incremental learning using Class-wise Flags

# Requirements
1. pytorch, python : pytorch 1.7, python 3.8

# Experimental Result
1. Data(Model) : Cifar100(ResNet32)
  <table> 
    <thead> 
     <tr> 
      <th rowspan=2>Dataset</th>
      <th colspan=4>Cifar10</th>
     </tr>
     <tr> 
      <th>Major / Minor</th>
      <th>Major class images per group</th>
      <th>Minor class images per group</th>
      <th>Accuracy</th>
     </tr>
    </thead> 
    <tbody align='center'> 
     <tr> 
      <td>Balanced</td>
      <td rowspan=4>[0~4] / [5~9]</td>
      <td rowspan=4>5000</td>
      <td>5000</td>
      <td>91.29</td>
     </tr>
     <tr> 
      <td>Imbalance 20</td>
      <td>250</td>
      <td>68.93</td>
     </tr>
     <tr> 
      <td>Imbalance 50</td>
      <td>100</td>
      <td>58.51</td>
     </tr>
     <tr> 
      <td>Imbalance 100</td>
      <td>50</td>
      <td>52.45</td>
     </tr>
    </tbody> 
</table>


2. Data(Model) : , Tiny-ImageNet(WRN-16-2)


# Figure
1. Cosine similarity between classes on feature space
1.1. CIFAR-10
1.2. CIFAR-100