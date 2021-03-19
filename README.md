# ART 

# Attractive & Repulsive Trainig 
Code implementation of paper, ART: Class-Incremental learning using Class-wise Flags

# Prerequistes
1. pytorch, python : pytorch 1.7 ↑, python 3.7 ↑

# Measures Definition
1. Measure for the Easy / Hard
<table align='center'>
<tr align='center'>
</tr>
<tr>

</tr>
</table>

2. Not yet
<table align='center'>
<tr align='center'>
</tr>
<tr>

</tr>
</table>

3. Not yet
<table align='center'>
<tr align='center'>
</tr>
<tr>

</tr>
</table>

# Dataset / Model
1. Data / Model : Cifar100 / Resnet32
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


2. Data / Model : Imagenet / ResNet18


# Experiment Result
1. Neuron Membership 

2. Major class actvation - Minor class activation 

3. Class Selectivity
