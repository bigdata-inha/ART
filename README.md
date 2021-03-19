# ART 

# Attractive & Repulsive Trainig 
Code implementation of paper, ART: Class-Incremental learning using Class-wise Flags

# Requirements
1. pytorch, python : pytorch 1.7, python 3.8

# Experimental Result on overall accuracy
1. Data(Model) : Cifar100(ResNet32)
  <table> 
    <thead> 
     <tr> 
      <th rowspan=2></th>
      <th colspan=4>Cifar100(ResNet32)</th>
     </tr>
     <tr> 
      <th>5task (20 class)</th>
      <th>10task (10 class)</th>
      <th>20task (5 class)</th>
      <th>50task (2 class)</th>
     </tr>
    </thead> 
    <tbody align='center'> 
     <tr> 
      <td>r+KD</td>
      <td>57.73</td>
      <td>57.00</td>
      <td>55.63</td>
      <td>59.08</td>
     </tr>
     <tr> 
      <td>iCaRL</td>
      <td>57.63</td>
      <td>57.60</td>
      <td>55.68</td>
      <td>58.87</td>
     </tr>
     <tr> 
      <td>EEIL</td>
      <td>63.33</td>
      <td>61.59</td>
      <td>57.40</td>
      <td>57.29</td>
     </tr>
     <tr> 
      <td>BiC</td>
      <td>65.19</td>
      <td>61.59</td>
      <td>57.40</td>
      <td>57.29</td>
     </tr>
     <tr> 
      <td>WA</td>
      <td>65.26</td>
      <td>63.91</td>
      <td>61.83</td>
      <td>60.77</td>
     </tr>
     <tr> 
      <td>CTL+WA</td>
      <td>58.07</td>
      <td>57.07</td>
      <td>55.12</td>
      <td>55.07</td>
     </tr>
     <tr> 
      <td>ART+WA(ours)</td>
      <td>65.72</td>
      <td>65.80</td>
      <td>63.29</td>
      <td>62.52</td>
     </tr>
    </tbody> 
</table>


2. Data(Model) : , Tiny-ImageNet(WRN-16-2)

# Experimental Result on inter-task accuracy 

# Figure
1. Cosine similarity between classes on feature space
1.1. CIFAR-10
1.2. CIFAR-100
