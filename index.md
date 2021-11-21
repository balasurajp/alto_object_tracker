---
layout: default
---

<!-- 
Text can be **bold**, _italic_, or ~~strikethrough~~.

[Link to another page](./another-page.html).

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

# Header 1

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

## Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
 -->
 
 
# Introduction

Visual object tracking is a classical problem of estimating the trajectory of an target object in a video sequence, provided its location in the first frame. It is one of the fundamental problems in computer vision and can serve as a building block in complex vision systems. There are wide range of practical applications such as automatic surveillance, autonomous driving, Medical vision systems and Video analysis. This project will deal with the problem of improving the accuracy of state-of-the-art Siamese family trackers by incorporating adversarial mechanism into their training. There are number of challenges in designing a robust tracker because the target object undergoes a variety of complex appearance changes.

Object Transformations:
  - Illumination changes
  - Scale changes
  - Shape changes
  - Rotation changes
  - Background clutter
  - Motion blur
  - Occlusion

# Thesis Statement

In this project, we design a generic, robust and novel end-to-end framework with two primary things to address in Visual Object Tracking. They are:

To utilize the backbones in various deep CNN architectures by investigating the control aspects governing the tracking accuracy and robustness.
  - To harness power of residual module in deep CNNs for creating powerful embeddings invariant of changes in object appearances.
  - To ensemble similarity learning and dissimilarity learning network and get combined response for better localization precision.
  - To improve target localization by combined decision of different spatial resolution responses.

To explore the role of adversarial learning in improving the power of deep convolution embedding networks to enhance tracking accuracy .
  - To improve the foreground-background discriminative capabilities of siamese networks through harnessing the generative power from adversarial learning.
  - To design a novel end-to-end learning framework for tracking by combining two powerful breakthroughs Siamese networks and Generative Adversarial Networks (GANs) for balanced accuracy and speed.

# Proposed Implementation 

In this project, we have implemented several ways of improving the performance of Siamese Networks for object tracking.

  - **Ensemble of Similarity and Dissimilarity network**: In this experiment, we train two siamese networks(i.e Similarity Network & Dissimilarity Network) independently, in which one learns similarity between two image patches and vice-versa. In other words, it is an efficient way of designing correlation and anti-correlation blocks. In dissimilarity net, the ground-truth of score map is designed exactly opposite to siamese network. Then, the final response map is the combination obtained by subtracting Disimilarity Network's score map from Similarity Network's score map.

  <p align="center"> <img src="images/sndn.png" /> </p>

  - **Fusion of coarse-fine response maps in Siamese network**: In any siamese network version, we must obtain two embeddings from the network for each image i.e. one embedding is collected from intermediate convolution layer till where effective stride’s 4 and other embedding is obtained from end of the network having effective stride 8. Then, correlate the embeddings with effective stride 4, obtained from the template and search image to response maps of sizes 33. Similarly, we obtain score map with size 17 by correlating embeddings with effective stride 8.

  <p align="center"> <img src="images/fcfrm.png" /> </p>

  - **Siamese network with modified residual blocks**: In SiamFC tracker, the backbone CNN (AlexNet) is shallow in nature, which does not take complete advantage of the capability of modern deep CNNs. In this experiment, we leverage the power of deep convolutional neural networks to enhance tracking robustness and accuracy. We observe that directly replacing the backbones with existing powerful architectures, such as ResNet does not improve the tracking perspective. After analyzing the siameseFC network, there are two inferred reasons that control the tracking performance. 1) Increasing the size of receptive field leads to reduction of feature discrimination and localization precision and 2) the padding factor for convolutions blocks induces a positional bias and redundant features in learning. To address these issues, we can built new residual blocks to eliminate the negative impact of padding, and design new architectures using these blocks with controlled size of receptive field and effective network stride. The designed architectures are lightweight in nature to guarantee real-time tracking speed.

  <p align="center"> <img src="images/rnsa.png" /> </p>

  - **Adversarial learning into siamese architecture**: In this framework, the key concept is the integration of similarity-learning CNN into the GAN for leveraging the benefits such as balanced accuracy, high speed and high-end generative power. We propose that adversarial learning can improve the precise target localization through the feedback from patch based discriminator, classifying the predicted target patch cropped using the response 27 map and the actual ground-truth patch. To make training procedure converge fast, we incorporate the additional constraint such as distance regression between the predicted and the actual target co-ordinates. Unlike the conventional GANs, our generator is a similarity learning CNN which predicts the trajectory of the target motion by minimizing the error between the predicted and the actual trajectory through the adversarial learning.

  <p align="center"> <img src="images/alto.png" /> </p>

# Qualitative Results

  We show some qualitative results of this tracker on standard benchmark sequences from VOT2016 dataset. The first sequence (bolt1) contain frequent changes in target pose leading to large changes in the target state, and is well suited to evaluate our approach. The second sequence (motocross1sequence) contain the geometric and photo-metric variations such as rotations and illumination changes. For comparison, we included the output of the state-of-the-art Siamese FC tracker. These approaches employ multi-scale search algorithm for determining the target scale i.e width and height of the bounding box. The results show that our adversarial learning approach improves the target localization indeed improving the target scale estimation.

  <p align="center"> <img src="images/qr1.png" /> </p>

  We compare the original shallow SiamFC- AlexNet with deep SiamResNet22 with OTB15, OTB15, VOT16 and VOT17 benchmark datasets to prove that deep state-of-art architectures can improve tracking accuracy and robustness with necessary modifications in impact parameters.
 
  <p align="center"> <img src="images/qr2.png" /> </p>

  Our experiments show that the proposed adversarial approach outperforms the baseline approach and also performs better than many state-of-the-art methods on the VOT2016 benchmark. Though the proposed adversarial learning based tracking framework is demonstrated with Siamese network, mainly due to its simplicity, the key idea can be extended to other versions of Siamese trackers for the generalization.
  
  <p align="center"> <img src="images/qr3.png" /> </p>

  We compare our proposed adversarial approach with SiamFC to display the success plot and precision plot over all OTB100 benchmark challenging videos and Temple128 dataset videos. This tracker outperforms SiamFC by achieving Area Under Curve to comparable percentage.

  <p align="center"> <img src="images/qr4.png" /> </p>
  <p align="center"> <img src="images/qr5.png" /> </p>

# Conclusion 

In this project, the problem of accurate localization in tracking was studied and addressed. The proposed framework ALTO enables better location prediction of the target in similarity-learning trackers through incorporation of the adversarial learning, through correction of the predictions made by the baseline tracker. This approach was shown to provide the best results on the well known challenging tracking datasets, outperforming other state-of-the-art trackers, thus demonstrating the impact of the proposed approach. Though the proposed framework is demonstrated with Siamese network, mainly due to its simplicity, the key concept of the idea can be extended to
other versions of Siamese trackers. Hence, we conclude that this kind of adversarial framework for tracking has paramount importance and holds unprecedented scopes for further improvements in the field of object tracking. There is a necessity for extensive baseline experiments to investigate and understand the impact of the various design choices on the proposed methodology.