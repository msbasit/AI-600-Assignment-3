# Assignment 3 Results and Analysis

This README is auto-generated from `pa3_notebook.ipynb`.
It includes analysis text, key numeric outputs (epochs/loss/accuracies), and all notebook figures.

## Analysis

## Task 1A – Standard MNIST CNN

**Objective:** Design a compact CNN (≤ 3 conv layers, ≤ 2 FC layers, ≤ 50 k parameters),
train it on MNIST with Adam + CrossEntropyLoss, and analyse the learned filters.

### 1A-1: Data Loading

### 1A-2: Model Architecture

### 1A-3: Training

### 1A-4: Learning Curves

**Analytical Question 1.1 – Overfitting, underfitting, or good generalisation?**

Looking at the loss and accuracy curves, the training and validation lines track each other
closely throughout all 15 epochs.  The final training accuracy and validation accuracy are
within roughly 0.5 percentage points of each other, and both loss curves decrease together
without the validation curve turning back upward.  This is the signature of a model that
generalises well: there is no growing gap between the two curves, which would indicate
overfitting, and the model is not stuck at a high loss, which would indicate underfitting.
The small parameter count (< 50 k), the Dropout layer, and the weight decay in Adam all
work together to keep the model from memorising the training set.

### 1A-5: First-Layer Filter Visualisation

**Analytical Question 1.2 – What are the filters detecting?**

Even with only 3×3 kernels, we can identify rudimentary visual patterns in the filters
after training.  Filters that have a positive region on one side and a negative region
on the other act as edge detectors: when such a filter slides over an image, it responds
strongly wherever pixel intensity transitions sharply from dark to light (or vice versa).
Some filters show a centre-positive / surround-negative pattern, which is similar to a
Laplacian and responds to blobs or corners.  Others appear more diffuse, possibly acting
as low-frequency smoothers that carry average brightness information.  These behaviours
are consistent with what is known about early convolutional layers in well-trained CNNs:
they learn to detect basic local structure (edges, textures, orientations) that higher
layers then combine into class-discriminative representations.

## Task 1B – Colored MNIST (C-MNIST)

**Objective:** Adapt the MNIST CNN for 3-channel RGB input, train on biased C-MNIST,
then evaluate on both the biased and unbiased test sets to observe shortcut learning.

### 1B-1: Data Loading

### 1B-2: Visualise the Colour-Digit Bias

### 1B-3: Model – RGB Adaptation

### 1B-4: Training on Biased C-MNIST

### 1B-5: Evaluation on Both Test Sets

**Analytical Question 1.3 – Why the large drop on the unbiased test set?**

The model is trained on a dataset where every digit class is perfectly correlated with a
specific colour.  From the perspective of the loss minimiser (Adam with cross-entropy), colour
is an extremely easy feature to learn: it can be extracted by the very first convolutional layer
in a single pass, and it provides nearly perfect classification on the training distribution.
Shape, on the other hand, requires building up hierarchical representations across multiple
layers, and its gradients are weaker because the shape alone is sufficient but not the
easiest path to low training loss.

Mathematically, the decision boundary learned by the network encodes:

$$P(y = c \mid x) \approx P(y = c \mid \text{colour}(x))$$

because the colour feature is essentially a lookup table: class $c$ has colour $k_c$ with
probability $\approx 1$ in the training set.  Any filter that responds to that colour will
receive a large gradient signal that drives its weights far from zero.  Shape-based filters
also receive gradient, but the colour signal dominates because both features are equally
predictive during training and colour is much simpler to represent.

On the unbiased test set the colour–class pairing is broken.  The model's decision boundary,
which was primarily colour-based, is no longer correlated with the true labels, and accuracy
collapses.  This is a canonical instance of **shortcut learning**: the network exploits a
spurious correlation present in the training distribution rather than learning the causal
feature (digit shape) that generalises to new distributions.

**Analytical Question 1.4 – Training strategies to force shape learning**

Several approaches can discourage shortcut learning on colour:

1. **Greyscale augmentation during training** – Randomly convert images to grayscale (and then
   back to 3-channel) with some probability $p$.  This forces the model to maintain shape-based
   representations that still work when colour is absent.

2. **Colour jitter augmentation** – Apply random hue, saturation, and brightness perturbations
   so that the fixed digit–colour mapping is disrupted during training.  With a strong enough
   jitter the colour becomes an unreliable cue and the network must rely on shape.

3. **Learning Invariant Predictors (IRM)** – A principled method that adds a regularisation
   term to the loss penalising any classifier whose gradient is non-zero on a "dummy" invariant
   classifier.  It explicitly discourages features that work only within one environment
   (colour-paired) rather than across all environments.

4. **Resampling / reweighting** – If a held-out unbiased validation set is available, up-weight
   examples where the spurious correlation is absent so that the loss landscape favours
   shape-based features.

5. **Adversarial colour removal head** – Add a secondary classifier trained to predict colour
   and negate its gradient before it flows into the shared backbone (gradient reversal layer).
   This explicitly removes colour information from the learned representation.

## Task 2A – Transfer Learning: Fine-tuning ResNet-18 on STL-10

### 2A-1: Data Loading – STL-10

### 2A-2: Load Pre-trained ResNet-18 and Freeze Backbone

### 2A-3: Training the Classification Head

**Analytical Question 2.1 – Why freeze the backbone?**

There are two reasons, one computational and one functional.

*Computationally*, freezing the backbone means gradients do not need to be computed or stored
for the vast majority of parameters (over 11 million in ResNet-18, versus 5 120 in the new
FC layer).  This dramatically reduces memory usage and speeds up each training step, which
matters when the dataset is small.

*Functionally*, the early layers of a CNN trained on ImageNet have learned to detect general
visual primitives: oriented edges (Gabor-like filters), colour blobs, and simple textures in
the shallowest layers; progressively more complex patterns (corners, curves, object parts) in
the middle layers.  These representations are largely dataset-agnostic and transfer well
to any natural-image classification task.  The later layers are more task-specific and encode
ImageNet class structure (distinguishing dog breeds, bird species, etc.), which is less useful
for STL-10.

Fine-tuning all layers on a small dataset like STL-10 (5 000 training images) would risk
catastrophic forgetting of the general low-level features and lead to overfitting, because
there are far more parameters than needed to explain the small training distribution.  By
keeping the backbone frozen, we effectively use it as a fixed feature extractor and only
teach the last layer the STL-10 class boundaries.

## Task 2B – Visualising Decisions with GradCAM

### 2B-1: GradCAM Implementation

### 2B-2: Collect 4 Test Images (2 correct, 2 incorrect)

### 2B-3: Generate and Display GradCAM Heatmaps

**Analytical Question 2.2 – What do the correct-prediction heatmaps show?**

For the two correctly classified images, the GradCAM heatmap concentrates its high-activation
region (red and yellow in the jet colourmap) on the primary object.  For example, if the true
class is "airplane", the hottest pixels fall over the fuselage and wings rather than over the
sky background.  For "dog", activation concentrates around the animal's head or body.

This is the expected behaviour of a well-trained network: the model has learned to associate
class-discriminative features (the shape of wings, the texture of fur) with its prediction.
The background regions (sky, grass, walls) appear mostly blue, confirming that the network
is not relying on context to make its decision for these samples.

**Analytical Question 2.3 – What do the incorrect-prediction heatmaps show?**

For the two incorrectly classified images the heatmap reveals the source of the error.
Common failure patterns include:

- **Background confusion**: The network's attention falls on the background (e.g., a large blue
  sky region) rather than on the object.  If a background colour or texture happens to be
  more diagnostic of the wrong class in the training set, the model will exploit it.

- **Part confusion**: The heatmap highlights a localised object part that resembles a part of
  the predicted (wrong) class.  For example, a ship's prow might activate the same filters
  as an airplane nose, causing a misclassification.

- **Scale or occlusion**: When the object is small, occluded, or at an unusual angle, the
  model may fail to activate the right feature maps and instead rely on whatever contextual
  cues are available, leading to an incorrect decision.

In all these cases, GradCAM makes the misclassification interpretable: instead of just
knowing the model was wrong, we can see *why* it was wrong and which part of the image
it was focused on when it made its error.

## Results

### Result 1: 1A-1: Data Loading

**Key Metrics**

- Train: 54,000  |  Val: 6,000  |  Test: 10,000

### Result 2: 1A-2: Model Architecture

**Key Metrics**

- Total trainable parameters: 8,650  (limit: 50,000)

### Result 3: 1A-3: Training

**Key Metrics**

- Epoch 01/15  train_loss=1.2214  val_loss=0.4862  train_acc=56.94%  val_acc=86.18%
- Epoch 02/15  train_loss=0.4256  val_loss=0.2734  train_acc=87.13%  val_acc=91.72%
- Epoch 03/15  train_loss=0.2995  val_loss=0.2133  train_acc=91.06%  val_acc=93.38%
- Epoch 04/15  train_loss=0.2475  val_loss=0.1734  train_acc=92.69%  val_acc=94.83%
- Epoch 05/15  train_loss=0.2160  val_loss=0.1721  train_acc=93.71%  val_acc=94.60%
- Epoch 06/15  train_loss=0.1857  val_loss=0.1494  train_acc=94.61%  val_acc=95.50%
- Epoch 07/15  train_loss=0.1792  val_loss=0.1444  train_acc=94.67%  val_acc=95.70%
- Epoch 08/15  train_loss=0.1642  val_loss=0.1316  train_acc=95.18%  val_acc=96.00%
- Epoch 09/15  train_loss=0.1580  val_loss=0.1308  train_acc=95.38%  val_acc=96.12%
- Epoch 10/15  train_loss=0.1494  val_loss=0.1161  train_acc=95.61%  val_acc=96.53%
- Epoch 11/15  train_loss=0.1378  val_loss=0.1132  train_acc=96.00%  val_acc=96.53%
- Epoch 12/15  train_loss=0.1336  val_loss=0.1136  train_acc=96.02%  val_acc=96.50%
- Epoch 13/15  train_loss=0.1325  val_loss=0.1035  train_acc=96.20%  val_acc=96.83%
- Epoch 14/15  train_loss=0.1252  val_loss=0.1031  train_acc=96.28%  val_acc=96.92%
- Epoch 15/15  train_loss=0.1229  val_loss=0.1031  train_acc=96.44%  val_acc=96.95%

### Result 4: 1A-3: Training

**Key Metrics**

- Final test accuracy: 97.31%

### Result 5: 1A-4: Learning Curves

**Output Snippet**

- <Figure size 1300x400 with 2 Axes>

**Figures**

![1A-4: Learning Curves](readme_images/cell_015_img_1.png)

### Result 6: 1A-5: First-Layer Filter Visualisation

**Output Snippet**

- <Figure size 1400x250 with 9 Axes>

**Figures**

![1A-5: First-Layer Filter Visualisation](readme_images/cell_018_img_1.png)

### Result 7: 1B-1: Data Loading

**Key Metrics**

- Train (biased):        torch.Size([60000, 3, 28, 28])  labels: torch.Size([60000])
- Test  (biased):        torch.Size([10000, 3, 28, 28])
- Test  (unbiased):      torch.Size([10000, 3, 28, 28])

### Result 8: 1B-1: Data Loading

**Key Metrics**

- Train: 54,000  Val: 6,000  Biased test: 10,000  Unbiased test: 10,000

### Result 9: 1B-2: Visualise the Colour-Digit Bias

**Output Snippet**

- <Figure size 1600x400 with 20 Axes>

**Figures**

![1B-2: Visualise the Colour-Digit Bias](readme_images/cell_025_img_1.png)

### Result 10: 1B-3: Model – RGB Adaptation

**Key Metrics**

- C-MNIST model parameters: 8,794

### Result 11: 1B-4: Training on Biased C-MNIST

**Key Metrics**

- Epoch 01/15  train_loss=0.7884  val_loss=0.4109  train_acc=82.03%  val_acc=93.70%
- Epoch 02/15  train_loss=0.3824  val_loss=0.3348  train_acc=94.16%  val_acc=94.48%
- Epoch 03/15  train_loss=0.3354  val_loss=0.3037  train_acc=94.81%  val_acc=94.57%
- Epoch 04/15  train_loss=0.3016  val_loss=0.2826  train_acc=94.95%  val_acc=94.62%
- Epoch 05/15  train_loss=0.2735  val_loss=0.2526  train_acc=95.10%  val_acc=94.83%
- Epoch 06/15  train_loss=0.2473  val_loss=0.2326  train_acc=95.30%  val_acc=94.97%
- Epoch 07/15  train_loss=0.2296  val_loss=0.2255  train_acc=95.41%  val_acc=95.07%
- Epoch 08/15  train_loss=0.2132  val_loss=0.2026  train_acc=95.51%  val_acc=95.18%
- Epoch 09/15  train_loss=0.1979  val_loss=0.1847  train_acc=95.66%  val_acc=95.40%
- Epoch 10/15  train_loss=0.1808  val_loss=0.1649  train_acc=95.81%  val_acc=95.62%
- Epoch 11/15  train_loss=0.1641  val_loss=0.1568  train_acc=96.04%  val_acc=95.72%
- Epoch 12/15  train_loss=0.1555  val_loss=0.1509  train_acc=96.14%  val_acc=95.75%
- Epoch 13/15  train_loss=0.1501  val_loss=0.1419  train_acc=96.19%  val_acc=95.98%
- Epoch 14/15  train_loss=0.1429  val_loss=0.1363  train_acc=96.32%  val_acc=96.08%
- Epoch 15/15  train_loss=0.1361  val_loss=0.1305  train_acc=96.44%  val_acc=96.32%

### Result 12: 1B-4: Training on Biased C-MNIST

**Output Snippet**

- <Figure size 1300x400 with 2 Axes>

**Figures**

![1B-4: Training on Biased C-MNIST](readme_images/cell_030_img_1.png)

### Result 13: 1B-5: Evaluation on Both Test Sets

**Key Metrics**

- Accuracy on biased test set:   96.75%
- Accuracy on unbiased test set: 36.64%

### Result 14: 2A-1: Data Loading – STL-10

**Key Metrics**

- STL-10 train: 5,000  |  test: 8,000

### Result 15: 2A-2: Load Pre-trained ResNet-18 and Freeze Backbone

**Key Metrics**

- Frozen parameters:    11,176,512
- Trainable parameters: 513,000  (original FC only)
- New trainable parameters: 5,130

### Result 16: 2A-3: Training the Classification Head

**Key Metrics**

- Epoch 01/10  train_loss=0.7496  train_acc=81.64%
- Epoch 02/10  train_loss=0.2741  train_acc=93.02%
- Epoch 03/10  train_loss=0.2181  train_acc=93.68%
- Epoch 04/10  train_loss=0.1806  train_acc=94.76%
- Epoch 05/10  train_loss=0.1711  train_acc=94.88%
- Epoch 06/10  train_loss=0.1609  train_acc=95.12%
- Epoch 07/10  train_loss=0.1490  train_acc=95.66%
- Epoch 08/10  train_loss=0.1504  train_acc=95.52%
- Epoch 09/10  train_loss=0.1441  train_acc=95.86%
- Epoch 10/10  train_loss=0.1361  train_acc=96.12%

### Result 17: 2A-3: Training the Classification Head

**Key Metrics**

- STL-10 test accuracy (frozen backbone + linear head): 94.92%

### Result 18: 2A-3: Training the Classification Head

**Output Snippet**

- <Figure size 1300x400 with 2 Axes>

**Figures**

![2A-3: Training the Classification Head](readme_images/cell_043_img_1.png)

### Result 19: 2B-1: GradCAM Implementation

**Output Snippet**

- GradCAM ready

### Result 20: 2B-2: Collect 4 Test Images (2 correct, 2 incorrect)

**Output Snippet**

- Collected 2 correct and 2 incorrect samples

### Result 21: 2B-3: Generate and Display GradCAM Heatmaps

**Output Snippet**

- <Figure size 1800x1200 with 12 Axes>

**Figures**

![2B-3: Generate and Display GradCAM Heatmaps](readme_images/cell_051_img_1.png)
