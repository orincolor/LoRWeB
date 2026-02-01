# LoRWeB Overview

## Description:
LoRWeB is an image editing model that takes an input pair of example "before" and "after" images, and edits a new image based on them, as well as based on a textual prompt. LoRWeB dynamically constructs a single LoRA (Low-Rank Adaptation) from a learnable basis of LoRA modules to perform visual analogy tasks, generating an output image that applies a demonstrated transformation.

LoRWeB was developed by NVIDIA, Technion - Israel Institute of Technology, Bar-Ilan University as a part of an adaption on Flux.1-Kontext.

Visual analogy learning enables image manipulation through demonstration rather than textual description, allowing users to specify complex transformations difficult to articulate in words. Given a triplet a,a',b, the goal is to generate b' such that a:a' :: b:b'. Recent methods adapt text-to-image models to this task using a single Low-Rank Adaptation (LoRA) module, but they face a fundamental limitation: attempting to capture the diverse space of visual transformations within a fixed adaptation module constrains generalization capabilities. Inspired by recent work showing that LoRAs in constrained domains span meaningful, interpolatable semantic spaces, we propose LoRWeB, a novel approach that specializes the model for each analogy task at inference time through dynamic composition of learned transformation primitives, informally, choosing a point in a "space of LoRAs". We introduce two key components: (1) a learnable basis of LoRA modules, to span the space of different visual transformations, and (2) a lightweight encoder that dynamically selects and weighs these basis LoRAs based on the input analogy pair. Comprehensive evaluations demonstrate our approach achieves state-of-the-art performance and significantly improves generalization to unseen visual transformations. Our findings suggest that LoRA basis decompositions are a promising direction for flexible visual manipulation.

This model is ready for non-commercial/research use only.

### License/Terms of Use:
NVIDIA License [Non-Commercial]

## Model Architecture:
**Architecture Type:** Transformer
**Network Architecture:** Flux.1-Kontext

**This model was developed based on Flux.1-Kontext.

**Number of model parameters:** 835,051,520

### Input:
**Input Type(s):** Image, Text
**Input Format(s):** Jpeg, String
**Input Parameters:** Two-Dimensional (2D), One-Dimensional (1D)
**Other Properties Related to Input:** Image quadruplet {a, a', b, b} 512x512 images.

### Output:
**Output Type(s):** Image
**Output Format:** jpeg
**Output Parameters:** Two-Dimensional (2D)
**Other Properties Related to Output:** Image quadruplet {a, a', b, b'} 512x512 images, b' (the bottom-right quadrant) is the edited output.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems NVIDIA GPU (inferred from author affiliation). By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Supported Hardware Microarchitecture Compatibility:**
* Tested on NVIDIA Hopper, should be compatible with NVIDIA Ampere, NVIDIA Blackwell, NVIDIA Lovelace and NVIDIA Jetson.

**[Preferred/Supported] Operating System(s):**
* Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):
v1

LoRWeB can be integrated into an AI system by utilizing its inference capabilities through the provided `inference.py` script, which leverages model checkpoints from HuggingFace. This enables dynamic visual transformation tasks based on input image triplets.

## Training, Testing, and Evaluation Datasets:

### Training Dataset:
**Link:** Relation252K    https://huggingface.co/datasets/handsomeWilliam/Relation252K/tree/main

**Data Modality:**
* Image and text

**Image Training Data Size (If Applicable):**
* 33k image pairs [Less than a Million Images]

**Text Training Data Size (If Applicable):**
* 33k prompts [Less than a Billion Tokens]

**Data Collection Method by dataset:**
* Human, Automatic, Synthetic

**Labeling Method by dataset:**
* Human, Automatic, Synthetic

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**
Relation252K is an image-analogy dataset containing 33,274 images with accompanying textual prompt editing description across 218 editing tasks.
This yields 251,580 editing samples generated through image pair permutations.
Part of the data includes curated subsets of well-known benchmark datasets through manual collection. The data is further extended by adding synthetic images, editing instructions, and editing results generated using MidJourney and GPT-40.

### Testing & Evaluation Dataset:

**Link:** HuggingFace
https://unsplash.com/
https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev

**Data Collection Method by dataset:**
Hybrid: Human, Automatic, Synthetic

**Labeling Method by dataset:**
Hybrid: Human, Automatic, Synthetic

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**
The custom evaluation set extends Relation252k test set with 270 further curated images of animals, persons, and objects, edited in varying ways (e.g. added flame background, or different cloths).
108 images were first collected manually from [UnSplash](https://unsplash.com/). Then, editing instructions generated via GPT-4o and Claude Sonnet 4 to edit the collected images using [Flux.1-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) and various community LoRAs from [HuggingFace](https://huggingface.co).

### Inference:
**Acceleration Engine:** PyTorch

**Test Hardware:**
* NVIDIA H100

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please make sure you have proper rights and permissions for all input image and video content; if image includes people, personal health information, or intellectual property, the image generated will not blur or maintain proportions of image subjects included.

Please report model quality, risk, security vulnerabilities or NVIDIA AI concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).