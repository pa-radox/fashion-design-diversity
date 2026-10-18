# Exploring Design Diversity in AI-Generated Fashion

## Overview

Generative artificial intelligence has the potential to assist fashion designers by producing multiple design concepts from a single initial idea. However, the extent to which AI can generate diverse garment designs while working from the same sketch remains an important question.

This project investigates the diversity of AI-generated fashion designs by using the same garment sketches as inputs and generating multiple outputs under different conditioning configurations. Using Multimodal Garment Designer (MGD), the study examines how much variation can be achieved from a single design concept and how generation settings influence the resulting outputs.

## Research Question

How diverse can AI-generated garment designs be when generated from the same input sketch, and how do different conditioning configurations influence this diversity?

## Objectives

- Investigate the range of garment designs that can be generated from a single sketch.
- Measure the diversity of generated designs across multiple generations.
- Examine how changes in conditioning configurations affect the variation between outputs.
- Analyze the relationship between design diversity and the preservation of the original garment concept.

## Methodology

### Model

This study uses Multimodal Garment Designer (MGD), an image-to-image garment-generation model built on a Stable Diffusion inpainting architecture. MGD incorporates garment sketches, human pose information, and textual information to guide the generation of garment designs.

Rather than developing a new generative architecture, this project evaluates the diversity of designs produced by the model under different generation conditions.

### Experimental Procedure

1. Select a set of garment sketches to serve as input designs.
2. Generate multiple garment designs from each sketch using different random seeds.
3. Repeat the generation process under different conditioning configurations.
4. Compare the resulting designs to quantify variation within outputs generated from the same input.
5. Analyze the results to determine how conditioning settings influence design diversity.

The same input sketches will be used across experimental configurations wherever possible, allowing differences in the generated outputs to be examined under controlled conditions.

### Experimental Variables

The experiment focuses on two conditioning parameters:

- **Sketch-conditioning rate (`sketch_cond_rate`):** Controls the influence of the input sketch during generation.
- **Starting-image-conditioning rate (`start_cond_rate`):** Controls the influence of the starting-image condition during generation.

These parameters will be varied systematically to investigate their relationship with the diversity of generated designs. Other generation settings will be held constant wherever possible.

## Dataset

The project uses garment-design and virtual try-on data to obtain suitable inputs and visual references.

Potential data sources include:

- **DressCode:** A dataset of garment images and associated visual information.
- **VITON-HD:** A high-resolution virtual try-on dataset containing clothing images and associated annotations.

The final dataset description will specify the selected data source, sample selection criteria, preprocessing steps, and number of sketches used in the experiments.

## Evaluation

### Design Diversity

The primary focus of the evaluation is the diversity of generated garment designs from the same input sketch.

Potential evaluation approaches include:

- **Visual feature diversity:** Measuring differences between generated designs in an image-feature representation space.
- **Structural diversity:** Examining variation in garment silhouettes, shapes, and design components.
- **Visual analysis:** Comparing output grids to identify differences in colors, patterns, details, and overall design.

### Preservation of the Original Design

Design diversity will also be examined in relation to the original sketch. This secondary consideration helps distinguish meaningful design variations from outputs that depart so substantially from the input that they no longer represent the same garment concept.

The final evaluation will specify the exact metrics used and explain how each metric reflects the study's definition of design diversity.

## Results

*Results will be added after the experiments and analysis are complete.*

### Quantitative Results

| Conditioning Configuration | Design Diversity | Design Preservation |
|---|---:|---:|
| Configuration 1 | TBD | TBD |
| Configuration 2 | TBD | TBD |
| Configuration 3 | TBD | TBD |

The completed analysis will report the selected metrics across multiple sketches and generations.

### Qualitative Results

Visual comparisons will show multiple designs generated from the same sketch. These examples will illustrate the range of variation achievable under different conditioning configurations.

### Discussion

The analysis will address:

- The range of design diversity achievable from a single garment sketch.
- The influence of conditioning configurations on generated variation.
- The types of design changes produced by the model.
- The relationship between diversity and consistency with the original concept.
- Limitations of the model, dataset, and evaluation methods.

## Technologies

- Python
- PyTorch
- Hugging Face Diffusers
- Stable Diffusion inpainting
- NumPy
- Image-processing and evaluation libraries

## Project Status

**In progress.** Model configuration, dataset preparation, image generation, and evaluation are being developed and tested. Results and conclusions will be added as experiments are completed.

## Reproducibility

The final project documentation will include the software environment, model checkpoints, dataset details, conditioning configurations, random seeds, sample counts, and evaluation procedures needed to reproduce the experiments.

Large datasets and pretrained model weights are not included in this repository unless explicitly stated.

## Acknowledgments

This project builds upon the Multimodal Garment Designer implementation by the original authors.

Original repository: [aimagelab/multimodal-garment-designer](https://github.com/aimagelab/multimodal-garment-designer)
