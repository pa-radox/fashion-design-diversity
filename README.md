# Exploring Design Diversity in AI-Generated Fashion

## Overview

Generative artificial intelligence can assist fashion designers by producing multiple design concepts from a single initial idea. However, how much diversity can generative models achieve when given the same garment sketch?

This project investigates the diversity of AI-generated fashion designs by generating multiple outputs from identical sketch inputs and examining how generation settings affect the resulting designs.

The study uses Multimodal Garment Designer (MGD) as its primary model and investigates its behavior under different conditioning configurations. The broader goal is to evaluate the range of designs AI can produce from a shared starting point and investigate how this diversity can be measured and compared across generative models.

## Research Question

**How diverse can AI-generated garment designs be when generated from the same input sketch?**

## Research Contributions

This project builds upon an existing generative model rather than introducing an entirely new architecture. The original model and the research contributions are distinguished below.

### Original Implementation

The following components originate from the existing [Multimodal Garment Designer (MGD) repository](https://github.com/aimagelab/multimodal-garment-designer):

- **Model architecture:** The underlying garment-generation architecture and its integration with Stable Diffusion inpainting.
- **Pretrained weights:** The existing pretrained model parameters.
- **Generation pipeline:** The original framework for generating garment designs using multimodal conditioning.
- **Existing model utilities:** Supporting code and preprocessing components inherited from the original implementation.

These components provide the foundation for the experiments conducted in this project.

### My Contributions

The original implementation is used as the basis for an independent investigation of AI-generated design diversity.

My contributions include:

- **Research design:** Formulating a research question focused on the diversity of AI-generated garment designs.
- **Experimental configuration:** Modifying and systematically testing conditioning parameters to investigate their effects on generated outputs.
- **Repeated generation:** Generating multiple designs from the same input sketches using different random seeds to examine the variation between outputs.
- **Diversity evaluation:** Developing an evaluation approach to quantify and analyze differences between generated designs.
- **Visual analysis:** Comparing generated designs to identify variation in garment silhouettes, shapes, colors, patterns, and details.
- **Model comparison:** Developing a framework for comparing MGD with other generative models under controlled input conditions, where technically feasible.

The evaluation, comparison, and analysis components are part of the research workflow and will be documented according to their implementation status. The project does not claim ownership of the original MGD architecture or pretrained model.

## Methodology

### Primary Model

The primary model is Multimodal Garment Designer (MGD), an image-to-image garment-generation system that uses garment sketches and additional conditioning information to guide generation.

The study focuses on evaluating the diversity of its outputs rather than developing a new generative architecture.

### Experimental Procedure

1. Select garment sketches to serve as input designs.
2. Generate multiple outputs from each sketch using different random seeds.
3. Repeat generation under different conditioning configurations.
4. Compare outputs generated from the same sketch.
5. Calculate diversity metrics and conduct visual analysis.
6. Where feasible, compare results with other generative models using matched inputs and documented settings.

Using identical input sketches across configurations allows the study to investigate how generation settings influence the variation observed between outputs.

### Experimental Variables

The experiment investigates two conditioning parameters:

- **Sketch-conditioning rate (`sketch_cond_rate`):** A parameter used to control the influence of sketch conditioning.
- **Starting-image-conditioning rate (`start_cond_rate`):** A parameter used to control the influence of the starting-image condition.

These settings are varied to investigate their relationship with generated design diversity. Final parameter values will be documented alongside the experimental results.

Other generation settings, including the input sketch and relevant sampling parameters, will be controlled wherever possible.

## Dataset

The project uses garment-design data to provide input sketches and relevant visual information.

Potential data sources include:

- **DressCode:** A virtual try-on dataset containing garment images and associated annotations.
- **VITON-HD:** A high-resolution virtual try-on dataset containing clothing images and associated annotations.

The final documentation will identify the datasets and splits actually used, the number of input sketches evaluated, and the preprocessing procedures applied.

## Evaluation

### Primary Evaluation: Design Diversity

The primary objective is to determine how much variation can be produced from a single garment sketch.

Potential evaluation approaches include:

- **Visual feature diversity:** Measuring differences between generated images in a feature representation space.
- **Structural diversity:** Examining variation in garment silhouettes, shapes, and major design components.
- **Visual analysis:** Comparing generated designs to identify differences in patterns, colors, details, and overall appearance.

The final implementation will specify which metrics are used, how they are calculated, and what aspects of diversity they measure.

### Secondary Evaluation: Design Preservation

Diversity alone does not establish whether generated outputs remain relevant to the original design.

A secondary evaluation will examine how well generated garments preserve important structural characteristics of the input sketch. This helps contextualize the diversity measurements and distinguish meaningful design variation from outputs that depart substantially from the intended garment concept.

## Results

*Experimental results will be added after data collection and analysis are complete.*

### Quantitative Results

The final report will include the selected diversity metrics, summary statistics, sample counts, and comparisons across experimental configurations.

| Experimental Configuration | Diversity Metric(s) | Design Preservation |
|---|---|---|
| Configuration 1 | TBD | TBD |
| Configuration 2 | TBD | TBD |
| Configuration 3 | TBD | TBD |

### Qualitative Results

Visual comparisons will show multiple designs generated from the same input sketch. These examples will illustrate the range of variation produced under different generation conditions.

### Discussion

The analysis will investigate:

- The range of design diversity achievable from a single sketch.
- How conditioning configurations influence generated variation.
- Which aspects of garment design vary most across outputs.
- How design diversity relates to preservation of the original concept.
- How the observed results compare across generative models, where applicable.
- Limitations of the datasets, models, and evaluation methods.

## Technologies

- Python
- PyTorch
- Hugging Face Diffusers
- Stable Diffusion inpainting
- NumPy
- Image-processing and evaluation libraries

## Project Status

**In progress.**

The project builds upon the existing MGD implementation while developing an experimental framework for investigating AI-generated garment-design diversity. Generation experiments, evaluation, and model comparisons will be documented as they are implemented and completed.

## Reproducibility

The final repository will document:

- Software environment and dependency versions.
- Original model and checkpoint sources.
- Dataset sources, splits, and preprocessing.
- Experimental conditioning configurations.
- Random seeds and generation settings.
- Evaluation metrics and analysis procedures.

Large datasets and pretrained weights are not included unless explicitly stated.

## Acknowledgments

This project builds upon the Multimodal Garment Designer implementation.

**Original repository:** [aimagelab/multimodal-garment-designer](https://github.com/aimagelab/multimodal-garment-designer)

The original authors retain credit for their model, architecture, and existing implementation. This repository documents the independent research experiments and analysis built around that work.
