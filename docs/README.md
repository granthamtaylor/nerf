# Neural Radiance Fields - Flyte Tutorial

This repository contains a basic tutorial on using [Flyte](https://flyte.org/) to train Neural Radiance Field (NeRF) models to compress basic bitmap images.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)

## Introduction

Neural Radiance Fields (NeRF) is a method for representing 3D scenes using neural networks. This tutorial demonstrates how to use Flyte to orchestrate and manage the training and evaluation of a NeRF model for basic 2D images.

## Setup

To get started, follow these steps:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/granthamtaylor/nerf
    ```

2. **Install the required dependencies:**

    - `uv`
    - `just`

3 **Initialiize python environment**

    ```sh
    uv venv
    uv sync
    ```

## Usage

This tutorial includes several Flyte tasks and workflows to train and evaluate a NeRF model. Here are the basic steps to run the tutorial:

1. **Define Flyte tasks:**

    Flyte tasks are defined in the `tasks` directory. Each task represents a unit of work, such as data preprocessing, model training, or evaluation.

2. **Define Flyte workflows:**

    Workflows are defined in the `workflows` directory. A workflow orchestrates multiple tasks to achieve a specific goal, such as training a NeRF model.

3. **Run the workflow:**

    Use the Flyte CLI or Flyte console to launch the workflow. For example:

    - Run the model training workflow with a small image locally: `just dev`
    - Run the model training workflow with a larger image remotely `just run`
