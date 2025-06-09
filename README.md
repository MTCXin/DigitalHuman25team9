# DigitalHuman25 — Team 9: Project Repository

Welcome to the DigitalHuman25, Team 9 repository! This space hosts all the necessary code and resources to reproduce our project's results.

### Image Data

Before you begin, please download the required image dataset:

[**TODO:Download Image Data Here**](TODO_DOWNLOAD_LINK_HERE)

### Project Structure

For seamless execution, ensure your project directory is organized as follows:
```bash
.
├── train_xxx.py
├── calculate_xxx.py
└── cats/
└──     xxx.jpg
```
This structure ensures all scripts can correctly locate necessary files.

### Reproducing Our Results

To get started, **adjust the file paths within the scripts** to align with your specific working directory. Once configured, follow these steps:

* **Train the Fine-Tune Model:**
    Kick off the training process for our fine-tuned model:
    ```bash
    python3 train_dreambooth.py
    ```
* **Generate Image Samples:**
    To create new image samples, first **edit the `prompt_list` within `inference.py`** to define your desired outputs. Then, run:
    ```bash
    python3 inference.py
    ```
* **Generate Video Samples:**
    Similarly, for video generation, **modify the `prompt_list` in `animatediff.py`** with your chosen prompts. Execute the script:
    ```bash
    python3 animatediff.py
    ```
* **Calculate Performance Metrics:**
    To evaluate the performance of our models, run the following command:
    ```bash
    python3 calculate_xxx.py
    ```