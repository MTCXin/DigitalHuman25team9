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

        * temp
    @ARTICLE{importanceGAI,
  author={Bengesi, Staphord and El-Sayed, Hoda and Sarker, MD Kamruzzaman and Houkpati, Yao and Irungu, John and Oladunni, Timothy},
  journal={IEEE Access}, 
  title={Advancements in Generative AI: A Comprehensive Review of GANs, GPT, Autoencoders, Diffusion Model, and Transformers}, 
  year={2024},
  volume={12},
  number={},
  pages={69812-69837},
  keywords={Decoding;Mathematical models;Task analysis;Vectors;Codes;Transformers;Neural networks;Generative AI;Generative adversarial networks;Artificial intelligence;Chatbots;Encoding;Generative AI;GPT;bard;ChatGPT;diffusion model;transformer;GAN;autoencoder;artificial intelligence},
  doi={10.1109/ACCESS.2024.3397775}}

@ARTICLE{SurveyDiffusion,
  author={Cao, Hanqun and Tan, Cheng and Gao, Zhangyang and Xu, Yilun and Chen, Guangyong and Heng, Pheng-Ann and Li, Stan Z.},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={A Survey on Generative Diffusion Models}, 
  year={2024},
  volume={36},
  number={7},
  pages={2814-2830},
  keywords={Mathematical models;Kernel;Computational modeling;Training;Surveys;Noise reduction;Markov processes;Diffusion model;deep generative model;diffusion algorithm;diffusion applications},
  doi={10.1109/TKDE.2024.3361474}}

  Citation for: "The ability to generate novel, high-quality images and videos featuring specific individuals or subjects holds immense potential across diverse applications, from personalized content creation in marketing and entertainment to advanced digital prototyping and virtual try-ons."


@misc{DALLE2,
      title={Hierarchical Text-Conditional Image Generation with CLIP Latents}, 
      author={Aditya Ramesh and Prafulla Dhariwal and Alex Nichol and Casey Chu and Mark Chen},
      year={2022},
      eprint={2204.06125},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2204.06125}, 
}
@misc{imagen,
      title={Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding}, 
      author={Chitwan Saharia and William Chan and Saurabh Saxena and Lala Li and Jay Whang and Emily Denton and Seyed Kamyar Seyed Ghasemipour and Burcu Karagol Ayan and S. Sara Mahdavi and Rapha Gontijo Lopes and Tim Salimans and Jonathan Ho and David J Fleet and Mohammad Norouzi},
      year={2022},
      eprint={2205.11487},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2205.11487}, 
}
Citation for: "...significant advancements in generative artificial intelligence, particularly with the advent of text-to-image diffusion models..."


@misc{TrainingFreeConsistent,
      title={Training-Free Consistent Text-to-Image Generation}, 
      author={Yoad Tewel and Omri Kaduri and Rinon Gal and Yoni Kasten and Lior Wolf and Gal Chechik and Yuval Atzmon},
      year={2024},
      eprint={2402.03286},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2402.03286}, 
}
@misc{FreeLunchConsistent,
      title={One-Prompt-One-Story: Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt}, 
      author={Tao Liu and Kai Wang and Senmao Li and Joost van de Weijer and Fahad Shahbaz Khan and Shiqi Yang and Yaxing Wang and Jian Yang and Ming-Ming Cheng},
      year={2025},
      eprint={2501.13554},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.13554}, 
}
Citation for: "...achieving truly identity-consistent generation remains a formidable challenge for existing methods. While these models excel at producing diverse and aesthetically pleasing outputs based on broad textual descriptions, they typically struggle to maintain the fine-grained characteristics of a specific individual across various poses, expressions, and environments."
