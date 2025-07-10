# Learning-to-Synthesize-a-4D-RGBD-Light-Field-from-a-Single-Image
PyTorch Implementation of the paper "Learning to Synthesize a 4D RGBD Light Field from a Single Image"

Paper: https://arxiv.org/abs/1708.03292 

## 🚀 Instructions for Training

- ✅ Setup the Python environment:
  ```bash
  pip install -r requirements.txt
  ```

- 📦 Prepare the dataset for training. Follow the file structure below (shown for a 2-image dataset with a 13×13 grid of sub-aperture views):

<pre>
📁 Light Field Dataset Root Directory
├── 📁 image_1
│   ├── 📄 view_0_0.png
│   ├── 📄 view_0_1.png
│   ├── ...
│   └── 📄 view_12_12.png
├── 📁 image_2
│   ├── 📄 view_0_0.png
│   ├── 📄 view_0_1.png
│   ├── ...
│   └── 📄 view_12_12.png
</pre>

- ▶️ Run training:
  ```bash
  python train.py --training_data_path <path_to_lightfield_data> --weights_save_path <path_to_save_weights>
  ```

- ℹ️ For help with arguments:
  ```bash
  python train.py -h
  ```

---

## 🧪 Instructions for Inference

- ✅ Setup the Python environment:
  ```bash
  pip install -r requirements.txt
  ```

- ▶️ Run inference:
  ```bash
  python test.py ----pretrained_weights <path_to_pretrained_weights> ----input_image <path_to_input_image> --output_path <output_directory>
  ```

- ℹ️ For help with arguments:
  ```bash
  python test.py -h
  ```
