<h1 align="center">MetaMorph: AI Assistance to Transform Lo-Fi Sketches to Higher Fidelities</h1>
<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://api.metamorph.designwitheve.com/docs/" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <a href="#" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <br/>
  <a href="#" target="_blank">
    <img alt="Python: 3.6" src="https://img.shields.io/badge/Python-3.6-important" />
  </a>
  <a href="#" target="_blank">
    <img alt="Dependency: Tensorflow 1.9" src="https://img.shields.io/badge/Tensorflow-1.9-important" />
  </a>
  <a href="https://python-poetry.org/" target="_blank">
    <img alt="Dependency Manager: Poetry" src="https://img.shields.io/badge/Dependency Manager-Poetry-important" />
  </a>
  <br/>
  <br/>
  <span>🏠 </span>
  <a href="https://metamorph.designwitheve.com" target="_blank">
    Homepage
  </a>
  <span>&nbsp;&nbsp;&nbsp;</span>
  <span>✨ </span>
  <a href="https://metamorph.designwitheve.com/try-it-out" target="_blank">
    Demo
  </a>
  <br/>
</p>

> MetaMorph is an AI tool to detect the constituent UI elements of low fidelity prototype sketches.

---

## Dataset

MetaMorph uses

- [UISketch](https://www.kaggle.com/vinothpandian/uisketch)
- [Syn dataset](https://www.kaggle.com/vinothpandian/syn-dataset)

---

## Setup and usage

MetaMorph API uses Python 3.6 and Tensorflow Object Detection API (TFOD).

To install and use MetaMorph API, follow the steps below

- Download the following files to the `models/` directory

  - [frozen_inference_graph.pb](https://designwitheve.com/f/frozen_inference_graph.pb)
  - [labels.json](https://designwitheve.com/f/labels.json)

- Install poetry

  ```sh
  pip install poetry
  ```

- Upgrade pip, as older version of pip causes installation issue with opencv

  ```
  poetry run pip install --upgrade pip
  ```

- Install dependencies
  ```sh
  poetry install
  ```
- Run the API
  ```sh
  poetry run python app.py
  ```

---

## Docker

MetaMorph API can be quickly deployed from docker without any installation or model download steps.

To use it, pull the image from [dockerhub at vinothpandian/metamorph](https://hub.docker.com/repository/docker/vinothpandian/metamorph)

```
docker pull vinothpandian/metamorph:latest
```

Run it with docker the following command

```sh
docker run -p 8000:8000 --name metamorph vinothpandian/metamorph:latest
```

---

## Development

To retrain or further improve MetaMorph model

### Installation

- Install Tensorflow Object Detection API using [this installation guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md).
- Copy the config file from `configs` folder or setup your own [Tensorflow Object Detection API Model with config](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

- Download the [UISketch](https://www.kaggle.com/vinothpandian/syn-dataset) dataset from Kaggle.
- Generate synthetic data

  ```sh
  poetry run python syn_datagen.py -d /path/to/uisketch -o /path/to/tfod/object_detection -l no.of.sketches
  ```

  To check more options on data generation use `poetry run python syn_datagen.py --help`

- Follow the guide from [training and evaluation guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_training_and_evaluation.md) by TFOD team.

---

## Citation

If you use MetaMorph, please use the following citation:

- V. P. S. Pandian, S. Suleri, C. Beecks, M. Jarke. **MetaMorph: AI Assistance to Transform Lo-Fi Sketches to Higher Fidelities.** _Proceedings of the 32st Australian Conference on Human-Computer-Interaction._

```bib
@inproceedings{Pandian_MetaMorph,
	title        = {MetaMorph: AI Assistance to Transform Lo-Fi Sketches to Higher Fidelities.},
	author       = {Pandian, Vinoth Pandian Sermuga and Suleri, Sarah and Beecks, Christian and Jarke, Matthias},
	year         = 2020,
	booktitle    = {Proceedings of the 32st Australian Conference on Human-Computer-Interaction},
	publisher    = {Association for Computing Machinery},
	address      = {New York, NY, USA},
	series       = {OZCHI'20},
	doi          = {10.1145/3441000.3441030},
	isbn         = {978-1-4503-8975-4/20/12},
	url          = {https://doi.org/10.1145/3441000.3441030},
	numpages     = 10,
	keywords     = {UI Element Dataset, Neural Networks, Deep Learning, Sketch Detection, Sketch Recognition, Artificial Intelligence, Prototyping}
}
```

If you use Syn or UISketch, please use the following citation:

```bib
@inproceedings{Pandin_Syn,
	title        = {Syn: Synthetic Dataset for Training UI Element Detector From Lo-Fi Sketches},
	author       = {Pandian, Vinoth Pandian Sermuga and Suleri, Sarah and Jarke, Matthias},
	year         = 2020,
	booktitle    = {Proceedings of the 25th International Conference on Intelligent User Interfaces Companion},
	location     = {Cagliari, Italy},
	publisher    = {Association for Computing Machinery},
	address      = {New York, NY, USA},
	series       = {IUI '20},
	pages        = {79–80},
	doi          = {10.1145/3379336.3381498},
	isbn         = 9781450375139,
	url          = {https://doi.org/10.1145/3379336.3381498},
	numpages     = 2,
	keywords     = {Neural Networks, Sketch Detection, Prototyping, Sketch Recognition, UI Element Dataset, Deep Learning}
}
```

---

## Authors

👤 **Vinoth Pandian**

- Website: [vinoth.info](https://vinoth.info)
- Github: [@vinothpandian](https://github.com/vinothpandian)
- LinkedIn: [@vinothpandian](https://linkedin.com/in/vinothpandian)

👤 **Sarah Suleri**

- Website: [sarahsuleri.info](https://sarahsuleri.info)
- LinkedIn: [@sarahsuleri](https://linkedin.com/in/sarahsuleri)
