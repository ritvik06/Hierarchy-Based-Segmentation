# Hierarchy-Based-Segmentation
Python implementation of our paper ["Assessing hierarchies by their consistent segmentations"](https://hal.archives-ouvertes.fr/hal-03633805).

Authors : Zeev Gutman, Ritvik Vij, Laurent Najman, Michael Lindenbaum

## Usage

A Jupyter notebook [`src/Interactive.ipynb`](https://github.com/ritvik06/Hierarchy-Based-Segmentation/blob/main/src/Interactive.ipynb) contains the user-friendly code for running the experiments given in the paper. 

All helper scripts and implementation of the auxiliary algorithms are given in the [`src`](https://github.com/ritvik06/Hierarchy-Based-Segmentation/blob/main/src/) directory.

The data is placed in the [`data`](https://github.com/ritvik06/Hierarchy-Based-Segmentation/blob/main/data/) directory. The HED gradient images are in [`HED`](https://github.com/ritvik06/Hierarchy-Based-Segmentation/blob/main/HED/) and SLIC superpixels are stored in [`SLIC`](https://github.com/ritvik06/Hierarchy-Based-Segmentation/blob/main/SLIC/).

## Using your own data
You can easily load your custom images in the interactive notebook.

To generate your own superpixels and to generate your own gradient images, you can use the scripts in [`Helper_Scripts`](https://github.com/ritvik06/Hierarchy-Based-Segmentation/blob/main/Helper_Scripts/)

```
cd Helper_Scripts
bash run_all_slic.sh
bash run_hed_all.sh
```

## Requirements
```
numpy=1.16.4
higra=0.5.3
numba=0.51.2
scipy=1.5.2
matplotlib=3.3.2
opencv-contrib-python=4.1.2.30
```

## Cite
Please cite our paper if you use the code or ideas in your own work
```
@article{gutman22,
  author = {Zeev Gutman and Ritvik Vij and Laurent Najman and Michael Lindenbaum},
  title = {Assessing hierarchies by their consistent segmentations},
  journal = {HAL preprint HAL:hal-03633805},
  year = {2022}
}

```

## Contact
For any communication related to the code or the paper, feel free to contact me at ritvikvij06@gmail.com.

