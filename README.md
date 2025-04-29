#### ViT and Swin Implementation 
This branch contains the work and code performed for the Vision Transformer part of the project. Navigate to `ViT.ipynb` for the initial vision transformer implementation and `Swin_ViT.ipynb` for the final Swin implementation. They are both Jupyter notebooks and compatible/recommended to be with Colab. If you wish to run it locally, Python 3.12 with any PyTorch install is likely to be compatible.


#### Terminal Commands to Download Kaggle Dataset through API
1) Get kaggle.json from Kaggle.com -> Your profile -> Settings -> Create New Token <br/>
2) Upload kaggle.json you get through kaggle account settings <br/>
mkdir -p ~/.kaggle <br/>
mv kaggle.json ~/.kaggle/ <br/> 
chmod 600 ~/.kaggle/kaggle.json <br/>
kaggle datasets download -d xhlulu/leafsnap-dataset <br/>
unzip leafsnap-dataset.zip <br/>

#### Dataloader Params
root_folder -> Path to kaggle dataset <br/>
image_path -> Path to leafsnap-dataset-images.txt <br/>
use_segmented -> True/False <br/>
source -> "both", "lab", "field" <br/> 
transform -> Pass in your own transformer <br/>

#### Example Usage
root_directory = "/content/leafsnap-dataset/" <br/>
image_paths_file = root_directory + "leafsnap-dataset-images.txt" <br/>
dataset = LeafsnapDataset(image_paths_file, root_directory, use_segmented=False, transform=transform) <br/>
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) <br/>
