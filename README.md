#### CS 4644 – LeafCrunch: Surveying ML Methods on LeafSnap
Richard Koulen, Jade Spooner, Priya Nayak, Peter Wang

#### Abstract
Identifying dense taxonomies is an increasing area of study,
as deep learning methods improve differentiation between
very similar images. We explore this problem through tree
leaf classification with the LeafSnap dataset. We attempted
3 different deep learning methodologies used in the past
for image identification: Convolutional Neural Networks,
Autoencoders, and Vision Transformers. 

Our CNN model,DarkLeafNet, resulted in the highest top-1 (80.34%) and
top-5 (95.61%) accuracy of all of our models. DarkLeafNet
had better top-1 accuracy than the LeafSnap model in the
original paper, and a similar top-5 accuracy. Our Vision
Transformer, which used the Shifted-Window Transformer
Tiny architecture, also performed well, 
with a top-1 accuracy of 92.48% and top-5 of 99.66%.

#### Please checkout to the respective branches to see the models in more detail! 


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
