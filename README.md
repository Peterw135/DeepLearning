#### CS 4644 – LeafCrunch: Surveying ML Methods on LeafSnap
Richard Koulen, Jade Spooner, Priya Nayak, Peter Wang

#### Abstract
Image classification is a classic problem, and architectures are soon going to have to contend with dense classification taxonomies. We explore this problem through tree leaf classification with the LeafSnap dataset. We attempted 3 different deep learning methodologies used in the past for image identification: Convolutional Neural Networks, Autoencoders, and Vision Transformers. Our CNN model, DarkLeafNet, resulted in a top-1 accuracy of 80.34\% and a top-5 accuracy of 95.61\%. The SWin ViT-Tiny architecture, which we fine-tuned on LeafSnap data, achieved a top-1 accuracy of 92.48\% and a top-5 of 99.66\%. Both of these methods outperformed the original LeafSnap approach, and we found that a dense taxonomy is not specifically problematic for these approaches, requiring at best, slightly more expressivity.

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
