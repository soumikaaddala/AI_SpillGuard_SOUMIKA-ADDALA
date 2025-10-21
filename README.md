🌊 Oil Spill Segmentation using U-Net

This project uses a U-Net deep learning model to detect and segment oil spills from satellite images.
The model is trained on the Zenodo Oil Spill Dataset
 and deployed using Streamlit for an interactive web interface.

🧠 Project Overview

Model: U-Net (built with TensorFlow / Keras)

Task: Binary image segmentation – identify oil spill regions

Dataset: Oil Spill dataset from Zenodo

Interface: Streamlit web app for image upload & mask visualization

📁 Repository Structure
├── sam.ipynb              # (Optional) Original SAM training notebook
├── oilspill_unet.ipynb    # U-Net training notebook
├── app.py                 # Streamlit app for inference
├── oilspill_unet.h5       # Saved trained model
├── requirements.txt       # Dependencies (optional)
└── README.md              # Project documentation

⚙️ Installation
1️⃣ Clone this repository
git clone https://github.com/soumikaaddala/Infosys_Ai_Oil_Spill.git
cd Infosys_Ai_Oil_Spill

2️⃣ Install dependencies
pip install streamlit tensorflow numpy opencv-python pillow matplotlib


If you also want to train the model, install:

pip install scikit-learn

🧩 Running the Streamlit App

Make sure your trained model file (oilspill_unet.h5) is in the same folder as app.py.

Then run:

streamlit run app.py


Once it starts, open the provided local URL (usually http://localhost:8501) in your browser.

🚀 How to Use

Click Browse files to upload a satellite image (JPG/PNG).

The model will predict oil spill regions.

You’ll see:

The original image

The predicted mask

The overlayed result highlighting the oil spill areas in red.

🖼️ Example Output
Original Image	Predicted Mask	Overlay

	
	

(Replace these with your actual output images if available.)

📊 Model Training Summary

Architecture: U-Net

Input size: 256×256×3

Loss: Binary Cross-Entropy

Optimizer: Adam (1e-4)

Metrics: Accuracy, Dice Coefficient

Training Data: 811 images (Train) + 203 images (Validation)

🙌 Acknowledgements

Zenodo Oil Spill Dataset

U-Net Architecture Paper

Streamlit
 for web deployment

🧾 License

This project is for educational and research purposes only.
You may modify and redistribute it with proper attribution.
