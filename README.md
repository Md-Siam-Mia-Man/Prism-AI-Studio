<img src="https://github.com/Md-Siam-Mia-Code/Prism-AI-Studio/blob/main/assets/Banner.png"></img>

# ✨ Prism AI Studio ✨ <br> 🪄 Effortless AI Upscaling 🪄

Get ready to witness some AI magic! 🚀 Prism AI Studio is a super-sleek, easy-to-use web application that lets you upscale your images to stunning high resolutions using the power of **Real-ESRGAN** and optionally enhance faces with **Prism-AI-Studio**. All wrapped up in a vibrant, interactive interface built with FastAPI and vanilla JavaScript! 🎨

---

<img src="https://github.com/Md-Siam-Mia-Code/Prism-AI-Studio/blob/main/assets/Prism-AI-Studio.png"></img>

## 🌟 Key Features 🌟

*   **✨ AI-Powered Upscaling:** Leverages state-of-the-art Real-ESRGAN models to intelligently enlarge your images.
*   **😊 Optional Face Enhancement:** Integrates Prism-AI-Studio to magically restore and improve faces in your upscaled images (if Prism-AI-Studio is installed).
*   **🖥️ Sleek Web Interface:** Modern UI built with FastAPI (backend) and pure HTML/CSS/JavaScript (frontend). No heavy frameworks needed!
*   **🔮 Glassmorphism UI & Themes:** Features a cool glassmorphism design with multiple vibrant themes (Vibrant, Pastel Dream, Ocean Breeze, Sunset Glow) to suit your mood! 🎨
*   **🧠 Multiple Model Support:** Choose from various Real-ESRGAN models (General, Anime, etc.) tailored for different image types.
*   **☁️ Automatic Model Downloads:** Required AI models are downloaded automatically on first use, with fancy progress updates!
*   **📊 Real-time Progress:** Stay informed with WebSocket-powered status updates for uploads, downloads, and the enhancement process.
*   **🖱️ Drag & Drop Uploads:** Easily upload multiple images by dragging them onto the app.
*   **🖼️ Instant Previews:** See thumbnails of your input images and the enhanced results as they're processed.
*   **⚙️ Configurable Options:** Fine-tune the upscaling factor, output format (PNG, JPG, WEBP), denoise strength (for specific models), and advanced tiling parameters.
*   **💻 GPU & CPU Support:** Automatically detects NVIDIA GPUs (via CUDA) for speedy processing, but gracefully falls back to CPU if needed. Option to force CPU usage.
*   **💾 Download Options:** Download individual enhanced images or grab them all at once in a convenient ZIP archive.
*   **🗑️ Task Management:** Remove individual images or clear the entire session, with cancellation support for ongoing tasks.
*   **📱 Responsive(ish) Design:** The UI adapts reasonably well to different screen sizes.

---

## 🛠️ Technology Stack 🛠️

*   **Backend:** FastAPI, Uvicorn, WebSockets
*   **AI Models:** Real-ESRGAN, Prism-AI-Studio (Optional)
*   **Image Processing:** PyTorch, OpenCV-Python, basicsr
*   **Frontend:** HTML5, CSS3, Vanilla JavaScript
*   **Core Libraries:** `requests`, `tqdm`

---

## 🚀 Installation & Setup 🚀

Get ready to launch Prism AI Studio! Follow these steps:

## **Prerequisites:**
- 🐉 [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
- 🐍 [Python](https://www.python.org/) 3.7 or Higher
- 📦 [pip](https://pypi.org/project/pip/) (Python Package Installer)
- ♨️ [PyTorch](https://pytorch.org/) >= 1.7
- ➕ [Git](https://git-scm.com/) Installed
- ❗[NVIDIA GPU](https://www.nvidia.com/en-us/geforce/graphics-cards/) + [CUDA](https://developer.nvidia.com/cuda-downloads) (Optional)
- 🐧[Linux](https://www.linux.org/pages/download/) (Optional)

1. **Clone the Repository**
```bash
git clone https://github.com/Md-Siam-Mia-Code/Prism-AI-Studio.git
cd Prism-AI-Studio
```

2. **Create a Virtual Environment**
```bash
conda create -n Prism-AI-Studio python=3.7 -y
conda activate Prism-AI-Studio
```
3. **Install PyTorch**
 ```bash
# Try to make sure your drivers are updated
# For NVIDIA GPU
conda install pytorch torchvision torchaudio pytorch-cuda=<your_cuda_version> -c pytorch -c nvidia -y

# For CPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

4. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage
### ▶️ Running the Application
**Start the Flask Server:**
```bash
python Prism-AI-Studio.py
```

**For one-click RUN**
    Create a new Prism-AI-Studio.bat Batch File on your Prism-AI-Studio directory using the script given below:

    @echo off

    :: Activate the conda environment for Prism-AI-Studio
    CALL "C:\ProgramData\<Your Anaconda distribution name>\Scripts\activate.bat" Prism-AI-Studio

    :: Navigate to the Prism-AI-Studio directory (Change path according to yours)
    cd /D path/to/your/Prism-AI-Studio
    
    :: Run the Prism-AI-Studio web interface script
    python Prism-AI-Studio.py

**Access the Web Interface:**  
🌐 Open your browser and visit: [http://127.0.0.1:3005](http://127.0.0.1:3025)

*Note: Check the exact versions required by the specific versions of Real-ESRGAN and Prism-AI-Studio you intend to use.*

---

## ✨ How to Use ✨

Using Prism AI Studio is a breeze!

1.  **Access the App:** Open `http://127.0.0.1:3025` in your browser.
2.  **Upload Images:** Drag and drop your image files onto the designated area, or click it to browse your files. Accepted formats: PNG, JPG/JPEG, WEBP, BMP, TIFF.
3.  **Configure Settings (Sidebar):**
    *   Choose your desired **Upscaling Model**.
    *   Select the **Upscale Factor** (e.g., 2x, 4x).
    *   Pick an **Output Format** (PNG recommended for quality).
    *   Toggle **Face Enhancement** (Prism-AI-Studio) On/Off (if available).
    *   Adjust **Advanced Settings** like Tile Size if needed (especially if you encounter memory errors).
    *   Select a UI **Theme** if you like! 🌈
4.  **Model Downloads:** If you're using a model for the first time, the app will automatically download it. You'll see progress notifications! ☁️➡️💻
5.  **Enhance!** Once your images are uploaded and settings are configured, smash that **"Enhance Images"** button! ✨
6.  **Watch the Progress:** See the status updates on the image thumbnails in real-time.
7.  **Download Results:** Once completed, download buttons will appear on the enhanced image thumbnails. You can also use the **"Download All"** button to get a ZIP file of all successful enhancements. 📦

---

## 🧠 Models & Configuration 🧠

*   **Real-ESRGAN Models:** The app supports several pre-defined models, downloaded automatically to the `./weights/` directory:
    *   `RealESRGAN_x4plus` (Default, good general-purpose 4x)
    *   `RealESRNet_x4plus` (Faster alternative to x4plus)
    *   `RealESRGAN_x4plus_anime_6B` (Optimized for 4x Anime upscaling)
    *   `RealESRGAN_x2plus` (General-purpose 2x)
    *   `realesr-animevideov3` (For Anime Video frames, 4x)
    *   `realesr-general-x4v3` (General-purpose 4x with denoise option)

---

## 🎨 Frontend Themes 🎨

Spice up your upscaling experience! Choose from several themes in the top-right dropdown:

1.  **Vibrant Glassmorphism (Default):** The sleek, modern look with purple/blue gradients.
2.  **Pastel Dream:** A softer, lighter theme with pastel orange/yellow tones.
3.  **Ocean Breeze:** Cool and refreshing deep blue to cyan gradients.
4.  **Sunset Glow:** Warm pink and orange hues for a cozy feel.

---

## ❗ Troubleshooting & Notes ❗

*   **CUDA Out of Memory Errors:** If you see errors related to GPU memory during enhancement (especially with large images or high scale factors):
    *   Try reducing the **Tile Size** in the "Advanced Settings". Start with values like `256` or `128`. A value of `0` attempts to process the whole image at once.
    *   Try reducing the **Upscale Factor**.
*   **CPU Performance:** Processing on the CPU will be significantly slower than using a compatible NVIDIA GPU. Be patient!
*   **Model Downloads:** Initial model downloads require an active internet connection and can take some time depending on your network speed and the model size. Progress is shown via notifications.
*   **Single Worker Limitation:** Remember to run the server with only **one worker** (`--workers 1` if using Uvicorn directly) due to the use of shared in-memory state.

---

## 🙌 Contributing 🙌

Contributions are welcome! If you have suggestions, bug reports, or want to add features:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -am 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Create a new Pull Request.

Please report any issues using the GitHub Issues tracker.

---

## 🙏 Acknowledgements 🙏

This project heavily relies on the amazing work by the creators of:

*   **Real-ESRGAN:** <https://github.com/xinntao/Real-ESRGAN>
*   **FastAPI:** <https://fastapi.tiangolo.com/>
*   **BasicSR:** <https://github.com/xinntao/BasicSR>

Give them a star! ⭐
