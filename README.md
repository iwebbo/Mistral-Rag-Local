# README : Mistral RAG Local (Safe your privacy life)
 
## Technologies 

![C++](https://img.shields.io/badge/C++-3776AB?style=for-the-badge&logo=C++&logoColor=white) ![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white) ![Yolov](https://img.shields.io/badge/Yolov-FCC624?style=for-the-badge&logo=Yolov&logoColor=black) ![CUDA](https://img.shields.io/badge/CUDA-3776AB?style=for-the-badge&logo=CUDA&logoColor=white) ![VISUALSTUDIO](https://img.shields.io/badge/VISUALSTUDIO-3776AB?style=for-the-badge&logo=VISUALSTUDIO&logoColor=white) ![OPENCV](https://img.shields.io/badge/OPENCV-3776AB?style=for-the-badge&logo=OPENCV&logoColor=white) 

## 📌 Prerequisites
### 🖥 Hardware:

A computer capable of running Python and deep learning models.

### 🛠 Run IT:

## Need to have a owner dll to control your mouse, build it with a own.cpp for example (i can't share totaly code, because it will be use for a cheating, i'm not agree with this. I love the open source, i share only my source code)

```
Build a dll before 
g++ -shared -o own.dll own.cpp -Wl,--add-stdcall-alias -m64
```
```
Cmake configure & Build
Cmake configure > CMakeLists
Launch the executable
```
#Requirements Python : 

pip install -U langchain langchain-community
pip install streamlit
pip install pdfplumber
pip install semantic-chunkers
pip install open-text-embeddings
pip install faiss
pip install ollama
pip install prompt-template
pip install langchain
pip install langchain_experimental
pip install sentence-transformers
pip install faiss-cpu
