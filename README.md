# LLM-trials
 Consolidation of trying out different opensource Large Language Models 

# Installing hugging face and relevant libtraries for inference

```shell
pip install "accelerate>=0.16.0,<1" "transformers[torch]>=4.28.1,<5" "torch>=1.13.1,<2" langchain
```

# Running the model in 8bit version

Step 1 - 
Download specific CUDA libraries
`wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run`
Step 2 -  install it in local folder.
`bash cuda_11.8.0_520.61.05_linux.run --no-drm --no-man-page --override --toolkitpath=/home/ubuntu/cuda-11.8/ --toolkit --silent` 
<br/>
Step 3 - Set CUDA Home
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/ubuntu/cuda-11.8/lib64" >> ~/.bashrc
echo "export PATH=\$PATH:/home/ubuntu/cuda-11.8/bin" >> ~/.bashrc
source ~/.bashrc 

Step 3 - Install bitsandbytes from source
```
cd ~
git clone https://github.com/TimDettmers/bitsandbytes
cd bitsandbytes
CUDA_HOME=/home/ubuntu/cuda-11.8 CUDA_VERSION=118 make cuda11x
pip install -e .
````

https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md