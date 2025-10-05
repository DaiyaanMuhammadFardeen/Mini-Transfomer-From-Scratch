# Mini-Transfomer-From-Scratch
Trying to create an efficient mini transformer from scratch by introducing efficiency techniques from recent 2024 and 2025 papers. I mostly don't know what I am doing most of the time. But I will attempt anyway

# 🧠 Transformer Implementation Progress Log
---

### 📅 **Day 1 — Initial Setup & Modularization**
**Date:** Sunday, October 5, 2025

#### ✅ What I Did
- Followed this [Medium article](https://medium.com/data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb) to implement a basic transformer model from scratch using pytorch's nn module.
- Copied the full code implementation and **modularized** each component into separate files for better configurability and clarity.  
- Training runs on the GPU memory

#### 🧩 Issues Faced
- Tried to modularize the code but faced import issues 
- Mask generation was being done on the CPU with RAM

#### 🛠️ How I Fixed It
- Fixed the import issues by making import corrections
- Mask matricies are now created in GPU memory in parallel

#### 💡 What I Learned
- Learned how to implement a basic transformer model from scratch using pytorch's nn module.

#### 🎯 Next Steps
- Improve the train.py code to train using GPU and add progress bars and loggers.
---
