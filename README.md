
Yao Y, Guo P, Mao Z, et al. Multi-scale contrastive learning via aggregated subgraph for link prediction[J]. Applied Intelligence, 2025, 55(6): 1-20.

# 项目名称（如：HLP-MSM）

Yao Y, Guo P, Mao Z, et al. Multi-scale contrastive learning via aggregated subgraph for link prediction[J]. Applied Intelligence, 2025, 55(6): 1-20.


## 🧠 Abstract

高阶链路预测旨在挖掘网络中超越二元节点对的多元交互关系，识别尚未形成但潜在存在的高阶结构。在本文中，我们提出了一种基于**多尺度单纯形模体建模**的高阶链路预测方法 HLP-MSM。该方法从微观、中观与宏观三个层面提取特征信息，并利用神经网络模型进行联合学习。

具体而言，微观层面编码目标单纯形中节点的结构特征；中观层面通过统计其关联的单纯形模体数量刻画局部结构模式；宏观层面则利用目标单纯形邻域构成的子图，通过 Graph2Vec 技术生成图级嵌入表示。最终将多尺度特征融合，并利用监督学习完成高阶链接的闭包预测。

实验在多个真实网络数据集上进行，结果表明 HLP-MSM 在预测精度与鲁棒性方面均优于多种对比方法，尤其在大规模高阶结构中展现出良好的性能。此外，进一步分析显示，多尺度模体结构在增强模型表达能力方面具有显著贡献。

## 🔗 Code

点击访问代码仓库 👉 [GitHub - HLP-MSM](https://github.com/YourRepo/HLP-MSM)

## 📝 Citing

If you find **HLP-MSM** useful in your research, please consider citing the following paper:

```bibtex
@article{HLP-MSM-2025,
  title={High-order Link Prediction via Multi-scale Simplicial Motif Modeling},
  author={Your Name and Collaborator A and Collaborator B},
  journal={Journal of Complex Network Intelligence},
  pages={xx--xx},
  year={2025},
  publisher={Springer},
  doi={https://doi.org/xx.xxxx/xxxxxx},
  url={https://link.springer.com/article/xx.xxxx/xxxxxx}
}
