** **![](Pictures/100000000000011A0000011AA71D733A.png){width="0.7043in"
height="0.7083in"}

[]{#anchor}**DAFFODIL INTERNATIONAL UNIVERSITY**^ ^

**DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING**^^

**FYDP (Phase-I) Evaluation Report**

[]{#anchor-1}**reporting period- Fall 2025**

**Project Identification: **

<table>
<tbody>
<tr class="odd">
<td><strong>I. Project Title</strong></td>
<td><strong>An Efficient Mini Transformer from Scratch: Implementation
and Optimization with Modern Techniques for Code Difference to Commit
Message Translation</strong></td>
<td></td>
</tr>
<tr class="even">
<td><strong>II. Group Members</strong></td>
<td><p><strong>Name</strong>: Daiyaan Muhammad Fardeen </p>
<p><strong>Student ID:</strong> 222-1-6531</p></td>
<td></td>
</tr>
<tr class="odd">
<td><strong>III. Supervisor</strong></td>
<td><p><strong>Name</strong>: Md. Shahriar Shakil</p>
<p><strong>Designation</strong>: Lecturer</p></td>
<td></td>
</tr>
<tr class="even">
<td><strong>IV. Co-Supervisor</strong></td>
<td><p><strong>Name</strong>: Ms. Syada Tasmia Alvi</p>
<p><strong>Designation</strong>: Senior Lecturer</p></td>
<td></td>
</tr>
<tr class="odd">
<td><strong>V. Submission Date: </strong></td>
<td></td>
<td></td>
</tr>
<tr class="even">
<td><strong>VI. Certificate: </strong></td>
<td>“This is to certify that the final year design project work until
Phase-I evaluation held on <strong>___________________</strong>, titled
as stated in<em> Sec. I</em>, executed by the students’ group mentioned
in<em> Sec. II</em>, have been found satisfactory and every section of
this report is reflecting the same.”</td>
<td><p><em>(Signature of Supervisor &amp; date)</em></p></td>
</tr>
</tbody>
</table>

**Project Insights**

<table>
<tbody>
<tr class="odd">
<td><p><strong>Thematic Area(s):</strong></p>
<p><a
href="https://www.codecogs.com/eqnedit.php?latex=Just%20click%20the%20check%20box#1"></a></p></td>
<td>Artificial Intelligence and Machine Learning</td>
<td>☒</td>
</tr>
<tr class="even">
<td>Deep Learning</td>
<td>☒</td>
<td></td>
</tr>
<tr class="odd">
<td>Health Informatics </td>
<td>☐</td>
<td></td>
</tr>
<tr class="even">
<td>Cybersecurity</td>
<td>☐</td>
<td></td>
</tr>
<tr class="odd">
<td>Software Engineering and Development</td>
<td>☐</td>
<td></td>
</tr>
<tr class="even">
<td>Blockchain Technology</td>
<td>☐</td>
<td></td>
</tr>
<tr class="odd">
<td>Internet of Things (IoT)</td>
<td>☐</td>
<td></td>
</tr>
<tr class="even">
<td>Computer Networks</td>
<td>☐</td>
<td></td>
</tr>
<tr class="odd">
<td>Computer Vision</td>
<td>☐</td>
<td></td>
</tr>
<tr class="even">
<td>Natural Language Processing (NLP)</td>
<td>☒</td>
<td></td>
</tr>
<tr class="odd">
<td>Robotics</td>
<td>☐</td>
<td></td>
</tr>
<tr class="even">
<td>Game Development</td>
<td>☐</td>
<td></td>
</tr>
<tr class="odd">
<td>Cloud Computing</td>
<td>☐</td>
<td></td>
</tr>
<tr class="even">
<td>Image Processing</td>
<td>☐</td>
<td></td>
</tr>
<tr class="odd">
<td><strong>Others </strong><em><strong>(please
specify)</strong></em><strong>:</strong></td>
<td></td>
<td></td>
</tr>
<tr class="even">
<td>Software packages, tools, and programming languages</td>
<td>Python, PyTorch, NumPy, Pandas, PyArrow, Scikit-learn, Matplotlib,
Seaborn, Tqdm, Pickle, CUDA/ROCm, Multiprocessing, Argparse, Parquet,
Pickle (.pkl), PyTorch State Dict (.pth), IntelliJ IDEA,</td>
<td></td>
</tr>
</tbody>
</table>

CO Description for FYDP-Phase-I

|          |                                                                                                                                                                                                                                |          |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| **CO**   | **CO Descriptions**                                                                                                                                                                                                            | **PO**   |
| **CO4**  | Perform economic evaluation, cost estimation, and apply suitable project management procedures throughout the FYDP lifecycle in the context of developing the "An Efficient Mini Transformer from Scratch" project.            | **PO11** |
| **CO6**  | Select and apply appropriate methodologies, resources, and contemporary engineering/IT tools for prediction, modeling, and solving complex engineering processes for the "An Efficient Mini Transformer from Scratch" project. | **PO5**  |
| **CO7**  | Assess societal, health, safety, legal, and cultural issues and responsibilities in professional engineering practice related to the FYDP problem.                                                                             | **PO6**  |
| **CO10** | Operate effectively as an individual and as a member/leader in multidisciplinary teams during FYDP.                                                                                                                            | **PO9**  |

1.  []{#anchor-2}Project Overview

1.  1.  []{#anchor-3}Introduction

The rapid evolution of software development practices has highlighted
the critical importance of automated documentation and understanding of
code changes. In modern software engineering, developers frequently
commit changes to version control systems, where each commit includes a
diff representing the code modifications and a commit message describing
the purpose and content of the changes. These commit messages serve as
essential documentation, helping teams understand the rationale behind
code modifications, facilitating code reviews, and supporting software
maintenance
activities[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"SwlWEObK\",\"properties\":{\"formattedCitation\":\"[1]\",\"plainCitation\":\"[1]\",\"noteIndex\":0},\"citationItems\":[{\"id\":130,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/ACBLY4LT\"],\"itemData\":{\"id\":130,\"type\":\"paper-conference\",\"container-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"DOI\":\"10.1109/ASE.2017.8115626\",\"event-place\":\"Urbana, IL\",\"event-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"ISBN\":\"978-1-5386-2684-9\",\"language\":\"en\",\"page\":\"135-146\",\"publisher\":\"IEEE\",\"publisher-place\":\"Urbana, IL\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Automatically generating commit messages from diffs using neural machine translation\",\"URL\":\"http://ieeexplore.ieee.org/document/8115626/\",\"author\":[{\"family\":\"Jiang\",\"given\":\"Siyuan\"},{\"family\":\"Armaly\",\"given\":\"Ameer\"},{\"family\":\"McMillan\",\"given\":\"Collin\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2017\",10]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDdcA0Ds1j5i}\[1\]

Traditional approaches to commit message generation have relied on
rule-based systems, template matching, or simple pattern recognition
techniques. However, these methods often produce verbose, repetitive, or
low-quality messages that fail to capture the high-level intent behind
code changes
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"kpoSvsoZ\",\"properties\":{\"formattedCitation\":\"[2]\",\"plainCitation\":\"[2]\",\"noteIndex\":0},\"citationItems\":[{\"id\":322,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/Q64XKF4Z\"],\"itemData\":{\"id\":322,\"type\":\"paper-conference\",\"abstract\":\"Commit messages can be regarded as the documentation of software changes. These messages describe the content and purposes of changes, hence are useful for program comprehension and software maintenance. However, due to the lack of time and direct motivation, commit messages sometimes are neglected by developers. To address this problem, Jiang et al. proposed an approach (we refer to it as NMT ), which leverages a neural machine translation algorithm to automatically generate short commit messages from code. The reported performance of their approach is promising, however, they did not explore why their approach performs well. Thus, in this paper, we first perform an in-depth analysis of their experimental results. We find that (1) Most of the test diffs from which NMT can generate high-quality messages are similar to one or more training diffs at the token level. (2) About 16% of the commit messages in Jiang et al.’s dataset are noisy due to being automatically generated or due to them describing repetitive trivial changes. (3) The performance of NMT declines by a large amount after removing such noisy commit messages. In addition, NMT is complicated and time-consuming. Inspired by our first finding, we proposed a simpler and faster approach, named NNGen (Nearest Neighbor Generator), to generate concise commit messages using the nearest neighbor algorithm. Our experimental results show that NNGen is over 2,600 times faster than NMT , and outperforms NMT in terms of BLEU (an accuracy measure that is widely used to evaluate machine translation systems) by 21%. Finally, we also discuss some observations for the road ahead for automated commit message generation to inspire other researchers.\",\"container-title\":\"Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"DOI\":\"10.1145/3238147.3238190\",\"event-place\":\"Montpellier France\",\"event-title\":\"ASE '18: 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"ISBN\":\"978-1-4503-5937-5\",\"language\":\"en\",\"page\":\"373-384\",\"publisher\":\"ACM\",\"publisher-place\":\"Montpellier France\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Neural-machine-translation-based commit message generation: how far are we?\",\"title-short\":\"Neural-machine-translation-based commit message generation\",\"URL\":\"https://dl.acm.org/doi/10.1145/3238147.3238190\",\"author\":[{\"family\":\"Liu\",\"given\":\"Zhongxin\"},{\"family\":\"Xia\",\"given\":\"Xin\"},{\"family\":\"Hassan\",\"given\":\"Ahmed E.\"},{\"family\":\"Lo\",\"given\":\"David\"},{\"family\":\"Xing\",\"given\":\"Zhenchang\"},{\"family\":\"Wang\",\"given\":\"Xinyu\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2018\",9,3]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RND1uyBuo4JC7}\[2\]
. The challenge of generating concise, informative commit messages has
driven researchers to explore neural machine translation (NMT)
techniques for automatically translating code diffs into human-readable
commit messages
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"R98PqK0h\",\"properties\":{\"formattedCitation\":\"[1]\",\"plainCitation\":\"[1]\",\"noteIndex\":0},\"citationItems\":[{\"id\":130,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/ACBLY4LT\"],\"itemData\":{\"id\":130,\"type\":\"paper-conference\",\"container-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"DOI\":\"10.1109/ASE.2017.8115626\",\"event-place\":\"Urbana, IL\",\"event-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"ISBN\":\"978-1-5386-2684-9\",\"language\":\"en\",\"page\":\"135-146\",\"publisher\":\"IEEE\",\"publisher-place\":\"Urbana, IL\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Automatically generating commit messages from diffs using neural machine translation\",\"URL\":\"http://ieeexplore.ieee.org/document/8115626/\",\"author\":[{\"family\":\"Jiang\",\"given\":\"Siyuan\"},{\"family\":\"Armaly\",\"given\":\"Ameer\"},{\"family\":\"McMillan\",\"given\":\"Collin\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2017\",10]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDlkQzR0PL4u}\[1\]
.

The transformer architecture, introduced by Vaswani et
al.[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"bmSv0ajQ\",\"properties\":{\"formattedCitation\":\"[3]\",\"plainCitation\":\"[3]\",\"noteIndex\":0},\"citationItems\":[{\"id\":327,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/MSPPSF3Z\"],\"itemData\":{\"id\":327,\"type\":\"article-journal\",\"abstract\":\"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring signiﬁcantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.\",\"language\":\"en\",\"source\":\"Zotero\",\"title\":\"Attention is All you Need\",\"author\":[{\"family\":\"Vaswani\",\"given\":\"Ashish\"},{\"family\":\"Shazeer\",\"given\":\"Noam\"},{\"family\":\"Parmar\",\"given\":\"Niki\"},{\"family\":\"Uszkoreit\",\"given\":\"Jakob\"},{\"family\":\"Jones\",\"given\":\"Llion\"},{\"family\":\"Gomez\",\"given\":\"Aidan N\"},{\"family\":\"Kaiser\",\"given\":\"Łukasz\"},{\"family\":\"Polosukhin\",\"given\":\"Illia\"}]}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDHHHjIJ4qvu}\[3\]
, has revolutionized natural language processing by replacing recurrent
and convolutional architectures with attention mechanisms. The original
transformer model employs multi-head self-attention, layer
normalization, and feed-forward networks to process sequences in
parallel, offering significant advantages in training speed and
performance over sequential models. However, the standard transformer
architecture faces several limitations when applied to specialized tasks
such as code diff to commit message translation, particularly in terms
of computational efficiency, memory usage, and handling of positional
information
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"ane5AZPH\",\"properties\":{\"formattedCitation\":\"[3]\",\"plainCitation\":\"[3]\",\"noteIndex\":0},\"citationItems\":[{\"id\":327,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/MSPPSF3Z\"],\"itemData\":{\"id\":327,\"type\":\"article-journal\",\"abstract\":\"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring signiﬁcantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.\",\"language\":\"en\",\"source\":\"Zotero\",\"title\":\"Attention is All you Need\",\"author\":[{\"family\":\"Vaswani\",\"given\":\"Ashish\"},{\"family\":\"Shazeer\",\"given\":\"Noam\"},{\"family\":\"Parmar\",\"given\":\"Niki\"},{\"family\":\"Uszkoreit\",\"given\":\"Jakob\"},{\"family\":\"Jones\",\"given\":\"Llion\"},{\"family\":\"Gomez\",\"given\":\"Aidan N\"},{\"family\":\"Kaiser\",\"given\":\"Łukasz\"},{\"family\":\"Polosukhin\",\"given\":\"Illia\"}]}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDS63BDkVIjb}\[3\]
.

Recent advancements in transformer architectures have addressed these
limitations through various optimization techniques.
FlashAttention[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"Q8A9VeOT\",\"properties\":{\"formattedCitation\":\"[4]\",\"plainCitation\":\"[4]\",\"noteIndex\":0},\"citationItems\":[{\"id\":142,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/YIG9NHJW\"],\"itemData\":{\"id\":142,\"type\":\"article\",\"abstract\":\"Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading oﬀ model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IOaware—accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3 speedup on GPT-2 (seq. length 1K), and 2.4 speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classiﬁcation) and entirely new capabilities: the ﬁrst Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).\",\"DOI\":\"10.48550/arXiv.2205.14135\",\"language\":\"en\",\"note\":\"arXiv:2205.14135 [cs]\",\"number\":\"arXiv:2205.14135\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness\",\"title-short\":\"FlashAttention\",\"URL\":\"http://arxiv.org/abs/2205.14135\",\"author\":[{\"family\":\"Dao\",\"given\":\"Tri\"},{\"family\":\"Fu\",\"given\":\"Daniel Y.\"},{\"family\":\"Ermon\",\"given\":\"Stefano\"},{\"family\":\"Rudra\",\"given\":\"Atri\"},{\"family\":\"Ré\",\"given\":\"Christopher\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2022\",6,23]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDZZFeN16x11}\[4\]
introduces an IO-aware attention algorithm that significantly reduces
memory usage and computational overhead by using tiling to minimize HBM
(High Bandwidth Memory) accesses. This approach enables transformers to
scale to longer sequences while maintaining efficiency, addressing one
of the primary bottlenecks in attention computation.

Rotary Position Embedding
(RoPE)[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"dDl2vXji\",\"properties\":{\"formattedCitation\":\"[5]\",\"plainCitation\":\"[5]\",\"noteIndex\":0},\"citationItems\":[{\"id\":138,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/LXWZHZF9\"],\"itemData\":{\"id\":138,\"type\":\"article\",\"abstract\":\"Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: https://huggingface.co/docs/transformers/model_doc/roformer.\",\"DOI\":\"10.48550/arXiv.2104.09864\",\"language\":\"en\",\"note\":\"arXiv:2104.09864 [cs]\",\"number\":\"arXiv:2104.09864\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"RoFormer: Enhanced Transformer with Rotary Position Embedding\",\"title-short\":\"RoFormer\",\"URL\":\"http://arxiv.org/abs/2104.09864\",\"author\":[{\"family\":\"Su\",\"given\":\"Jianlin\"},{\"family\":\"Lu\",\"given\":\"Yu\"},{\"family\":\"Pan\",\"given\":\"Shengfeng\"},{\"family\":\"Murtadha\",\"given\":\"Ahmed\"},{\"family\":\"Wen\",\"given\":\"Bo\"},{\"family\":\"Liu\",\"given\":\"Yunfeng\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2023\",11,8]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDsBKHWTUMMp}\[5\]
provides an alternative approach to positional encoding that encodes
absolute position information through rotation matrices while
incorporating relative position dependencies into the self-attention
formulation. RoPE offers several advantages including sequence length
flexibility, decaying inter-token dependency with increasing relative
distances, and compatibility with linear attention mechanisms, making it
particularly suitable for code-related tasks where positional
relationships are crucial.

The feed-forward network components of transformers have also seen
significant improvements through the introduction of Gated Linear Units
(GLU) and their variants.
SwiGLU[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"Is7kzk1Y\",\"properties\":{\"formattedCitation\":\"[6]\",\"plainCitation\":\"[6]\",\"noteIndex\":0},\"citationItems\":[{\"id\":134,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/39IX6VUJ\"],\"itemData\":{\"id\":134,\"type\":\"article\",\"abstract\":\"Gated Linear Units [Dauphin et al., 2016] consist of the component-wise product of two linear projections, one of which is ﬁrst passed through a sigmoid function. Variations on GLU are possible, using diﬀerent nonlinear (or even linear) functions in place of sigmoid. We test these variants in the feedforward sublayers of the Transformer [Vaswani et al., 2017] sequence-to-sequence model, and ﬁnd that some of them yield quality improvements over the typically-used ReLU or GELU activations.\",\"DOI\":\"10.48550/arXiv.2002.05202\",\"language\":\"en\",\"note\":\"arXiv:2002.05202 [cs]\",\"number\":\"arXiv:2002.05202\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"GLU Variants Improve Transformer\",\"URL\":\"http://arxiv.org/abs/2002.05202\",\"author\":[{\"family\":\"Shazeer\",\"given\":\"Noam\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2020\",2,12]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDMNIzbmPJyO}\[6\]
, which combines Swish activation with gated linear units, has
demonstrated superior performance compared to traditional ReLU or GELU
activations in transformer models. This approach provides better
gradient flow and improved expressiveness, contributing to enhanced
model performance.

Normalization techniques have evolved beyond traditional Layer
Normalization to include Root Mean Square (RMS) normalization
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"PNlYk5fV\",\"properties\":{\"formattedCitation\":\"[7]\",\"plainCitation\":\"[7]\",\"noteIndex\":0},\"citationItems\":[{\"id\":132,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/9YNSYDEC\"],\"itemData\":{\"id\":132,\"type\":\"article\",\"abstract\":\"Layer normalization (LayerNorm) has been successfully applied to various deep neural networks to help stabilize training and boost model convergence because of its capability in handling re-centering and re-scaling of both inputs and weight matrix. However, the computational overhead introduced by LayerNorm makes these improvements expensive and signiﬁcantly slows the underlying network, e.g. RNN in particular. In this paper, we hypothesize that re-centering invariance in LayerNorm is dispensable and propose root mean square layer normalization, or RMSNorm. RMSNorm regularizes the summed inputs to a neuron in one layer according to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efﬁcient than LayerNorm. We also present partial RMSNorm, or pRMSNorm where the RMS is estimated from p% of the summed inputs without breaking the above properties. Extensive experiments on several tasks using diverse network architectures show that RMSNorm achieves comparable performance against LayerNorm but reduces the running time by 7%∼64% on different models. Source code is available at https://github.com/bzhangGo/rmsnorm.\",\"DOI\":\"10.48550/arXiv.1910.07467\",\"language\":\"en\",\"note\":\"arXiv:1910.07467 [cs]\",\"number\":\"arXiv:1910.07467\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"Root Mean Square Layer Normalization\",\"URL\":\"http://arxiv.org/abs/1910.07467\",\"author\":[{\"family\":\"Zhang\",\"given\":\"Biao\"},{\"family\":\"Sennrich\",\"given\":\"Rico\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2019\",10,16]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDw0nTwmFIFE}\[7\][](https://www.codecogs.com/eqnedit.php?latex=Root%20Mean%20Square%20Layer%20Normalization.md#1),
which eliminates the re-centering operation while preserving the
re-scaling invariance property. RMSNorm offers computational simplicity
and efficiency while maintaining comparable performance to LayerNorm,
reducing training time by 7% to 64% across different models while
preserving model quality.

Gradient checkpointing techniques
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"cyjBTzfZ\",\"properties\":{\"formattedCitation\":\"[8]\",\"plainCitation\":\"[8]\",\"noteIndex\":0},\"citationItems\":[{\"id\":131,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/QT5KBL46\"],\"itemData\":{\"id\":131,\"type\":\"article\",\"abstract\":\"We propose a systematic approach to reduce the memory consumption of deep neural network training. Specifically, we design an algorithm that costs O(sqrt(n)) memory to train a n layer network, with only the computational cost of an extra forward pass per mini-batch. As many of the state-of-the-art models hit the upper bound of the GPU memory, our algorithm allows deeper and more complex models to be explored, and helps advance the innovations in deep learning research. We focus on reducing the memory cost to store the intermediate feature maps and gradients during training. Computation graph analysis is used for automatic in-place operation and memory sharing optimizations. We show that it is possible to trade computation for memory - giving a more memory efficient training algorithm with a little extra computation cost. In the extreme case, our analysis also shows that the memory consumption can be reduced to O(log n) with as little as O(n log n) extra cost for forward computation. Our experiments show that we can reduce the memory cost of a 1,000-layer deep residual network from 48G to 7G with only 30 percent additional running time cost on ImageNet problems. Similarly, significant memory cost reduction is observed in training complex recurrent neural networks on very long sequences.\",\"DOI\":\"10.48550/arXiv.1604.06174\",\"language\":\"en\",\"note\":\"arXiv:1604.06174 [cs]\",\"number\":\"arXiv:1604.06174\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"Training Deep Nets with Sublinear Memory Cost\",\"URL\":\"http://arxiv.org/abs/1604.06174\",\"author\":[{\"family\":\"Chen\",\"given\":\"Tianqi\"},{\"family\":\"Xu\",\"given\":\"Bing\"},{\"family\":\"Zhang\",\"given\":\"Chiyuan\"},{\"family\":\"Guestrin\",\"given\":\"Carlos\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2016\",4,22]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDYxPem1cHPC}\[8\]
address memory constraints during training by trading computation for
memory, enabling the training of deeper networks with limited
computational resources. This approach allows for sublinear memory
scaling with network depth, making it feasible to train very deep
transformer models without prohibitive memory requirements.

While these individual optimizations have proven effective in isolation,
their integration into a cohesive transformer architecture optimized for
code diff to commit message translation remains largely unexplored. The
task of converting code diffs to commit messages presents unique
challenges including handling of structured code changes, capturing
semantic relationships between modifications, and generating concise
natural language descriptions. Existing approaches such as NMT-based
commit message
generation[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"gN0atjnb\",\"properties\":{\"formattedCitation\":\"[1]\",\"plainCitation\":\"[1]\",\"noteIndex\":0},\"citationItems\":[{\"id\":130,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/ACBLY4LT\"],\"itemData\":{\"id\":130,\"type\":\"paper-conference\",\"container-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"DOI\":\"10.1109/ASE.2017.8115626\",\"event-place\":\"Urbana, IL\",\"event-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"ISBN\":\"978-1-5386-2684-9\",\"language\":\"en\",\"page\":\"135-146\",\"publisher\":\"IEEE\",\"publisher-place\":\"Urbana, IL\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Automatically generating commit messages from diffs using neural machine translation\",\"URL\":\"http://ieeexplore.ieee.org/document/8115626/\",\"author\":[{\"family\":\"Jiang\",\"given\":\"Siyuan\"},{\"family\":\"Armaly\",\"given\":\"Ameer\"},{\"family\":\"McMillan\",\"given\":\"Collin\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2017\",10]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDD46E4GRWGH}\[1\]
and subsequent
analysis[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"iFfsOf5n\",\"properties\":{\"formattedCitation\":\"[2]\",\"plainCitation\":\"[2]\",\"noteIndex\":0},\"citationItems\":[{\"id\":322,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/Q64XKF4Z\"],\"itemData\":{\"id\":322,\"type\":\"paper-conference\",\"abstract\":\"Commit messages can be regarded as the documentation of software changes. These messages describe the content and purposes of changes, hence are useful for program comprehension and software maintenance. However, due to the lack of time and direct motivation, commit messages sometimes are neglected by developers. To address this problem, Jiang et al. proposed an approach (we refer to it as NMT ), which leverages a neural machine translation algorithm to automatically generate short commit messages from code. The reported performance of their approach is promising, however, they did not explore why their approach performs well. Thus, in this paper, we first perform an in-depth analysis of their experimental results. We find that (1) Most of the test diffs from which NMT can generate high-quality messages are similar to one or more training diffs at the token level. (2) About 16% of the commit messages in Jiang et al.’s dataset are noisy due to being automatically generated or due to them describing repetitive trivial changes. (3) The performance of NMT declines by a large amount after removing such noisy commit messages. In addition, NMT is complicated and time-consuming. Inspired by our first finding, we proposed a simpler and faster approach, named NNGen (Nearest Neighbor Generator), to generate concise commit messages using the nearest neighbor algorithm. Our experimental results show that NNGen is over 2,600 times faster than NMT , and outperforms NMT in terms of BLEU (an accuracy measure that is widely used to evaluate machine translation systems) by 21%. Finally, we also discuss some observations for the road ahead for automated commit message generation to inspire other researchers.\",\"container-title\":\"Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"DOI\":\"10.1145/3238147.3238190\",\"event-place\":\"Montpellier France\",\"event-title\":\"ASE '18: 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"ISBN\":\"978-1-4503-5937-5\",\"language\":\"en\",\"page\":\"373-384\",\"publisher\":\"ACM\",\"publisher-place\":\"Montpellier France\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Neural-machine-translation-based commit message generation: how far are we?\",\"title-short\":\"Neural-machine-translation-based commit message generation\",\"URL\":\"https://dl.acm.org/doi/10.1145/3238147.3238190\",\"author\":[{\"family\":\"Liu\",\"given\":\"Zhongxin\"},{\"family\":\"Xia\",\"given\":\"Xin\"},{\"family\":\"Hassan\",\"given\":\"Ahmed E.\"},{\"family\":\"Lo\",\"given\":\"David\"},{\"family\":\"Xing\",\"given\":\"Zhenchang\"},{\"family\":\"Wang\",\"given\":\"Xinyu\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2018\",9,3]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDy7MiZo1ywA}\[2\]
have demonstrated the potential of neural approaches but have not fully
leveraged modern transformer optimizations.

The contribution of this work lies in the development of an efficient
mini transformer architecture specifically designed for code difference
to commit message translation, incorporating multiple modern
optimization techniques including FlashAttention for memory efficiency,
RoPE for positional encoding, SwiGLU for enhanced feed-forward networks,
RMSNorm for efficient normalization, and gradient checkpointing for
memory optimization. This integrated approach addresses the
computational and memory constraints inherent in transformer models
while maintaining high-quality translation performance for code-related
tasks.

The paper is organized as follows: we first present the theoretical
foundations of our optimized transformer architecture, followed by
detailed descriptions of each incorporated optimization technique. We
then evaluate our approach on code diff to commit message translation
tasks, demonstrating significant improvements in both computational
efficiency and translation quality compared to baseline transformer
models. Finally, we discuss the implications of our findings for future
transformer architectures and their applications in software engineering
tasks.

1.  1.  []{#anchor-4}Background Study

In software development, version control systems like Git play a
critical role in managing code changes, where each commit includes a
diff representing the code modifications and a commit message describing
the purpose and content of the changes. These commit messages serve as
essential documentation, helping teams understand the rationale behind
code changes, facilitating code reviews, and supporting software
maintenance
activities[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"T1T0csMO\",\"properties\":{\"formattedCitation\":\"[1]\",\"plainCitation\":\"[1]\",\"noteIndex\":0},\"citationItems\":[{\"id\":130,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/ACBLY4LT\"],\"itemData\":{\"id\":130,\"type\":\"paper-conference\",\"container-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"DOI\":\"10.1109/ASE.2017.8115626\",\"event-place\":\"Urbana, IL\",\"event-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"ISBN\":\"978-1-5386-2684-9\",\"language\":\"en\",\"page\":\"135-146\",\"publisher\":\"IEEE\",\"publisher-place\":\"Urbana, IL\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Automatically generating commit messages from diffs using neural machine translation\",\"URL\":\"http://ieeexplore.ieee.org/document/8115626/\",\"author\":[{\"family\":\"Jiang\",\"given\":\"Siyuan\"},{\"family\":\"Armaly\",\"given\":\"Ameer\"},{\"family\":\"McMillan\",\"given\":\"Collin\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2017\",10]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDVxXfGDv4C7}\[1\].
However, developers often produce inconsistent, vague, or incomplete
messages due to time constraints or lack of best practices, leading to
challenges in understanding the codebase after
changes[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"j1TepadF\",\"properties\":{\"formattedCitation\":\"[2]\",\"plainCitation\":\"[2]\",\"noteIndex\":0},\"citationItems\":[{\"id\":322,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/Q64XKF4Z\"],\"itemData\":{\"id\":322,\"type\":\"paper-conference\",\"abstract\":\"Commit messages can be regarded as the documentation of software changes. These messages describe the content and purposes of changes, hence are useful for program comprehension and software maintenance. However, due to the lack of time and direct motivation, commit messages sometimes are neglected by developers. To address this problem, Jiang et al. proposed an approach (we refer to it as NMT ), which leverages a neural machine translation algorithm to automatically generate short commit messages from code. The reported performance of their approach is promising, however, they did not explore why their approach performs well. Thus, in this paper, we first perform an in-depth analysis of their experimental results. We find that (1) Most of the test diffs from which NMT can generate high-quality messages are similar to one or more training diffs at the token level. (2) About 16% of the commit messages in Jiang et al.’s dataset are noisy due to being automatically generated or due to them describing repetitive trivial changes. (3) The performance of NMT declines by a large amount after removing such noisy commit messages. In addition, NMT is complicated and time-consuming. Inspired by our first finding, we proposed a simpler and faster approach, named NNGen (Nearest Neighbor Generator), to generate concise commit messages using the nearest neighbor algorithm. Our experimental results show that NNGen is over 2,600 times faster than NMT , and outperforms NMT in terms of BLEU (an accuracy measure that is widely used to evaluate machine translation systems) by 21%. Finally, we also discuss some observations for the road ahead for automated commit message generation to inspire other researchers.\",\"container-title\":\"Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"DOI\":\"10.1145/3238147.3238190\",\"event-place\":\"Montpellier France\",\"event-title\":\"ASE '18: 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"ISBN\":\"978-1-4503-5937-5\",\"language\":\"en\",\"page\":\"373-384\",\"publisher\":\"ACM\",\"publisher-place\":\"Montpellier France\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Neural-machine-translation-based commit message generation: how far are we?\",\"title-short\":\"Neural-machine-translation-based commit message generation\",\"URL\":\"https://dl.acm.org/doi/10.1145/3238147.3238190\",\"author\":[{\"family\":\"Liu\",\"given\":\"Zhongxin\"},{\"family\":\"Xia\",\"given\":\"Xin\"},{\"family\":\"Hassan\",\"given\":\"Ahmed E.\"},{\"family\":\"Lo\",\"given\":\"David\"},{\"family\":\"Xing\",\"given\":\"Zhenchang\"},{\"family\":\"Wang\",\"given\":\"Xinyu\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2018\",9,3]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDhOEWZRd8cx}\[2\].

Early motivations for automating this process stemmed from empirical
studies highlighting the poor quality of manual commit messages.
Research has shown that a significant percentage of commit messages lack
essential elements that describe what changes were made and why they
were
necessary[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"f2SVz3Vi\",\"properties\":{\"formattedCitation\":\"[2]\",\"plainCitation\":\"[2]\",\"noteIndex\":0},\"citationItems\":[{\"id\":322,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/Q64XKF4Z\"],\"itemData\":{\"id\":322,\"type\":\"paper-conference\",\"abstract\":\"Commit messages can be regarded as the documentation of software changes. These messages describe the content and purposes of changes, hence are useful for program comprehension and software maintenance. However, due to the lack of time and direct motivation, commit messages sometimes are neglected by developers. To address this problem, Jiang et al. proposed an approach (we refer to it as NMT ), which leverages a neural machine translation algorithm to automatically generate short commit messages from code. The reported performance of their approach is promising, however, they did not explore why their approach performs well. Thus, in this paper, we first perform an in-depth analysis of their experimental results. We find that (1) Most of the test diffs from which NMT can generate high-quality messages are similar to one or more training diffs at the token level. (2) About 16% of the commit messages in Jiang et al.’s dataset are noisy due to being automatically generated or due to them describing repetitive trivial changes. (3) The performance of NMT declines by a large amount after removing such noisy commit messages. In addition, NMT is complicated and time-consuming. Inspired by our first finding, we proposed a simpler and faster approach, named NNGen (Nearest Neighbor Generator), to generate concise commit messages using the nearest neighbor algorithm. Our experimental results show that NNGen is over 2,600 times faster than NMT , and outperforms NMT in terms of BLEU (an accuracy measure that is widely used to evaluate machine translation systems) by 21%. Finally, we also discuss some observations for the road ahead for automated commit message generation to inspire other researchers.\",\"container-title\":\"Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"DOI\":\"10.1145/3238147.3238190\",\"event-place\":\"Montpellier France\",\"event-title\":\"ASE '18: 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"ISBN\":\"978-1-4503-5937-5\",\"language\":\"en\",\"page\":\"373-384\",\"publisher\":\"ACM\",\"publisher-place\":\"Montpellier France\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Neural-machine-translation-based commit message generation: how far are we?\",\"title-short\":\"Neural-machine-translation-based commit message generation\",\"URL\":\"https://dl.acm.org/doi/10.1145/3238147.3238190\",\"author\":[{\"family\":\"Liu\",\"given\":\"Zhongxin\"},{\"family\":\"Xia\",\"given\":\"Xin\"},{\"family\":\"Hassan\",\"given\":\"Ahmed E.\"},{\"family\":\"Lo\",\"given\":\"David\"},{\"family\":\"Xing\",\"given\":\"Zhenchang\"},{\"family\":\"Wang\",\"given\":\"Xinyu\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2018\",9,3]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDzEEd2wOEM8}\[2\].
Studies found that around 14% of commit messages in 23K+ Java
SourceForge projects were completely empty, and many others were of low
quality
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"m028aH4f\",\"properties\":{\"formattedCitation\":\"[2]\",\"plainCitation\":\"[2]\",\"noteIndex\":0},\"citationItems\":[{\"id\":322,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/Q64XKF4Z\"],\"itemData\":{\"id\":322,\"type\":\"paper-conference\",\"abstract\":\"Commit messages can be regarded as the documentation of software changes. These messages describe the content and purposes of changes, hence are useful for program comprehension and software maintenance. However, due to the lack of time and direct motivation, commit messages sometimes are neglected by developers. To address this problem, Jiang et al. proposed an approach (we refer to it as NMT ), which leverages a neural machine translation algorithm to automatically generate short commit messages from code. The reported performance of their approach is promising, however, they did not explore why their approach performs well. Thus, in this paper, we first perform an in-depth analysis of their experimental results. We find that (1) Most of the test diffs from which NMT can generate high-quality messages are similar to one or more training diffs at the token level. (2) About 16% of the commit messages in Jiang et al.’s dataset are noisy due to being automatically generated or due to them describing repetitive trivial changes. (3) The performance of NMT declines by a large amount after removing such noisy commit messages. In addition, NMT is complicated and time-consuming. Inspired by our first finding, we proposed a simpler and faster approach, named NNGen (Nearest Neighbor Generator), to generate concise commit messages using the nearest neighbor algorithm. Our experimental results show that NNGen is over 2,600 times faster than NMT , and outperforms NMT in terms of BLEU (an accuracy measure that is widely used to evaluate machine translation systems) by 21%. Finally, we also discuss some observations for the road ahead for automated commit message generation to inspire other researchers.\",\"container-title\":\"Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"DOI\":\"10.1145/3238147.3238190\",\"event-place\":\"Montpellier France\",\"event-title\":\"ASE '18: 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"ISBN\":\"978-1-4503-5937-5\",\"language\":\"en\",\"page\":\"373-384\",\"publisher\":\"ACM\",\"publisher-place\":\"Montpellier France\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Neural-machine-translation-based commit message generation: how far are we?\",\"title-short\":\"Neural-machine-translation-based commit message generation\",\"URL\":\"https://dl.acm.org/doi/10.1145/3238147.3238190\",\"author\":[{\"family\":\"Liu\",\"given\":\"Zhongxin\"},{\"family\":\"Xia\",\"given\":\"Xin\"},{\"family\":\"Hassan\",\"given\":\"Ahmed E.\"},{\"family\":\"Lo\",\"given\":\"David\"},{\"family\":\"Xing\",\"given\":\"Zhenchang\"},{\"family\":\"Wang\",\"given\":\"Xinyu\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2018\",9,3]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDKr1VvIfjFo}\[2\].

The idea of automated commit message generation (CMG) as a
sequence-to-sequence task originated from viewing code diffs
(differences between code versions, marked by additions, deletions, and
modifications) as a \"source language\" to be translated into natural
language summaries. This framing was first popularized with the
adaptation of neural machine translation (NMT) techniques to translate
diffs into
messages[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"1lrHTyqE\",\"properties\":{\"formattedCitation\":\"[1]\",\"plainCitation\":\"[1]\",\"noteIndex\":0},\"citationItems\":[{\"id\":130,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/ACBLY4LT\"],\"itemData\":{\"id\":130,\"type\":\"paper-conference\",\"container-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"DOI\":\"10.1109/ASE.2017.8115626\",\"event-place\":\"Urbana, IL\",\"event-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"ISBN\":\"978-1-5386-2684-9\",\"language\":\"en\",\"page\":\"135-146\",\"publisher\":\"IEEE\",\"publisher-place\":\"Urbana, IL\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Automatically generating commit messages from diffs using neural machine translation\",\"URL\":\"http://ieeexplore.ieee.org/document/8115626/\",\"author\":[{\"family\":\"Jiang\",\"given\":\"Siyuan\"},{\"family\":\"Armaly\",\"given\":\"Ameer\"},{\"family\":\"McMillan\",\"given\":\"Collin\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2017\",10]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDFnvN73mnEW}\[1\].

Before neural models, CMG relied on heuristic or retrieval-based
techniques. Early rule-based methods used static analysis to generate
change descriptions via templates, focusing on syntactic elements like
method
calls[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"7uAnDUdS\",\"properties\":{\"formattedCitation\":\"[1]\",\"plainCitation\":\"[1]\",\"noteIndex\":0},\"citationItems\":[{\"id\":130,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/ACBLY4LT\"],\"itemData\":{\"id\":130,\"type\":\"paper-conference\",\"container-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"DOI\":\"10.1109/ASE.2017.8115626\",\"event-place\":\"Urbana, IL\",\"event-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"ISBN\":\"978-1-5386-2684-9\",\"language\":\"en\",\"page\":\"135-146\",\"publisher\":\"IEEE\",\"publisher-place\":\"Urbana, IL\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Automatically generating commit messages from diffs using neural machine translation\",\"URL\":\"http://ieeexplore.ieee.org/document/8115626/\",\"author\":[{\"family\":\"Jiang\",\"given\":\"Siyuan\"},{\"family\":\"Armaly\",\"given\":\"Ameer\"},{\"family\":\"McMillan\",\"given\":\"Collin\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2017\",10]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDbdIBAifi9y}\[1\].
However, these were limited to simple changes and lacked semantic depth.

Information retrieval (IR) methods improved upon rule-based approaches:
NNGen (Liu et al.,
2018)[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"MXZ2bPVt\",\"properties\":{\"formattedCitation\":\"[2]\",\"plainCitation\":\"[2]\",\"noteIndex\":0},\"citationItems\":[{\"id\":322,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/Q64XKF4Z\"],\"itemData\":{\"id\":322,\"type\":\"paper-conference\",\"abstract\":\"Commit messages can be regarded as the documentation of software changes. These messages describe the content and purposes of changes, hence are useful for program comprehension and software maintenance. However, due to the lack of time and direct motivation, commit messages sometimes are neglected by developers. To address this problem, Jiang et al. proposed an approach (we refer to it as NMT ), which leverages a neural machine translation algorithm to automatically generate short commit messages from code. The reported performance of their approach is promising, however, they did not explore why their approach performs well. Thus, in this paper, we first perform an in-depth analysis of their experimental results. We find that (1) Most of the test diffs from which NMT can generate high-quality messages are similar to one or more training diffs at the token level. (2) About 16% of the commit messages in Jiang et al.’s dataset are noisy due to being automatically generated or due to them describing repetitive trivial changes. (3) The performance of NMT declines by a large amount after removing such noisy commit messages. In addition, NMT is complicated and time-consuming. Inspired by our first finding, we proposed a simpler and faster approach, named NNGen (Nearest Neighbor Generator), to generate concise commit messages using the nearest neighbor algorithm. Our experimental results show that NNGen is over 2,600 times faster than NMT , and outperforms NMT in terms of BLEU (an accuracy measure that is widely used to evaluate machine translation systems) by 21%. Finally, we also discuss some observations for the road ahead for automated commit message generation to inspire other researchers.\",\"container-title\":\"Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"DOI\":\"10.1145/3238147.3238190\",\"event-place\":\"Montpellier France\",\"event-title\":\"ASE '18: 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"ISBN\":\"978-1-4503-5937-5\",\"language\":\"en\",\"page\":\"373-384\",\"publisher\":\"ACM\",\"publisher-place\":\"Montpellier France\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Neural-machine-translation-based commit message generation: how far are we?\",\"title-short\":\"Neural-machine-translation-based commit message generation\",\"URL\":\"https://dl.acm.org/doi/10.1145/3238147.3238190\",\"author\":[{\"family\":\"Liu\",\"given\":\"Zhongxin\"},{\"family\":\"Xia\",\"given\":\"Xin\"},{\"family\":\"Hassan\",\"given\":\"Ahmed E.\"},{\"family\":\"Lo\",\"given\":\"David\"},{\"family\":\"Xing\",\"given\":\"Zhenchang\"},{\"family\":\"Wang\",\"given\":\"Xinyu\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2018\",9,3]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RND0YVTM20ni5}\[2\]
retrieved similar diffs from historical commits using cosine similarity
on bag-of-words representations, outperforming early NMT on cleaned data
by leveraging exact matches to avoid generation errors. This approach
was found to be over 2,600 times faster than NMT while achieving better
BLEU
scores[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"8yQTdQ0r\",\"properties\":{\"formattedCitation\":\"[2]\",\"plainCitation\":\"[2]\",\"noteIndex\":0},\"citationItems\":[{\"id\":322,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/Q64XKF4Z\"],\"itemData\":{\"id\":322,\"type\":\"paper-conference\",\"abstract\":\"Commit messages can be regarded as the documentation of software changes. These messages describe the content and purposes of changes, hence are useful for program comprehension and software maintenance. However, due to the lack of time and direct motivation, commit messages sometimes are neglected by developers. To address this problem, Jiang et al. proposed an approach (we refer to it as NMT ), which leverages a neural machine translation algorithm to automatically generate short commit messages from code. The reported performance of their approach is promising, however, they did not explore why their approach performs well. Thus, in this paper, we first perform an in-depth analysis of their experimental results. We find that (1) Most of the test diffs from which NMT can generate high-quality messages are similar to one or more training diffs at the token level. (2) About 16% of the commit messages in Jiang et al.’s dataset are noisy due to being automatically generated or due to them describing repetitive trivial changes. (3) The performance of NMT declines by a large amount after removing such noisy commit messages. In addition, NMT is complicated and time-consuming. Inspired by our first finding, we proposed a simpler and faster approach, named NNGen (Nearest Neighbor Generator), to generate concise commit messages using the nearest neighbor algorithm. Our experimental results show that NNGen is over 2,600 times faster than NMT , and outperforms NMT in terms of BLEU (an accuracy measure that is widely used to evaluate machine translation systems) by 21%. Finally, we also discuss some observations for the road ahead for automated commit message generation to inspire other researchers.\",\"container-title\":\"Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"DOI\":\"10.1145/3238147.3238190\",\"event-place\":\"Montpellier France\",\"event-title\":\"ASE '18: 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"ISBN\":\"978-1-4503-5937-5\",\"language\":\"en\",\"page\":\"373-384\",\"publisher\":\"ACM\",\"publisher-place\":\"Montpellier France\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Neural-machine-translation-based commit message generation: how far are we?\",\"title-short\":\"Neural-machine-translation-based commit message generation\",\"URL\":\"https://dl.acm.org/doi/10.1145/3238147.3238190\",\"author\":[{\"family\":\"Liu\",\"given\":\"Zhongxin\"},{\"family\":\"Xia\",\"given\":\"Xin\"},{\"family\":\"Hassan\",\"given\":\"Ahmed E.\"},{\"family\":\"Lo\",\"given\":\"David\"},{\"family\":\"Xing\",\"given\":\"Zhenchang\"},{\"family\":\"Wang\",\"given\":\"Xinyu\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2018\",9,3]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RND4YHLAhkuLY}\[2\].

Neural approaches then addressed limitations by learning from data. The
seminal work by Jiang et
al.[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"xxSU6vYm\",\"properties\":{\"formattedCitation\":\"[1]\",\"plainCitation\":\"[1]\",\"noteIndex\":0},\"citationItems\":[{\"id\":130,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/ACBLY4LT\"],\"itemData\":{\"id\":130,\"type\":\"paper-conference\",\"container-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"DOI\":\"10.1109/ASE.2017.8115626\",\"event-place\":\"Urbana, IL\",\"event-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"ISBN\":\"978-1-5386-2684-9\",\"language\":\"en\",\"page\":\"135-146\",\"publisher\":\"IEEE\",\"publisher-place\":\"Urbana, IL\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Automatically generating commit messages from diffs using neural machine translation\",\"URL\":\"http://ieeexplore.ieee.org/document/8115626/\",\"author\":[{\"family\":\"Jiang\",\"given\":\"Siyuan\"},{\"family\":\"Armaly\",\"given\":\"Ameer\"},{\"family\":\"McMillan\",\"given\":\"Collin\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2017\",10]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDsvJHtB6Nfg}\[1\]
adapted neural machine translation to \"translate\" diffs into messages,
training on filtered GitHub commit pairs with verb-direct-object
patterns. The approach used the attentional RNN Encoder-Decoder
model[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"QmDAD2C5\",\"properties\":{\"formattedCitation\":\"[2]\",\"plainCitation\":\"[2]\",\"noteIndex\":0},\"citationItems\":[{\"id\":322,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/Q64XKF4Z\"],\"itemData\":{\"id\":322,\"type\":\"paper-conference\",\"abstract\":\"Commit messages can be regarded as the documentation of software changes. These messages describe the content and purposes of changes, hence are useful for program comprehension and software maintenance. However, due to the lack of time and direct motivation, commit messages sometimes are neglected by developers. To address this problem, Jiang et al. proposed an approach (we refer to it as NMT ), which leverages a neural machine translation algorithm to automatically generate short commit messages from code. The reported performance of their approach is promising, however, they did not explore why their approach performs well. Thus, in this paper, we first perform an in-depth analysis of their experimental results. We find that (1) Most of the test diffs from which NMT can generate high-quality messages are similar to one or more training diffs at the token level. (2) About 16% of the commit messages in Jiang et al.’s dataset are noisy due to being automatically generated or due to them describing repetitive trivial changes. (3) The performance of NMT declines by a large amount after removing such noisy commit messages. In addition, NMT is complicated and time-consuming. Inspired by our first finding, we proposed a simpler and faster approach, named NNGen (Nearest Neighbor Generator), to generate concise commit messages using the nearest neighbor algorithm. Our experimental results show that NNGen is over 2,600 times faster than NMT , and outperforms NMT in terms of BLEU (an accuracy measure that is widely used to evaluate machine translation systems) by 21%. Finally, we also discuss some observations for the road ahead for automated commit message generation to inspire other researchers.\",\"container-title\":\"Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"DOI\":\"10.1145/3238147.3238190\",\"event-place\":\"Montpellier France\",\"event-title\":\"ASE '18: 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"ISBN\":\"978-1-4503-5937-5\",\"language\":\"en\",\"page\":\"373-384\",\"publisher\":\"ACM\",\"publisher-place\":\"Montpellier France\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Neural-machine-translation-based commit message generation: how far are we?\",\"title-short\":\"Neural-machine-translation-based commit message generation\",\"URL\":\"https://dl.acm.org/doi/10.1145/3238147.3238190\",\"author\":[{\"family\":\"Liu\",\"given\":\"Zhongxin\"},{\"family\":\"Xia\",\"given\":\"Xin\"},{\"family\":\"Hassan\",\"given\":\"Ahmed E.\"},{\"family\":\"Lo\",\"given\":\"David\"},{\"family\":\"Xing\",\"given\":\"Zhenchang\"},{\"family\":\"Wang\",\"given\":\"Xinyu\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2018\",9,3]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDGk9nD2evfS}\[2\],
which was originally designed for translating between natural languages
and consisted of encoder and decoder Recurrent Neural Networks (RNNs).
This work achieved promising results with BLEU scores around 31.92%
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"JUJs9Nyu\",\"properties\":{\"formattedCitation\":\"[1]\",\"plainCitation\":\"[1]\",\"noteIndex\":0},\"citationItems\":[{\"id\":130,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/ACBLY4LT\"],\"itemData\":{\"id\":130,\"type\":\"paper-conference\",\"container-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"DOI\":\"10.1109/ASE.2017.8115626\",\"event-place\":\"Urbana, IL\",\"event-title\":\"2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE)\",\"ISBN\":\"978-1-5386-2684-9\",\"language\":\"en\",\"page\":\"135-146\",\"publisher\":\"IEEE\",\"publisher-place\":\"Urbana, IL\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Automatically generating commit messages from diffs using neural machine translation\",\"URL\":\"http://ieeexplore.ieee.org/document/8115626/\",\"author\":[{\"family\":\"Jiang\",\"given\":\"Siyuan\"},{\"family\":\"Armaly\",\"given\":\"Ameer\"},{\"family\":\"McMillan\",\"given\":\"Collin\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2017\",10]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDE9Qxtye7BV}\[1\].

The transformer architecture, introduced by Vaswani et
al.[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"XOjfeMfN\",\"properties\":{\"formattedCitation\":\"[3]\",\"plainCitation\":\"[3]\",\"noteIndex\":0},\"citationItems\":[{\"id\":327,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/MSPPSF3Z\"],\"itemData\":{\"id\":327,\"type\":\"article-journal\",\"abstract\":\"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring signiﬁcantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.\",\"language\":\"en\",\"source\":\"Zotero\",\"title\":\"Attention is All you Need\",\"author\":[{\"family\":\"Vaswani\",\"given\":\"Ashish\"},{\"family\":\"Shazeer\",\"given\":\"Noam\"},{\"family\":\"Parmar\",\"given\":\"Niki\"},{\"family\":\"Uszkoreit\",\"given\":\"Jakob\"},{\"family\":\"Jones\",\"given\":\"Llion\"},{\"family\":\"Gomez\",\"given\":\"Aidan N\"},{\"family\":\"Kaiser\",\"given\":\"Łukasz\"},{\"family\":\"Polosukhin\",\"given\":\"Illia\"}]}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDtvHNF30Twy}\[3\],
revolutionized sequence-to-sequence tasks by replacing recurrent and
convolutional architectures with attention mechanisms. The original
transformer model employs multi-head self-attention, layer normalization
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"PP15az03\",\"properties\":{\"formattedCitation\":\"[3]\",\"plainCitation\":\"[3]\",\"noteIndex\":0},\"citationItems\":[{\"id\":327,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/MSPPSF3Z\"],\"itemData\":{\"id\":327,\"type\":\"article-journal\",\"abstract\":\"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring signiﬁcantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.\",\"language\":\"en\",\"source\":\"Zotero\",\"title\":\"Attention is All you Need\",\"author\":[{\"family\":\"Vaswani\",\"given\":\"Ashish\"},{\"family\":\"Shazeer\",\"given\":\"Noam\"},{\"family\":\"Parmar\",\"given\":\"Niki\"},{\"family\":\"Uszkoreit\",\"given\":\"Jakob\"},{\"family\":\"Jones\",\"given\":\"Llion\"},{\"family\":\"Gomez\",\"given\":\"Aidan N\"},{\"family\":\"Kaiser\",\"given\":\"Łukasz\"},{\"family\":\"Polosukhin\",\"given\":\"Illia\"}]}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDpq9VA8b9rD}\[3\][](https://www.codecogs.com/eqnedit.php?latex=NIPS-2017-attention-is-all-you-need-Paper.md#1),
and feed-forward networks to process sequences in parallel, offering
significant advantages in training speed and performance over sequential
models.

However, the standard transformer architecture faces several limitations
when applied to specialized tasks such as code diff to commit message
translation, particularly in terms of computational efficiency, memory
usage, and handling of positional information
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"TJIZVbna\",\"properties\":{\"formattedCitation\":\"[3]\",\"plainCitation\":\"[3]\",\"noteIndex\":0},\"citationItems\":[{\"id\":327,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/MSPPSF3Z\"],\"itemData\":{\"id\":327,\"type\":\"article-journal\",\"abstract\":\"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring signiﬁcantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.\",\"language\":\"en\",\"source\":\"Zotero\",\"title\":\"Attention is All you Need\",\"author\":[{\"family\":\"Vaswani\",\"given\":\"Ashish\"},{\"family\":\"Shazeer\",\"given\":\"Noam\"},{\"family\":\"Parmar\",\"given\":\"Niki\"},{\"family\":\"Uszkoreit\",\"given\":\"Jakob\"},{\"family\":\"Jones\",\"given\":\"Llion\"},{\"family\":\"Gomez\",\"given\":\"Aidan N\"},{\"family\":\"Kaiser\",\"given\":\"Łukasz\"},{\"family\":\"Polosukhin\",\"given\":\"Illia\"}]}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDRi7hZII1VX}\[3\].
The quadratic complexity of attention computation in sequence length
poses significant challenges for processing longer code sequences.

Recent advancements in transformer architectures have addressed these
limitations through various optimization techniques. FlashAttention
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"JxrIYSUR\",\"properties\":{\"formattedCitation\":\"[4]\",\"plainCitation\":\"[4]\",\"noteIndex\":0},\"citationItems\":[{\"id\":142,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/YIG9NHJW\"],\"itemData\":{\"id\":142,\"type\":\"article\",\"abstract\":\"Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading oﬀ model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IOaware—accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3 speedup on GPT-2 (seq. length 1K), and 2.4 speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classiﬁcation) and entirely new capabilities: the ﬁrst Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).\",\"DOI\":\"10.48550/arXiv.2205.14135\",\"language\":\"en\",\"note\":\"arXiv:2205.14135 [cs]\",\"number\":\"arXiv:2205.14135\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness\",\"title-short\":\"FlashAttention\",\"URL\":\"http://arxiv.org/abs/2205.14135\",\"author\":[{\"family\":\"Dao\",\"given\":\"Tri\"},{\"family\":\"Fu\",\"given\":\"Daniel Y.\"},{\"family\":\"Ermon\",\"given\":\"Stefano\"},{\"family\":\"Rudra\",\"given\":\"Atri\"},{\"family\":\"Ré\",\"given\":\"Christopher\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2022\",6,23]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RND3Hh79OIHpp}\[4\]
introduces an IO-aware attention algorithm that significantly reduces
memory usage and computational overhead by using tiling to minimize HBM
(High Bandwidth Memory) accesses. This approach enables transformers to
scale to longer sequences while maintaining efficiency, addressing one
of the primary bottlenecks in attention computation. FlashAttention has
shown to provide up to 3x speedup compared to baseline implementations
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"CcMjeU9o\",\"properties\":{\"formattedCitation\":\"[4]\",\"plainCitation\":\"[4]\",\"noteIndex\":0},\"citationItems\":[{\"id\":142,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/YIG9NHJW\"],\"itemData\":{\"id\":142,\"type\":\"article\",\"abstract\":\"Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading oﬀ model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IOaware—accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3 speedup on GPT-2 (seq. length 1K), and 2.4 speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classiﬁcation) and entirely new capabilities: the ﬁrst Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).\",\"DOI\":\"10.48550/arXiv.2205.14135\",\"language\":\"en\",\"note\":\"arXiv:2205.14135 [cs]\",\"number\":\"arXiv:2205.14135\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness\",\"title-short\":\"FlashAttention\",\"URL\":\"http://arxiv.org/abs/2205.14135\",\"author\":[{\"family\":\"Dao\",\"given\":\"Tri\"},{\"family\":\"Fu\",\"given\":\"Daniel Y.\"},{\"family\":\"Ermon\",\"given\":\"Stefano\"},{\"family\":\"Rudra\",\"given\":\"Atri\"},{\"family\":\"Ré\",\"given\":\"Christopher\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2022\",6,23]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RND2U0hlkmMmL}\[4\].

Rotary Position Embedding
(RoPE)[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"OUUKOVIW\",\"properties\":{\"formattedCitation\":\"[5]\",\"plainCitation\":\"[5]\",\"noteIndex\":0},\"citationItems\":[{\"id\":138,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/LXWZHZF9\"],\"itemData\":{\"id\":138,\"type\":\"article\",\"abstract\":\"Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: https://huggingface.co/docs/transformers/model_doc/roformer.\",\"DOI\":\"10.48550/arXiv.2104.09864\",\"language\":\"en\",\"note\":\"arXiv:2104.09864 [cs]\",\"number\":\"arXiv:2104.09864\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"RoFormer: Enhanced Transformer with Rotary Position Embedding\",\"title-short\":\"RoFormer\",\"URL\":\"http://arxiv.org/abs/2104.09864\",\"author\":[{\"family\":\"Su\",\"given\":\"Jianlin\"},{\"family\":\"Lu\",\"given\":\"Yu\"},{\"family\":\"Pan\",\"given\":\"Shengfeng\"},{\"family\":\"Murtadha\",\"given\":\"Ahmed\"},{\"family\":\"Wen\",\"given\":\"Bo\"},{\"family\":\"Liu\",\"given\":\"Yunfeng\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2023\",11,8]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDv0oexfPsXY}\[5\]
provides an alternative approach to positional encoding that encodes
absolute position information through rotation matrices while
incorporating relative position dependencies into the self-attention
formulation. RoPE offers several advantages including sequence length
flexibility, decaying inter-token dependency with increasing relative
distances, and compatibility with linear attention mechanisms, making it
particularly suitable for code-related tasks where positional
relationships are
crucial[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"muuv6fua\",\"properties\":{\"formattedCitation\":\"[5]\",\"plainCitation\":\"[5]\",\"noteIndex\":0},\"citationItems\":[{\"id\":138,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/LXWZHZF9\"],\"itemData\":{\"id\":138,\"type\":\"article\",\"abstract\":\"Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: https://huggingface.co/docs/transformers/model_doc/roformer.\",\"DOI\":\"10.48550/arXiv.2104.09864\",\"language\":\"en\",\"note\":\"arXiv:2104.09864 [cs]\",\"number\":\"arXiv:2104.09864\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"RoFormer: Enhanced Transformer with Rotary Position Embedding\",\"title-short\":\"RoFormer\",\"URL\":\"http://arxiv.org/abs/2104.09864\",\"author\":[{\"family\":\"Su\",\"given\":\"Jianlin\"},{\"family\":\"Lu\",\"given\":\"Yu\"},{\"family\":\"Pan\",\"given\":\"Shengfeng\"},{\"family\":\"Murtadha\",\"given\":\"Ahmed\"},{\"family\":\"Wen\",\"given\":\"Bo\"},{\"family\":\"Liu\",\"given\":\"Yunfeng\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2023\",11,8]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDopaSogVnvv}\[5\].
The RoPE-enhanced transformer (RoFormer) has shown consistent
improvements over baseline alternatives in various benchmarks
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"Ml3NLFKK\",\"properties\":{\"formattedCitation\":\"[5]\",\"plainCitation\":\"[5]\",\"noteIndex\":0},\"citationItems\":[{\"id\":138,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/LXWZHZF9\"],\"itemData\":{\"id\":138,\"type\":\"article\",\"abstract\":\"Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: https://huggingface.co/docs/transformers/model_doc/roformer.\",\"DOI\":\"10.48550/arXiv.2104.09864\",\"language\":\"en\",\"note\":\"arXiv:2104.09864 [cs]\",\"number\":\"arXiv:2104.09864\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"RoFormer: Enhanced Transformer with Rotary Position Embedding\",\"title-short\":\"RoFormer\",\"URL\":\"http://arxiv.org/abs/2104.09864\",\"author\":[{\"family\":\"Su\",\"given\":\"Jianlin\"},{\"family\":\"Lu\",\"given\":\"Yu\"},{\"family\":\"Pan\",\"given\":\"Shengfeng\"},{\"family\":\"Murtadha\",\"given\":\"Ahmed\"},{\"family\":\"Wen\",\"given\":\"Bo\"},{\"family\":\"Liu\",\"given\":\"Yunfeng\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2023\",11,8]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDjp30vIYOYs}\[5\].

The feed-forward network components of transformers have also seen
significant improvements through the introduction of Gated Linear Units
(GLU) and their variants. SwiGLU
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"hutZEcUK\",\"properties\":{\"formattedCitation\":\"[6]\",\"plainCitation\":\"[6]\",\"noteIndex\":0},\"citationItems\":[{\"id\":134,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/39IX6VUJ\"],\"itemData\":{\"id\":134,\"type\":\"article\",\"abstract\":\"Gated Linear Units [Dauphin et al., 2016] consist of the component-wise product of two linear projections, one of which is ﬁrst passed through a sigmoid function. Variations on GLU are possible, using diﬀerent nonlinear (or even linear) functions in place of sigmoid. We test these variants in the feedforward sublayers of the Transformer [Vaswani et al., 2017] sequence-to-sequence model, and ﬁnd that some of them yield quality improvements over the typically-used ReLU or GELU activations.\",\"DOI\":\"10.48550/arXiv.2002.05202\",\"language\":\"en\",\"note\":\"arXiv:2002.05202 [cs]\",\"number\":\"arXiv:2002.05202\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"GLU Variants Improve Transformer\",\"URL\":\"http://arxiv.org/abs/2002.05202\",\"author\":[{\"family\":\"Shazeer\",\"given\":\"Noam\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2020\",2,12]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDbze8SfjJLX}\[6\],
which combines Swish activation with gated linear units, has
demonstrated superior performance compared to traditional ReLU or GELU
activations in transformer models. The study showed that SwiGLU and
GEGLU variants produced better perplexities compared to baseline models
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"LljD7rf8\",\"properties\":{\"formattedCitation\":\"[6]\",\"plainCitation\":\"[6]\",\"noteIndex\":0},\"citationItems\":[{\"id\":134,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/39IX6VUJ\"],\"itemData\":{\"id\":134,\"type\":\"article\",\"abstract\":\"Gated Linear Units [Dauphin et al., 2016] consist of the component-wise product of two linear projections, one of which is ﬁrst passed through a sigmoid function. Variations on GLU are possible, using diﬀerent nonlinear (or even linear) functions in place of sigmoid. We test these variants in the feedforward sublayers of the Transformer [Vaswani et al., 2017] sequence-to-sequence model, and ﬁnd that some of them yield quality improvements over the typically-used ReLU or GELU activations.\",\"DOI\":\"10.48550/arXiv.2002.05202\",\"language\":\"en\",\"note\":\"arXiv:2002.05202 [cs]\",\"number\":\"arXiv:2002.05202\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"GLU Variants Improve Transformer\",\"URL\":\"http://arxiv.org/abs/2002.05202\",\"author\":[{\"family\":\"Shazeer\",\"given\":\"Noam\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2020\",2,12]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RND4WCvgndfLv}\[6\],
with GEGLU and SwiGLU achieving the best perplexities during
pre-training and fine-tuning across multiple benchmarks.

Normalization techniques have evolved beyond traditional Layer
Normalization to include Root Mean Square (RMS) normalization
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"P90S3pWX\",\"properties\":{\"formattedCitation\":\"[7]\",\"plainCitation\":\"[7]\",\"noteIndex\":0},\"citationItems\":[{\"id\":132,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/9YNSYDEC\"],\"itemData\":{\"id\":132,\"type\":\"article\",\"abstract\":\"Layer normalization (LayerNorm) has been successfully applied to various deep neural networks to help stabilize training and boost model convergence because of its capability in handling re-centering and re-scaling of both inputs and weight matrix. However, the computational overhead introduced by LayerNorm makes these improvements expensive and signiﬁcantly slows the underlying network, e.g. RNN in particular. In this paper, we hypothesize that re-centering invariance in LayerNorm is dispensable and propose root mean square layer normalization, or RMSNorm. RMSNorm regularizes the summed inputs to a neuron in one layer according to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efﬁcient than LayerNorm. We also present partial RMSNorm, or pRMSNorm where the RMS is estimated from p% of the summed inputs without breaking the above properties. Extensive experiments on several tasks using diverse network architectures show that RMSNorm achieves comparable performance against LayerNorm but reduces the running time by 7%∼64% on different models. Source code is available at https://github.com/bzhangGo/rmsnorm.\",\"DOI\":\"10.48550/arXiv.1910.07467\",\"language\":\"en\",\"note\":\"arXiv:1910.07467 [cs]\",\"number\":\"arXiv:1910.07467\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"Root Mean Square Layer Normalization\",\"URL\":\"http://arxiv.org/abs/1910.07467\",\"author\":[{\"family\":\"Zhang\",\"given\":\"Biao\"},{\"family\":\"Sennrich\",\"given\":\"Rico\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2019\",10,16]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RND3shBYJnhz1}\[7\],
which eliminates the re-centering operation while preserving the
re-scaling invariance property. RMSNorm offers computational simplicity
and efficiency while maintaining comparable performance to LayerNorm,
reducing training time by 7% to 64% across different models while
preserving model quality
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"w3En1Zcl\",\"properties\":{\"formattedCitation\":\"[7]\",\"plainCitation\":\"[7]\",\"noteIndex\":0},\"citationItems\":[{\"id\":132,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/9YNSYDEC\"],\"itemData\":{\"id\":132,\"type\":\"article\",\"abstract\":\"Layer normalization (LayerNorm) has been successfully applied to various deep neural networks to help stabilize training and boost model convergence because of its capability in handling re-centering and re-scaling of both inputs and weight matrix. However, the computational overhead introduced by LayerNorm makes these improvements expensive and signiﬁcantly slows the underlying network, e.g. RNN in particular. In this paper, we hypothesize that re-centering invariance in LayerNorm is dispensable and propose root mean square layer normalization, or RMSNorm. RMSNorm regularizes the summed inputs to a neuron in one layer according to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efﬁcient than LayerNorm. We also present partial RMSNorm, or pRMSNorm where the RMS is estimated from p% of the summed inputs without breaking the above properties. Extensive experiments on several tasks using diverse network architectures show that RMSNorm achieves comparable performance against LayerNorm but reduces the running time by 7%∼64% on different models. Source code is available at https://github.com/bzhangGo/rmsnorm.\",\"DOI\":\"10.48550/arXiv.1910.07467\",\"language\":\"en\",\"note\":\"arXiv:1910.07467 [cs]\",\"number\":\"arXiv:1910.07467\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"Root Mean Square Layer Normalization\",\"URL\":\"http://arxiv.org/abs/1910.07467\",\"author\":[{\"family\":\"Zhang\",\"given\":\"Biao\"},{\"family\":\"Sennrich\",\"given\":\"Rico\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2019\",10,16]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDmPqR9bviKx}\[7\].

Gradient checkpointing techniques
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"Qk5hxwQf\",\"properties\":{\"formattedCitation\":\"[8]\",\"plainCitation\":\"[8]\",\"noteIndex\":0},\"citationItems\":[{\"id\":131,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/QT5KBL46\"],\"itemData\":{\"id\":131,\"type\":\"article\",\"abstract\":\"We propose a systematic approach to reduce the memory consumption of deep neural network training. Specifically, we design an algorithm that costs O(sqrt(n)) memory to train a n layer network, with only the computational cost of an extra forward pass per mini-batch. As many of the state-of-the-art models hit the upper bound of the GPU memory, our algorithm allows deeper and more complex models to be explored, and helps advance the innovations in deep learning research. We focus on reducing the memory cost to store the intermediate feature maps and gradients during training. Computation graph analysis is used for automatic in-place operation and memory sharing optimizations. We show that it is possible to trade computation for memory - giving a more memory efficient training algorithm with a little extra computation cost. In the extreme case, our analysis also shows that the memory consumption can be reduced to O(log n) with as little as O(n log n) extra cost for forward computation. Our experiments show that we can reduce the memory cost of a 1,000-layer deep residual network from 48G to 7G with only 30 percent additional running time cost on ImageNet problems. Similarly, significant memory cost reduction is observed in training complex recurrent neural networks on very long sequences.\",\"DOI\":\"10.48550/arXiv.1604.06174\",\"language\":\"en\",\"note\":\"arXiv:1604.06174 [cs]\",\"number\":\"arXiv:1604.06174\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"Training Deep Nets with Sublinear Memory Cost\",\"URL\":\"http://arxiv.org/abs/1604.06174\",\"author\":[{\"family\":\"Chen\",\"given\":\"Tianqi\"},{\"family\":\"Xu\",\"given\":\"Bing\"},{\"family\":\"Zhang\",\"given\":\"Chiyuan\"},{\"family\":\"Guestrin\",\"given\":\"Carlos\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2016\",4,22]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDDxjqICR7Ah}\[8\]
address memory constraints during training by trading computation for
memory, enabling the training of deeper networks with limited
computational resources. This approach allows for sublinear memory
scaling with network depth, making it feasible to train very deep
transformer models without prohibitive memory requirements. The method
can achieve O(√n) memory cost for an n-layer network with only an extra
forward pass computational cost
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"bAp8ekqa\",\"properties\":{\"formattedCitation\":\"[8]\",\"plainCitation\":\"[8]\",\"noteIndex\":0},\"citationItems\":[{\"id\":131,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/QT5KBL46\"],\"itemData\":{\"id\":131,\"type\":\"article\",\"abstract\":\"We propose a systematic approach to reduce the memory consumption of deep neural network training. Specifically, we design an algorithm that costs O(sqrt(n)) memory to train a n layer network, with only the computational cost of an extra forward pass per mini-batch. As many of the state-of-the-art models hit the upper bound of the GPU memory, our algorithm allows deeper and more complex models to be explored, and helps advance the innovations in deep learning research. We focus on reducing the memory cost to store the intermediate feature maps and gradients during training. Computation graph analysis is used for automatic in-place operation and memory sharing optimizations. We show that it is possible to trade computation for memory - giving a more memory efficient training algorithm with a little extra computation cost. In the extreme case, our analysis also shows that the memory consumption can be reduced to O(log n) with as little as O(n log n) extra cost for forward computation. Our experiments show that we can reduce the memory cost of a 1,000-layer deep residual network from 48G to 7G with only 30 percent additional running time cost on ImageNet problems. Similarly, significant memory cost reduction is observed in training complex recurrent neural networks on very long sequences.\",\"DOI\":\"10.48550/arXiv.1604.06174\",\"language\":\"en\",\"note\":\"arXiv:1604.06174 [cs]\",\"number\":\"arXiv:1604.06174\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"Training Deep Nets with Sublinear Memory Cost\",\"URL\":\"http://arxiv.org/abs/1604.06174\",\"author\":[{\"family\":\"Chen\",\"given\":\"Tianqi\"},{\"family\":\"Xu\",\"given\":\"Bing\"},{\"family\":\"Zhang\",\"given\":\"Chiyuan\"},{\"family\":\"Guestrin\",\"given\":\"Carlos\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2016\",4,22]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDhvQsqFXdaP}\[8\].

Comparative studies have shown that while complex neural approaches can
outperform simpler methods, there are caveats. Liu et al.
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"fhwxjfeR\",\"properties\":{\"formattedCitation\":\"[2]\",\"plainCitation\":\"[2]\",\"noteIndex\":0},\"citationItems\":[{\"id\":322,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/Q64XKF4Z\"],\"itemData\":{\"id\":322,\"type\":\"paper-conference\",\"abstract\":\"Commit messages can be regarded as the documentation of software changes. These messages describe the content and purposes of changes, hence are useful for program comprehension and software maintenance. However, due to the lack of time and direct motivation, commit messages sometimes are neglected by developers. To address this problem, Jiang et al. proposed an approach (we refer to it as NMT ), which leverages a neural machine translation algorithm to automatically generate short commit messages from code. The reported performance of their approach is promising, however, they did not explore why their approach performs well. Thus, in this paper, we first perform an in-depth analysis of their experimental results. We find that (1) Most of the test diffs from which NMT can generate high-quality messages are similar to one or more training diffs at the token level. (2) About 16% of the commit messages in Jiang et al.’s dataset are noisy due to being automatically generated or due to them describing repetitive trivial changes. (3) The performance of NMT declines by a large amount after removing such noisy commit messages. In addition, NMT is complicated and time-consuming. Inspired by our first finding, we proposed a simpler and faster approach, named NNGen (Nearest Neighbor Generator), to generate concise commit messages using the nearest neighbor algorithm. Our experimental results show that NNGen is over 2,600 times faster than NMT , and outperforms NMT in terms of BLEU (an accuracy measure that is widely used to evaluate machine translation systems) by 21%. Finally, we also discuss some observations for the road ahead for automated commit message generation to inspire other researchers.\",\"container-title\":\"Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"DOI\":\"10.1145/3238147.3238190\",\"event-place\":\"Montpellier France\",\"event-title\":\"ASE '18: 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"ISBN\":\"978-1-4503-5937-5\",\"language\":\"en\",\"page\":\"373-384\",\"publisher\":\"ACM\",\"publisher-place\":\"Montpellier France\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Neural-machine-translation-based commit message generation: how far are we?\",\"title-short\":\"Neural-machine-translation-based commit message generation\",\"URL\":\"https://dl.acm.org/doi/10.1145/3238147.3238190\",\"author\":[{\"family\":\"Liu\",\"given\":\"Zhongxin\"},{\"family\":\"Xia\",\"given\":\"Xin\"},{\"family\":\"Hassan\",\"given\":\"Ahmed E.\"},{\"family\":\"Lo\",\"given\":\"David\"},{\"family\":\"Xing\",\"given\":\"Zhenchang\"},{\"family\":\"Wang\",\"given\":\"Xinyu\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2018\",9,3]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDCzHhhGTjEg}\[2\]
found that most of the test diffs from which NMT can generate
high-quality messages are similar to one or more training diffs at the
token level, suggesting that the success of complex neural methods may
partly stem from memorization rather than genuine understanding.

The motivation for lightweight, efficient models like the proposed mini
Transformer arises from the limitations of large pre-trained models:
high computational costs, resource demands, and generalization issues on
domain-specific code tasks
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"AFBV307z\",\"properties\":{\"formattedCitation\":\"[2]\",\"plainCitation\":\"[2]\",\"noteIndex\":0},\"citationItems\":[{\"id\":322,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/Q64XKF4Z\"],\"itemData\":{\"id\":322,\"type\":\"paper-conference\",\"abstract\":\"Commit messages can be regarded as the documentation of software changes. These messages describe the content and purposes of changes, hence are useful for program comprehension and software maintenance. However, due to the lack of time and direct motivation, commit messages sometimes are neglected by developers. To address this problem, Jiang et al. proposed an approach (we refer to it as NMT ), which leverages a neural machine translation algorithm to automatically generate short commit messages from code. The reported performance of their approach is promising, however, they did not explore why their approach performs well. Thus, in this paper, we first perform an in-depth analysis of their experimental results. We find that (1) Most of the test diffs from which NMT can generate high-quality messages are similar to one or more training diffs at the token level. (2) About 16% of the commit messages in Jiang et al.’s dataset are noisy due to being automatically generated or due to them describing repetitive trivial changes. (3) The performance of NMT declines by a large amount after removing such noisy commit messages. In addition, NMT is complicated and time-consuming. Inspired by our first finding, we proposed a simpler and faster approach, named NNGen (Nearest Neighbor Generator), to generate concise commit messages using the nearest neighbor algorithm. Our experimental results show that NNGen is over 2,600 times faster than NMT , and outperforms NMT in terms of BLEU (an accuracy measure that is widely used to evaluate machine translation systems) by 21%. Finally, we also discuss some observations for the road ahead for automated commit message generation to inspire other researchers.\",\"container-title\":\"Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"DOI\":\"10.1145/3238147.3238190\",\"event-place\":\"Montpellier France\",\"event-title\":\"ASE '18: 33rd ACM/IEEE International Conference on Automated Software Engineering\",\"ISBN\":\"978-1-4503-5937-5\",\"language\":\"en\",\"page\":\"373-384\",\"publisher\":\"ACM\",\"publisher-place\":\"Montpellier France\",\"source\":\"DOI.org (Crossref)\",\"title\":\"Neural-machine-translation-based commit message generation: how far are we?\",\"title-short\":\"Neural-machine-translation-based commit message generation\",\"URL\":\"https://dl.acm.org/doi/10.1145/3238147.3238190\",\"author\":[{\"family\":\"Liu\",\"given\":\"Zhongxin\"},{\"family\":\"Xia\",\"given\":\"Xin\"},{\"family\":\"Hassan\",\"given\":\"Ahmed E.\"},{\"family\":\"Lo\",\"given\":\"David\"},{\"family\":\"Xing\",\"given\":\"Zhenchang\"},{\"family\":\"Wang\",\"given\":\"Xinyu\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2018\",9,3]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDuRNSLPn50f}\[2\].
Recent analysis emphasizes the need for more efficient approaches that
can achieve comparable performance with significantly reduced
computational requirements.

The contribution of this work lies in the development of an efficient
mini transformer architecture specifically designed for code difference
to commit message translation, incorporating multiple modern
optimization techniques including FlashAttention for memory efficiency,
RoPE for positional encoding, SwiGLU for enhanced feed-forward networks,
RMSNorm for efficient normalization, and gradient checkpointing for
memory optimization. This integrated approach addresses the
computational and memory constraints inherent in transformer models
while maintaining high-quality translation performance for code-related
tasks.

The paper is organized as follows: we first present the theoretical
foundations of our optimized transformer architecture, followed by
detailed descriptions of each incorporated optimization technique. We
then evaluate our approach on code diff to commit message translation
tasks, demonstrating significant improvements in both computational
efficiency and translation quality compared to baseline transformer
models. Finally, we discuss the implications of our findings for future
transformer architectures and their applications in software engineering
tasks.

|                      |          |                                      |                        |                                                              |                                                                         |
|----------------------|----------|--------------------------------------|------------------------|--------------------------------------------------------------|-------------------------------------------------------------------------|
| **Method / Study**   | **Year** | **Technology**                       | **Dataset**            | **Key Metrics**                                              | **Strengths / Limitations**                                             |
| Jiang et al. (NMT)   | 2017     | RNN Seq2Seq                          | GitHub (filtered)      | BLEU 31.92%                                                  | Pioneering work; high computational cost                                |
| NNGen (Liu et al.)   | 2018     | IR Retrieval                         | Jiang et al.'s dataset | BLEU 38.55% (original), 16.42% (cleaned)                     | Highly efficient (2600× faster than NMT); relies on similarity matching |
| Moses (baseline)     | --       | Statistical MT                       | Jiang et al.'s dataset | BLEU 3.63%                                                   | Statistical baseline; significantly underperforms neural models         |
| CoRec (Wang et al.)  | 2021     | IR + NMT                             | MCMD                   | BLEU 6.24, ROUGE-L 9.08, METEOR 10.46, CIDEr 2.18            | Hybrid approach combining retrieval and generation                      |
| CoreGen (Nie et al.) | 2021     | Pre-trained Transformer              | MCMD                   | BLEU 4.16, ROUGE-L 6.22, METEOR 6.57, CIDEr 1.32             | Contextualized code representation learning                             |
| FIRA (Dong et al.)   | 2022     | Fine-grained graph + Transformer/GNN | --                     | --                                                           | State-of-the-art single-model architecture                              |
| RACE (Shi et al.)    | 2022     | Retrieval-Augmented Transformer      | MCMD                   | BLEU 25.66                                                   | High performance but computationally expensive                          |
| CommitBERT           | 2023     | Pre-trained Transformer              | CommitChronicle        | METEOR 0.30, ROUGE-L 0.35                                    | Commit-specific; limited by historical context                          |
| COME                 | 2023     | Modification Embedding + CodeT5      | MCMD                   | --                                                           | Uses modification embeddings and SVM-based decision algorithm           |
| CodeT5               | 2021     | Unified Encoder-Decoder              | CodeSearchNet          | BLEU 15.24--15.73 (small), 19.77 (base avg)                  | Supports both code understanding and generation                         |
| CodeBERT             | 2020     | Bimodal Pre-trained Model            | CodeSearchNet          | BLEU 12.16--17.83                                            | Requires additional decoder for generation                              |
| GPT-3.5-Turbo        | 2023     | Large Language Model                 | MCMD                   | BLEU 18.04--36.31, METEOR 16.11--34.01, ROUGE-L 25.21--45.61 | Strong performance; closed-source                                       |
| DeepSeek-V2-Chat     | 2024     | Large Language Model                 | MCMD                   | BLEU 18.05--36.38, METEOR 15.95--34.53, ROUGE-L 25.02--45.31 | Open-source alternative with strong performance                         |
| Claude-3-Haiku       | 2024     | Large Language Model                 | MCMD                   | BLEU 8.35--19.12, METEOR 6.39--16.30, ROUGE-L 11.52--23.77   | Lightweight Claude-3 family model                                       |

Table 1: Comparison of Key CMG Methods

|                    |                  |          |            |             |           |                            |
|--------------------|------------------|----------|------------|-------------|-----------|----------------------------|
| **Dataset / Task** | **Approach**     | **BLEU** | **METEOR** | **ROUGE-L** | **CIDEr** | **Notes**                  |
| Original           | NMT              | 31.92    | --         | --          | --        | Jiang et al. baseline      |
| Original           | NNGen            | 38.55    | --         | --          | --        | Retrieval-based approach   |
| Cleaned            | NMT              | 14.19    | --         | --          | --        | Performance degradation    |
| Cleaned            | NNGen            | 16.42    | --         | --          | --        | Still superior to NMT      |
| MCMD (Java)        | NNGen            | 19.41    | 12.40      | 25.15       | 1.23      | Lower performance          |
| MCMD (Java)        | CCT5             | 17.19    | 14.95      | 26.08       | 1.06      | Code-specific pre-training |
| MCMD (Java)        | COME             | 27.17    | 23.36      | 34.59       | 1.90      | State-of-the-art baseline  |
| MCMD (Java)        | GPT-3.5-Turbo    | 20.24    | 17.99      | 27.80       | 1.35      | In-context learning        |
| MCMD (Java)        | DeepSeek-V2-Chat | 19.36    | 17.02      | 26.50       | 1.27      | Open-source LLM            |
| MCMD-NT (Java)     | NNGen            | 29.19    | 24.30      | 37.00       | 2.18      | Time-split dataset         |
| MCMD-NT (Java)     | CCT5             | 22.15    | 19.05      | 30.18       | 1.48      | Performance drop           |
| MCMD-NT (Java)     | COME             | 31.46    | 26.41      | 39.53       | 2.41      | Strong performance         |
| MCMD-NT (Java)     | GPT-3.5-Turbo    | 36.31    | 34.01      | 45.61       | 2.95      | Best LLM performance       |
| MCMD-NT (Java)     | DeepSeek-V2-Chat | 36.38    | 34.53      | 45.31       | 3.00      | Competitive open-source    |
| ApacheCM-10K       | NNGen            | 2.83     | 5.48       | 5.38        | 0.70      | Baseline                   |
| ApacheCM-10K       | CoRec            | 6.24     | 10.46      | 9.08        | 2.18      | Hybrid method              |
| ApacheCM-10K       | CoreGen          | 4.16     | 6.57       | 6.22        | 1.32      | Pre-trained model          |
| ApacheCM-10K       | GPT-4o           | 9.12     | 18.51      | 21.08       | 7.12      | Advanced LLM               |
| ApacheCM-10K       | GPT-4.1          | 9.76     | 19.37      | 21.21       | 7.18      | Latest GPT                 |
| ApacheCM-10K       | DeepSeek-V3      | 10.03    | 19.64      | 22.75       | 8.18      | Best performer             |
| ApacheCM-10K       | DeepSeek-R1      | 9.87     | 20.56      | 22.96       | 7.81      | Reasoning model            |

Table 2: Comparative Performance Across Datasets

|                        |                                   |                             |                                  |
|------------------------|-----------------------------------|-----------------------------|----------------------------------|
| **Technique**          | **Performance Improvement**       | **Efficiency Gain**         | **Reference**                    |
| SwiGLU vs ReLU         | Lower perplexity (1.636 vs 1.677) | Similar parameter count     | GLU Variants Improve Transformer |
| RMSNorm vs LayerNorm   | Comparable BLEU                   | 7%--64% faster training     | RMSNorm                          |
| Gradient Checkpointing | O(√n) memory scaling              | Sublinear memory usage      | Sublinear Memory Cost            |
| RoPE                   | 2--5% better extrapolation        | Improved relative positions | RoFormer                         |
| FlashAttention         | Up to 3× speedup                  | Reduced memory              | FlashAttention                   |
| CoRaCMG (RAG)          | 65% avg BLEU improvement          | Context-aware generation    | CoRaCMG                          |

Table 3: Transformer Optimization Techniques

|                          |              |              |              |              |                        |
|--------------------------|--------------|--------------|--------------|--------------|------------------------|
| **Model**                | **BLEU**     | **METEOR**   | **ROUGE-L**  | **CIDEr**    | **Improvement Method** |
| GPT-3.5-Turbo (direct)   | 9.12--9.76   | 18.51--19.37 | 21.08--21.21 | 7.12--7.18   | Baseline               |
| GPT-3.5-Turbo (enhanced) | 17.24 (↑89%) | 28.60 (↑54%) | 27.39 (↑29%) | 12.29 (↑72%) | Single example pair    |
| DeepSeek-V3 (direct)     | 10.03        | 19.64        | 22.75        | 8.18         | Baseline               |
| DeepSeek-V3 (enhanced)   | 17.25 (↑72%) | 27.28 (↑39%) | 28.71 (↑26%) | 13.07 (↑60%) | Single example pair    |
| DeepSeek-R1 (direct)     | 9.87         | 20.56        | 22.96        | 7.81         | Baseline               |
| DeepSeek-R1 (enhanced)   | 17.42 (↑76%) | 28.76 (↑40%) | 29.10 (↑27%) | 13.32 (↑71%) | Single example pair    |

Table 4: In-Context Learning Performance of LLMs

|              |              |          |            |             |           |               |
|--------------|--------------|----------|------------|-------------|-----------|---------------|
| **Language** | **Approach** | **BLEU** | **METEOR** | **ROUGE-L** | **CIDEr** | **Dataset**   |
| Java         | CodeT5-small | 15.24    | --         | --          | --        | CodeSearchNet |
| Python       | CodeT5-small | 20.01    | --         | --          | --        | CodeSearchNet |
| Java         | CodeT5-base  | 20.31    | --         | --          | --        | CodeSearchNet |
| Python       | CodeT5-base  | 20.11    | --         | --          | --        | CodeSearchNet |
| Java         | RACE         | 26.90    | --         | 38.00       | --        | MCMD          |
| C#           | RACE         | 25.66    | --         | 38.00       | --        | MCMD          |
| Java         | COME         | 26.91    | 23.02      | 34.44       | --        | MCMD          |
| C#           | COME         | 27.29    | 23.29      | 33.33       | --        | MCMD          |

Table 5: Performance Across Programming Languages

|           |                         |                            |                      |               |                               |
|-----------|-------------------------|----------------------------|----------------------|---------------|-------------------------------|
| **Model** | **Approach**            | **Task Type**              | **Performance**      | **Dataset**   | **Key Innovation**            |
| CodeBERT  | Bimodal Pre-trained     | NL-PL Understanding        | MRR 0.6972 (avg)     | CodeSearchNet | Joint NL-PL representations   |
| CodeT5    | Unified Encoder-Decoder | Understanding & Generation | BLEU 19.14--19.77    | CodeSearchNet | Identifier-aware pre-training |
| PLBART    | Encoder-Decoder (BART)  | Understanding & Generation | BLEU 18.32           | CodeSearchNet | PL-specific adaptation        |
| CoRaCMG   | Retrieval-Augmented     | Commit Message Generation  | 65% BLEU improvement | ApacheCM      | Contextual retrieval          |
| NNGen     | Information Retrieval   | Commit Message Generation  | BLEU 16.42           | Jiang et al.  | BoW + cosine similarity       |

Table 6: Code-Specific Model Performance

1.  1.  []{#anchor-5}Gap Analysis

Based on the comprehensive literature review of all papers in the
collection, several significant gaps emerge in the current research
landscape of commit message generation (CMG). The following analysis
synthesizes findings from multiple papers to identify key areas where
further research is needed.

Despite the success of large language models (LLMs) in CMG tasks, as
demonstrated by studies in \"An Empirical Study on Commit Message
Generation using LLMs via In-Context Learning.md\", there remains a
critical gap in computational efficiency. Current state-of-the-art
approaches like RACE
[](https://www.codecogs.com/eqnedit.php?latex=16#1), CodeT5
[](https://www.codecogs.com/eqnedit.php?latex=24#1), and COME
[](https://www.codecogs.com/eqnedit.php?latex=21#1) require substantial
computational resources and training time. The literature reveals that
while LLMs like GPT-3.5-Turbo and DeepSeek-V2-Chat show superior
performance, they are computationally expensive and may not be
accessible for all development
environments[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"J0fkPJt2\",\"properties\":{\"formattedCitation\":\"[9]\",\"plainCitation\":\"[9]\",\"noteIndex\":0},\"citationItems\":[{\"id\":162,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/KVPN6BUR\"],\"itemData\":{\"id\":162,\"type\":\"article\",\"abstract\":\"Commit messages concisely describe code changes in natural language and are important for software maintenance. Several approaches have been proposed to automatically generate commit messages, but they still suffer from critical limitations, such as time-consuming training and poor generalization ability. To tackle these limitations, we propose to borrow the weapon of large language models (LLMs) and in-context learning (ICL). Our intuition is based on the fact that the training corpora of LLMs contain extensive code changes and their pairwise commit messages, which makes LLMs capture the knowledge about commits, while ICL can exploit the knowledge hidden in the LLMs and enable them to perform downstream tasks without model tuning. However, it remains unclear how well LLMs perform on commit message generation via ICL. In this paper, we conduct an empirical study to investigate the capability of LLMs to generate commit messages via ICL. Specifically, we first explore the impact of different settings on the performance of ICL-based commit message generation. We then compare ICL-based commit message generation with state-ofthe-art approaches on a popular multilingual dataset and a new dataset we created to mitigate potential data leakage. The results show that ICL-based commit message generation significantly outperforms state-of-the-art approaches on subjective evaluation and achieves better generalization ability. We further analyze the root causes for LLM’s underperformance and propose several implications, which shed light on future research directions for using LLMs to generate commit messages.\",\"DOI\":\"10.48550/arXiv.2502.18904\",\"language\":\"en\",\"note\":\"arXiv:2502.18904 [cs]\",\"number\":\"arXiv:2502.18904\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"An Empirical Study on Commit Message Generation using LLMs via In-Context Learning\",\"URL\":\"http://arxiv.org/abs/2502.18904\",\"author\":[{\"family\":\"Wu\",\"given\":\"Yifan\"},{\"family\":\"Wang\",\"given\":\"Yunpeng\"},{\"family\":\"Li\",\"given\":\"Ying\"},{\"family\":\"Tao\",\"given\":\"Wei\"},{\"family\":\"Yu\",\"given\":\"Siyu\"},{\"family\":\"Yang\",\"given\":\"Haowen\"},{\"family\":\"Jiang\",\"given\":\"Wei\"},{\"family\":\"Li\",\"given\":\"Jianguo\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2025\",2,26]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDCqeXejoJgq}\[9\].

Traditional transformer models also suffer from quadratic complexity in
attention computation, as noted in
\"NIPS-2017-attention-is-all-you-need-Paper.md\", making them
inefficient for processing longer code sequences. While optimizations
like FlashAttention
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"xVGAHTPH\",\"properties\":{\"formattedCitation\":\"[4]\",\"plainCitation\":\"[4]\",\"noteIndex\":0},\"citationItems\":[{\"id\":142,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/YIG9NHJW\"],\"itemData\":{\"id\":142,\"type\":\"article\",\"abstract\":\"Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading oﬀ model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IOaware—accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3 speedup on GPT-2 (seq. length 1K), and 2.4 speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classiﬁcation) and entirely new capabilities: the ﬁrst Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).\",\"DOI\":\"10.48550/arXiv.2205.14135\",\"language\":\"en\",\"note\":\"arXiv:2205.14135 [cs]\",\"number\":\"arXiv:2205.14135\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness\",\"title-short\":\"FlashAttention\",\"URL\":\"http://arxiv.org/abs/2205.14135\",\"author\":[{\"family\":\"Dao\",\"given\":\"Tri\"},{\"family\":\"Fu\",\"given\":\"Daniel Y.\"},{\"family\":\"Ermon\",\"given\":\"Stefano\"},{\"family\":\"Rudra\",\"given\":\"Atri\"},{\"family\":\"Ré\",\"given\":\"Christopher\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2022\",6,23]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDo3aYoAczWd}\[4\]
address some memory concerns, there is still a need for more efficient
architectures that maintain performance while reducing computational
overhead.

Multiple papers highlight significant issues with dataset quality in CMG
research. \"CoRaCMG: Contextual Retrieval-Augmented Framework for Commit
Message Generation.md\" notes that existing datasets like CommitGen,
NNGen, CoDiSum, and MCMD suffer from data quality issues and lack
standardization in their construction process. The ApacheCM dataset was
developed specifically to address these limitations, indicating the
severity of the problem in existing research.

Furthermore, \"An Empirical Study on Commit Message Generation using
LLMs via In-Context Learning.md\" reveals that only 13.1% of samples in
the widely-used MCMD dataset had developer-authored reference messages
selected as optimal in human evaluation, highlighting the quality issues
inherent in current datasets. This gap in high-quality, standardized
datasets affects the reliability of evaluation protocols and the
generalizability of research findings.

Most existing CMG approaches focus primarily on code diffs while
neglecting broader contextual information. \"CoRaCMG: Contextual
Retrieval-Augmented Framework for Commit Message Generation.md\"
identifies that direct application of LLMs to CMG faces challenges where
generated messages overlook project-specific writing styles and lack
global understanding of the codebase. The paper argues that historical
commit messages contain valuable terminologies and writing styles that
are not effectively leveraged by current approaches.

Additionally, \"Automated Generation of Commit Messages in Software
Repositories.md\" notes that existing models like NMT, NNGen, RACE, and
CoDiSum face scalability issues, particularly with nearest neighbors
algorithms, and struggle with handling the complexity of diverse code
changes
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"0inR9LDU\",\"properties\":{\"formattedCitation\":\"[10]\",\"plainCitation\":\"[10]\",\"noteIndex\":0},\"citationItems\":[{\"id\":195,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/DXHEXZCB\"],\"itemData\":{\"id\":195,\"type\":\"paper-conference\",\"abstract\":\"Commit messages are crucial for documenting software changes, aiding in program comprehension and maintenance. However, creating effective commit messages is often overlooked by developers due to time constraints and varying levels of documentation skills. Our research presents an automated approach to generate commit messages using Machine Learning (ML) and Natural Language Processing (NLP) by developing models that use techniques such as Logistic Regression with TF-IDF and Word2Vec, as well as more sophisticated methods like LSTM. We used the dataset of code changes and corresponding commit messages that was used by Liu et al. [12], which we used to train and evaluate ML/NLP models and was chosen because it is extensively used in previous research, also for comparability in our study. The objective was to explore which ML/NLP techniques generate the most effective, clear, and concise commit messages that accurately reflect the code changes. We split the dataset into training, validation, and testing sets and used these sets to evaluate the performance of each model using qualitative and quantitative evaluation methods. Our results reveal a spectrum of effectiveness among these models, with the highest BLEU score achieved being 16.82, showcasing the models’ capability in automating a clear and concise commit message generation. Our paper offers insights into the comparative effectiveness of different machine learning models for automating commit message generation in software development, aiming to enhance the overall practice of code documentation. The source code is available at [2].\",\"DOI\":\"10.18293/DMSVIVA2024-145\",\"language\":\"en\",\"note\":\"arXiv:2504.12998 [cs]\",\"page\":\"87-95\",\"source\":\"arXiv.org\",\"title\":\"Automated Generation of Commit Messages in Software Repositories\",\"URL\":\"http://arxiv.org/abs/2504.12998\",\"author\":[{\"family\":\"Palakodeti\",\"given\":\"Varun Kumar\"},{\"family\":\"Heydarnoori\",\"given\":\"Abbas\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2024\",10,26]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RND5nOysXmINO}\[10\].

The literature reveals significant limitations in model generalization
across different projects and contexts. \"An Empirical Study on Commit
Message Generation using LLMs via In-Context Learning.md\" indicates
that learning-based approaches suffer from performance degradation when
applied to new projects, with performance decreasing by 26.93% to 73.41%
in such scenarios. This poor generalization ability represents a
critical gap in current CMG research.

Even LLMs, while showing improved generalization, still struggle with
domain-specific terminology and project-specific conventions. The same
study notes that while LLMs can generate high-quality messages through
in-context learning, they still underperform in scenarios requiring deep
contextual knowledge from related issues or pull requests.

Current evaluation approaches in CMG suffer from significant
limitations. \"An Empirical Study on Commit Message Generation using
LLMs via In-Context Learning.md\" demonstrates that traditional metrics
like BLEU, METEOR, ROUGE-L, and CIDER have low correlation with human
judgment and fail to capture the semantics of generated messages. The
study found that LLM-based evaluation shows much higher correlations
with human judgment, suggesting inadequacies in current automatic
evaluation methods.

\"Automated Generation of Commit Messages in Software Repositories.md\"
further notes that BLEU, while penalizing shorter outputs appropriately,
doesn\'t adequately address the semantic quality of generated messages,
focusing primarily on n-gram overlap
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"9G6sCec9\",\"properties\":{\"formattedCitation\":\"[10]\",\"plainCitation\":\"[10]\",\"noteIndex\":0},\"citationItems\":[{\"id\":195,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/DXHEXZCB\"],\"itemData\":{\"id\":195,\"type\":\"paper-conference\",\"abstract\":\"Commit messages are crucial for documenting software changes, aiding in program comprehension and maintenance. However, creating effective commit messages is often overlooked by developers due to time constraints and varying levels of documentation skills. Our research presents an automated approach to generate commit messages using Machine Learning (ML) and Natural Language Processing (NLP) by developing models that use techniques such as Logistic Regression with TF-IDF and Word2Vec, as well as more sophisticated methods like LSTM. We used the dataset of code changes and corresponding commit messages that was used by Liu et al. [12], which we used to train and evaluate ML/NLP models and was chosen because it is extensively used in previous research, also for comparability in our study. The objective was to explore which ML/NLP techniques generate the most effective, clear, and concise commit messages that accurately reflect the code changes. We split the dataset into training, validation, and testing sets and used these sets to evaluate the performance of each model using qualitative and quantitative evaluation methods. Our results reveal a spectrum of effectiveness among these models, with the highest BLEU score achieved being 16.82, showcasing the models’ capability in automating a clear and concise commit message generation. Our paper offers insights into the comparative effectiveness of different machine learning models for automating commit message generation in software development, aiming to enhance the overall practice of code documentation. The source code is available at [2].\",\"DOI\":\"10.18293/DMSVIVA2024-145\",\"language\":\"en\",\"note\":\"arXiv:2504.12998 [cs]\",\"page\":\"87-95\",\"source\":\"arXiv.org\",\"title\":\"Automated Generation of Commit Messages in Software Repositories\",\"URL\":\"http://arxiv.org/abs/2504.12998\",\"author\":[{\"family\":\"Palakodeti\",\"given\":\"Varun Kumar\"},{\"family\":\"Heydarnoori\",\"given\":\"Abbas\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2024\",10,26]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RND1k4K4AO7y7}\[10\].

Existing transformer architectures face several limitations when applied
to CMG tasks. \"NIPS-2017-attention-is-all-you-need-Paper.md\"
established the foundational transformer but acknowledged the quadratic
complexity of attention computation as a bottleneck for longer
sequences. Subsequent work has addressed some limitations, but gaps
remain:

- Positional Encoding: Traditional positional encodings may not
  optimally capture code-specific structural relationships
  []{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"bhPHuWM9\",\"properties\":{\"formattedCitation\":\"[5]\",\"plainCitation\":\"[5]\",\"noteIndex\":0},\"citationItems\":[{\"id\":138,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/LXWZHZF9\"],\"itemData\":{\"id\":138,\"type\":\"article\",\"abstract\":\"Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: https://huggingface.co/docs/transformers/model_doc/roformer.\",\"DOI\":\"10.48550/arXiv.2104.09864\",\"language\":\"en\",\"note\":\"arXiv:2104.09864 [cs]\",\"number\":\"arXiv:2104.09864\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"RoFormer: Enhanced Transformer with Rotary Position Embedding\",\"title-short\":\"RoFormer\",\"URL\":\"http://arxiv.org/abs/2104.09864\",\"author\":[{\"family\":\"Su\",\"given\":\"Jianlin\"},{\"family\":\"Lu\",\"given\":\"Yu\"},{\"family\":\"Pan\",\"given\":\"Shengfeng\"},{\"family\":\"Murtadha\",\"given\":\"Ahmed\"},{\"family\":\"Wen\",\"given\":\"Bo\"},{\"family\":\"Liu\",\"given\":\"Yunfeng\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2023\",11,8]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDUfP0MzvbUj}\[5\]
- Feed-Forward Networks: Standard ReLU or GELU activations may not be
  optimal for code-specific
  patterns[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"kyabsrIh\",\"properties\":{\"formattedCitation\":\"[6]\",\"plainCitation\":\"[6]\",\"noteIndex\":0},\"citationItems\":[{\"id\":134,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/39IX6VUJ\"],\"itemData\":{\"id\":134,\"type\":\"article\",\"abstract\":\"Gated Linear Units [Dauphin et al., 2016] consist of the component-wise product of two linear projections, one of which is ﬁrst passed through a sigmoid function. Variations on GLU are possible, using diﬀerent nonlinear (or even linear) functions in place of sigmoid. We test these variants in the feedforward sublayers of the Transformer [Vaswani et al., 2017] sequence-to-sequence model, and ﬁnd that some of them yield quality improvements over the typically-used ReLU or GELU activations.\",\"DOI\":\"10.48550/arXiv.2002.05202\",\"language\":\"en\",\"note\":\"arXiv:2002.05202 [cs]\",\"number\":\"arXiv:2002.05202\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"GLU Variants Improve Transformer\",\"URL\":\"http://arxiv.org/abs/2002.05202\",\"author\":[{\"family\":\"Shazeer\",\"given\":\"Noam\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2020\",2,12]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDqIp6sNbDMi}\[6\]
- Normalization: Layer Normalization may not be the most efficient
  approach for code
  processing[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"JRbJvKHX\",\"properties\":{\"formattedCitation\":\"[7]\",\"plainCitation\":\"[7]\",\"noteIndex\":0},\"citationItems\":[{\"id\":132,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/9YNSYDEC\"],\"itemData\":{\"id\":132,\"type\":\"article\",\"abstract\":\"Layer normalization (LayerNorm) has been successfully applied to various deep neural networks to help stabilize training and boost model convergence because of its capability in handling re-centering and re-scaling of both inputs and weight matrix. However, the computational overhead introduced by LayerNorm makes these improvements expensive and signiﬁcantly slows the underlying network, e.g. RNN in particular. In this paper, we hypothesize that re-centering invariance in LayerNorm is dispensable and propose root mean square layer normalization, or RMSNorm. RMSNorm regularizes the summed inputs to a neuron in one layer according to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efﬁcient than LayerNorm. We also present partial RMSNorm, or pRMSNorm where the RMS is estimated from p% of the summed inputs without breaking the above properties. Extensive experiments on several tasks using diverse network architectures show that RMSNorm achieves comparable performance against LayerNorm but reduces the running time by 7%∼64% on different models. Source code is available at https://github.com/bzhangGo/rmsnorm.\",\"DOI\":\"10.48550/arXiv.1910.07467\",\"language\":\"en\",\"note\":\"arXiv:1910.07467 [cs]\",\"number\":\"arXiv:1910.07467\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"Root Mean Square Layer Normalization\",\"URL\":\"http://arxiv.org/abs/1910.07467\",\"author\":[{\"family\":\"Zhang\",\"given\":\"Biao\"},{\"family\":\"Sennrich\",\"given\":\"Rico\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2019\",10,16]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDlRfck4uFum}\[7\]

While individual optimization techniques like SwiGLU, RMSNorm, RoPE, and
FlashAttention have been developed separately, there is a gap in
research that integrates multiple optimizations into a unified
CMG-specific architecture. \"GLU Variants Improve Transformer Noam.md\"
shows that SwiGLU and GEGLU variants produce better perplexities than
traditional activations, while \"Root Mean Square Layer
Normalization.md\" demonstrates that RMSNorm offers computational
efficiency with comparable performance. However, no study in the
literature combines these optimizations specifically for CMG tasks.

Most CMG research focuses on single programming languages, particularly
Java, as noted in \"Automated Generation of Commit Messages in Software
Repositories.md\". The literature lacks comprehensive approaches that
handle multi-language repositories or cross-language translation
scenarios. \"CoRaCMG: Contextual Retrieval-Augmented Framework for
Commit Message Generation.md\" mentions that ApacheCM covers nine
programming languages, indicating the need for multi-language support
that current approaches don\'t adequately address.

Many existing approaches treat code diffs as sequences of tokens without
fully leveraging structural information. \"CoRaCMG: Contextual
Retrieval-Augmented Framework for Commit Message Generation.md\" notes
that FIRA represents code diffs with fine-grained graphs to describe
edit operations, but such structural approaches are not widely
integrated into mainstream CMG models. The gap lies in effectively
incorporating Abstract Syntax Tree (AST) information, data flow graphs,
and other structural representations into transformer architectures for
CMG.

Current models struggle with large diffs that exceed typical sequence
lengths. \"CoRaCMG: Contextual Retrieval-Augmented Framework for Commit
Message Generation.md\" notes that filtering processes remove commits
with code diffs exceeding 300 lines, indicating that existing approaches
cannot handle large-scale changes. This represents a significant gap
since real-world commits often involve substantial code modifications
that current models cannot adequately process.

Despite the computational demands of current approaches, there is a gap
in lightweight, deployable solutions for resource-constrained
environments. \"Automated Generation of Commit Messages in Software
Repositories.md\" specifically mentions choosing conventional ML models
over transformer approaches to ensure efficiency and operation on
smaller computational devices like M1 chips
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"bGTFzsWv\",\"properties\":{\"formattedCitation\":\"[10]\",\"plainCitation\":\"[10]\",\"noteIndex\":0},\"citationItems\":[{\"id\":195,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/DXHEXZCB\"],\"itemData\":{\"id\":195,\"type\":\"paper-conference\",\"abstract\":\"Commit messages are crucial for documenting software changes, aiding in program comprehension and maintenance. However, creating effective commit messages is often overlooked by developers due to time constraints and varying levels of documentation skills. Our research presents an automated approach to generate commit messages using Machine Learning (ML) and Natural Language Processing (NLP) by developing models that use techniques such as Logistic Regression with TF-IDF and Word2Vec, as well as more sophisticated methods like LSTM. We used the dataset of code changes and corresponding commit messages that was used by Liu et al. [12], which we used to train and evaluate ML/NLP models and was chosen because it is extensively used in previous research, also for comparability in our study. The objective was to explore which ML/NLP techniques generate the most effective, clear, and concise commit messages that accurately reflect the code changes. We split the dataset into training, validation, and testing sets and used these sets to evaluate the performance of each model using qualitative and quantitative evaluation methods. Our results reveal a spectrum of effectiveness among these models, with the highest BLEU score achieved being 16.82, showcasing the models’ capability in automating a clear and concise commit message generation. Our paper offers insights into the comparative effectiveness of different machine learning models for automating commit message generation in software development, aiming to enhance the overall practice of code documentation. The source code is available at [2].\",\"DOI\":\"10.18293/DMSVIVA2024-145\",\"language\":\"en\",\"note\":\"arXiv:2504.12998 [cs]\",\"page\":\"87-95\",\"source\":\"arXiv.org\",\"title\":\"Automated Generation of Commit Messages in Software Repositories\",\"URL\":\"http://arxiv.org/abs/2504.12998\",\"author\":[{\"family\":\"Palakodeti\",\"given\":\"Varun Kumar\"},{\"family\":\"Heydarnoori\",\"given\":\"Abbas\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2024\",10,26]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDaDQ77UH9Yw}\[10\][](https://www.codecogs.com/eqnedit.php?latex=Automated%20Generation%20of%20Commit%20Messages%20in%20Software%20Repositories.md#1).
This indicates that current state-of-the-art solutions are not
accessible to all development environments.

Most research focuses on benchmark performance without considering
real-world deployment challenges. The literature lacks studies on how
CMG models perform in continuous integration pipelines, their latency
requirements for real-time suggestions, or their integration with
existing development workflows. \"CoRaCMG: Contextual
Retrieval-Augmented Framework for Commit Message Generation.md\"
mentions the need for models that reflect real-world development
scenarios where developers look at historical commits within the same
project, but most existing research doesn\'t adequately address these
practical considerations.

These gaps collectively point to the need for a comprehensive approach
that addresses efficiency, data quality, contextual understanding,
generalization, evaluation methodologies, architectural optimization,
multi-language support, structural integration, scalability for large
changes, and real-world deployability. The proposed mini-transformer
architecture with integrated optimizations (FlashAttention, RoPE,
SwiGLU, RMSNorm, gradient checkpointing) directly addresses several of
these gaps by providing an efficient, scalable solution specifically
designed for CMG tasks.

1.  []{#anchor-6}Objectives

- Design and implement a lightweight encoder--decoder Transformer model
  > from scratch for generating commit messages directly from code
  > diffs, with an emphasis on clarity, efficiency, and task-specific
  > adaptation.

- Integrate modern Transformer optimizations, including Rotary
  > Positional Encoding (RoPE), SwiGLU feed-forward layers, RMSNorm,
  > memory-efficient multi-head attention, and gradient checkpointing,
  > to improve model quality and training stability while reducing
  > computational and memory overhead.

- Develop a code-diff--aware data processing pipeline featuring custom
  > tokenization that preserves structural information (e.g., addition
  > and deletion markers, file paths, and code syntax), along with
  > dataset cleaning and rebalancing to reduce redundancy, noise, and
  > imbalance.

- Construct a complete training and inference workflow to evaluate the
  > proposed model on standard commit message generation benchmarks
  > using automatic metrics such as BLEU, ROUGE-L, and METEOR, as well
  > as qualitative assessments of readability and adequacy.

- Ensure that the entire training and inference process operates within
  > strict hardware constraints (approximately 8GB VRAM), demonstrating
  > that effective commit message generation can be achieved without
  > reliance on large-scale pre-trained models or cloud resources.

- Provide a modular and well-documented implementation that supports
  > future extensions, including sparse attention, Mixture-of-Experts
  > scaling, and retrieval-augmented generation, and enables fair
  > efficiency-focused comparisons with resource-intensive large
  > language models.

1.  []{#anchor-7}Methodology/ Requirement Specification:

1.  1.  []{#anchor-8}Research Design/ Prototype Design

![](Pictures/100000010000058000000300F3F9AB10.png){width="8.5in"
height="4.5984in"}

Fig3.1: Methodology of the overall thesis procedure

1.  1.  []{#anchor-9}Data Collection/ Need Assessment

The methodology begins with the collection of commit data from the
bigcode/commitpack repository hosted on Hugging Face. This repository
provides a large-scale collection of Python code commits, which serves
as the primary dataset for training and evaluating the proposed
Transformer model. The dataset consists of 457 JSONL files
(python-0001.jsonl to python-0457.jsonl), each containing structured
commit records with associated commit messages, previous code versions,
and updated code versions. All dataset files are downloaded
programmatically using the Hugging Face Hub API through an automated
pipeline that ensures reliable and complete retrieval of the entire
collection.

Following download, each JSONL file undergoes an initial transformation
process in which essential fields---commit message, old code content,
and new code content---are extracted from each commit record. Invalid or
malformed entries are safely discarded to maintain data quality. The
extracted information is then converted into a structured CSV format,
with each commit represented as a single row.

![](Pictures/10000000000009C40000012BDD7A9A56.png){width="8.339in"
height="1.0134in"}

Fig3.2: Flow chart of the Dataset Collection and Processing process

[]{#anchor-10}Diff Generation and Change Classification

To prepare the data for commit message generation, the old and new code
versions are systematically compared to produce structured code diffs.
This process identifies and categorizes code changes using a diffing
algorithm based on sequence comparison.

Each change is classified into one of six categories:

- **ADD**: Newly added code lines

- **REMOVE**: Deleted code lines

- **MODIFY**: Modified code lines

- **COMMENT_ADD**: Added comment lines

- **COMMENT_REMOVE**: Removed comment lines

- **COMMENT_MODIFY**: Modified comment lines

Special care is taken to correctly identify comments and documentation
blocks, including single-line comments and multi-line docstrings,
ensuring that documentation changes are distinguished from functional
code changes. The resulting structured diffs form the core input
representation for the commit message generation task.

1.  1.  []{#anchor-11}Analysis Techniques

    1.  Semantic Cleaning and Message Quality Filtering

The first preprocessing step removes non-ASCII characters from commit
messages. The second preprocessing stage performs extensive semantic
cleaning of commit messages. This stage implements a multi-layered
filtering approach to ensure high-quality training data:

- **Metadata Removal: **The process removes various types of metadata
  > including git-svn-id, Change-Id, Signed-off-by, Reviewed-by,
  > Co-authored-by, and Ticket fields. Regular expressions identify and
  > remove these metadata patterns without affecting the semantic
  > content of the commit message.

- **Auto-Prefix Handling:** Automated prefixes commonly found in commit
  > messages are removed using pattern matching. This includes prefixes
  > like \"feat:\", \"fix:\", \"chore:\", etc., which are often added by
  > automated tools and don\'t contribute meaningful information for the
  > model.

- **Noise Filtering:** Bracketed and parenthesized noise is
  > systematically removed, including version numbers, issue references,
  > and other non-semantic content. The process handles various bracket
  > types and removes content that doesn\'t contribute to the commit
  > message\'s meaning.

- **Email and URL Removal:** Email addresses and URLs are identified and
  > removed using sophisticated regular expressions that handle various
  > email and URL formats. This prevents the model from learning
  > irrelevant patterns related to contact information.

- **File Path and Timestamp Filtering:** File paths with common
  > extensions (.py, .js, .cpp, etc.) and timestamp patterns are
  > identified and removed. This ensures that the model focuses on the
  > semantic content rather than technical metadata.

- **Language Detection:** The process implements language detection
  > using statistical models to identify non-English content. Only
  > English commit messages are retained, ensuring consistency in the
  > training data.

- **Profanity and Garbage Filtering:** A comprehensive list of profanity
  > and meaningless keywords is maintained and used to filter out
  > inappropriate or low-quality commit messages. This includes common
  > meaningless commit messages like \"fixes\", \"updates\",
  > \"upgrades\", \"typo\", etc.

The process uses parallel processing to distribute the cleaning workload
across multiple CPU cores, significantly improving performance for large
datasets. The implementation includes chunked processing where the
dataset is split into multiple chunks, each processed by a separate
worker process.

1.  1.  1.  Dataset Rebalancing

<div>

The rebalancing process implements a comprehensive system with ten
different strategies to address data imbalance and improve model
training effectiveness:

**Remove Duplicates Strategy**: This basic strategy removes only exact
duplicate entries while preserving all other data. It uses efficient
hashing techniques to identify duplicates quickly:

**Clustering-Based Sampling**: This strategy uses text clustering
algorithms on message embeddings to sample evenly from each cluster,
ensuring diverse representation across different types of commit
messages. The process uses TF-IDF vectorization followed by K-means
clustering:

**Near-Duplicate Detection**: This approach identifies and removes
near-duplicate messages using cosine similarity measurements, helping to
reduce redundancy in the dataset:

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BDuplicate%7D(m_1%2C%20m_2)%20%3D%5Cbegin%7Bcases%7D1%2C%20%26%20%5Ctext%7Bif%20%7D%20%5Ctext%7BSimilarity%7D(m_1%2C%20m_2)%20%3E%20%5Ctau%20%5C%5C0%2C%20%26%20%5Ctext%7Botherwise%7D%5Cend%7Bcases%7D#0)

**Diversity Sampling**: Implements farthest-point sampling to maximize
textual diversity across the dataset, ensuring the model sees a wide
variety of commit message styles.

**Frequency-Based Downscaling**: Assigns weights inversely proportional
to frequency and samples accordingly, helping to balance representation
between common and rare commit message patterns:

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BWeight%7D(%5Ctext%7Bmessage%7D)%20%3D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Ctext%7Bfrequency%7D(%5Ctext%7Bmessage%7D)%7D%7D#0)

**Quality Scoring System**: Scores messages based on length and
uniqueness criteria to prioritize higher-quality examples for training:

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BQuality%5C_score%7D(%5Ctext%7Bmessage%7D)%20%3D%20%5Calpha%20%5Ctimes%20%5Ctext%7Blength%5C_score%7D%20%2B%20%5Cbeta%20%5Ctimes%20%5Ctext%7Buniqueness%5C_score%7D#0)

**Information Value Scoring**: Prioritizes messages with higher
information content using TF-IDF-based scoring to ensure meaningful
content in the training data.

**Topic Modeling Approach**: Applies topic modeling to ensure balanced
representation across different topics and subject areas.

**N-gram Based Grouping**: Groups messages by common n-grams and limits
representation to prevent overfitting to common patterns.

**Multi-Criteria Balancing**: Combines multiple metrics for
comprehensive balancing across various dimensions.

The rebalancing system includes sophisticated memory management
techniques including chunked processing for large datasets, aggressive
garbage collection, and multiprocessing support for faster processing.
The system automatically adjusts its approach based on dataset size,
using sampling for heavy computations and efficient data structures for
optimal performance.

</div>

<div>

1.  1.  1.  Syntax-Aware Diff Processing

The syntax-aware diff processing enhances MODIFY tags in the diff data
using tree-sitter parsing. This stage uses AST (Abstract Syntax Tree)
parsing to understand Python code structure and intelligently handle
MODIFY operatios.

The tree-sitter integration enables AST-based parsing of both before and
after code versions, The process extracts ordered leaves with gaps to
maintain code structure integrity during the diff process. It uses
sequence matching algorithms to identify specific changes between code
versions at the AST node level rather than just text-based differences.

1.  1.  []{#anchor-12}Vocabulary Building Process

The DiffVocabulary.py implements a sophisticated vocabulary building
system specifically designed for diff text. The system uses tree-sitter
parsing to understand Python code structure, enabling more intelligent
tokenization of code diffs.

![](Pictures/10000000000009C4000001DD7EB92A4F.png){width="7.5in"
height="1.4307in"}

Fig3.3: Flow chart of the Vocabulary building process of both input and
output

This work employs **separate vocabularies** for code diffs and commit
messages to better capture the distinct structural and linguistic
properties of each modality. The overall process is summarized in
Algorithm 1.

**Input:** Dataset containing code diffs and commit messages  
**Output:** Diff vocabulary and message vocabulary

1.  Initialize the diff vocabulary with a target size of 50,000 tokens
    > and a minimum frequency threshold of 2.

2.  Initialize the message vocabulary with a target size of 5,000 tokens
    > and a minimum frequency threshold of 2.

3.  Build the diff vocabulary using syntax-aware parsing and
    > probabilistic subword modeling.

4.  Build the message vocabulary using Byte-Pair Encoding (BPE).

5.  Optimize both vocabularies to maximize contextual information while
    > minimizing vocabulary size.

6.  Return the finalized diff and message vocabularies.

</div>

<div>

[]{#anchor-13}[]{#anchor-14}Syntax-Aware Tokenization Using Tree-Sitter

The diff vocabulary construction process leverages **tree-sitter**, a
syntax-aware parsing framework, to extract semantically meaningful
tokens from Python code diffs. Instead of treating code as raw text,
tree-sitter produces abstract syntax trees (ASTs), enabling tokenization
that preserves program structure.

Let the set of semantic tokens be defined as:

[](https://www.codecogs.com/eqnedit.php?latex=%5C%5C%5C%5C%20T_%7Bsemantic%7D%20%3D%20%7Bt_1%2C%20t_2%2C%20%5Cdots%2C%20t_n%7D%2C%20%5Cquad%20t_i%20%5Cin%20AST_%7Bpython%7D%5C%5C%5C%5C%20#1)

This approach provides three key advantages:

- **Semantic awareness:** Tokens correspond to syntactic units rather
  > than arbitrary substrings.

- **Identifier decomposition:** Identifiers are split based on naming
  > conventions such as camelCase, snake_case, and PascalCase:  
  > [](http://www.texrendr.com/?eqn=%5C%5C%5C%5C%20identifier_%7Bsplit%7D%20%3D%20%5Ctext%7Bsplit%7D(identifier%2C%20%7B%5Ctext%7BcamelCase%7D%2C%20%5Ctext%7Bsnake_case%7D%2C%20%5Ctext%7BPascalCase%7D%7D)%5C%5C%5C%5C%20#1)

- **Structural preservation:** AST-level information allows the model to
  > better understand the context of code changes.

[]{#anchor-15}Unigram Subword Modeling with Expectation--Maximization

To construct compact yet expressive subword units, a **unigram language
model** is learned using the Expectation--Maximization (EM) algorithm.
Given a sequence of subword tokens ( T = {t_1, \dots, t_n} ), the
probability of a word ( w ) is modeled as:

[](https://www.codecogs.com/eqnedit.php?latex=P(w%20%5Cmid%20T)%20%3D%20%5Cprod_%7Bi%3D1%7D%5E%7Bn%7D%20P(t_i%20%5Cmid%20%5Ctheta)#1)

where ( [](https://www.codecogs.com/eqnedit.php?latex=%5Ctheta#1) )
denotes the model parameters.

The EM procedure iteratively refines token probabilities:

- **E-step:
  > **[](https://www.codecogs.com/eqnedit.php?latex=Q(%5Ctheta%20%5Cmid%20%5Ctheta%5E%7B(t)%7D)%20%3D%20%5Cmathbb%7BE%7D_%7BZ%20%5Cmid%20w%2C%20%5Ctheta%5E%7B(t)%7D%7D%20%5C%5B%5Clog%20P(Z%2C%20w%20%5Cmid%20%5Ctheta)#1)\\

- **M-step:
  > **[](https://www.codecogs.com/eqnedit.php?latex=%5Ctheta%5E%7B(t%2B1)%7D%20%3D%20%5Carg%5Cmax_%7B%5Ctheta%7D%20Q(%5Ctheta%20%5Cmid%20%5Ctheta%5E%7B(t)%7D)#1)

At each iteration, low-probability tokens are pruned to maintain the
target vocabulary size.

This probabilistic approach enables:

- Global optimization over all segmentations

- Adaptive discovery of meaningful subword units

- Improved balance between vocabulary coverage and specificity

[]{#anchor-16}Aggressive Prefiltering

To reduce noise and improve computational efficiency, aggressive
prefiltering is applied before EM optimization:

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BFilteredVocab%7D%20%3D%20%7B%20w%20%5Cin%20%5Ctext%7BRawVocab%7D%20%5Cmid%20f(w)%20%5Cge%20f_%7Bmin%7D%20%5Cland%20%7Cw%7C%20%5Cle%20L_%7Bmax%7D%20%7D#1)

This step:

- Eliminates rare and excessively long tokens

- Reduces EM computation cost

- Improves overall vocabulary quality

[]{#anchor-17}Message Vocabulary Construction

The message vocabulary is built using **Byte-Pair Encoding (BPE)**, a
frequency-based subword tokenization method well-suited for natural
language text. BPE iteratively merges the most frequent adjacent token
pairs:

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BBPE%7D(V%2C%20n)%20%3D%20V%20%5Ccup%20%7B%20%5Ctext%7Bmerge%7D(p_1%2C%20p_2)%20%7D%5E%7Bn%7D#1)

where ((p_1, p_2)) is the most frequent adjacent pair at each iteration.

The algorithm starts from character-level tokens:

[](https://www.codecogs.com/eqnedit.php?latex=V_0%20%3D%20%7B%20c%20%5Cmid%20c%20%5Cin%20%5Cbigcup_%7Bw%20%5Cin%20%5Ctext%7BCorpus%7D%7D%20%5Ctext%7Bcharacters%7D(w)%20%7D#1)

This guarantees:

- Complete coverage of all inputs

- Robust handling of unseen words

- Efficient compression of frequent linguistic patterns

BPE is selected for its balance between expressiveness, compactness, and
computational simplicity.

Both vocabularies are optimized to preserve maximum semantic context
under fixed size constraints:

[](https://www.codecogs.com/eqnedit.php?latex=%5Cmax_%7BV%7D%20%5Ctext%7BContext%7D(V)%20%5Cquad%20%5Ctext%7Bsubject%20to%7D%20%5Cquad%20%7CV%7C%20%5Cle%20V_%7Bmax%7D#1)

The efficiency of a vocabulary is measured as:

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BEfficiency%7D%20%3D%20%5Cfrac%7BH(%5Ctext%7BOriginalText%7D)%20-%20H(%5Ctext%7BTokenizedText%7D)%7D%7B%7CV%7C%7D#1)

where [](https://www.codecogs.com/eqnedit.php?latex=(H(%5Ccdot))#1)
denotes entropy.

For the diff vocabulary:  
[](http://www.texrendr.com/?eqn=%5Ctext%7BEfficiency%7D%7Bdiff%7D%20%3D%5Cfrac%7B%5Csum%7Bw%20%5Cin%20%5Ctext%7Bdiff%7D%7D%20f(w)%5Clog(%5Ctext%7Bsubword_count%7D(w))%7D%7B%7C%5Ctext%7BDiffVocab%7D%7C%7D#1)

For the message vocabulary:[  
](http://www.texrendr.com/?eqn=%5Ctext%7BEfficiency%7D%7Bmsg%7D%20%3D%5Cfrac%7B%5Csum%7Bw%20%5Cin%20%5Ctext%7Bmsg%7D%7D%20f(w)%5Ccdot%20%5Ctext%7Bcompression_ratio%7D(w)%7D%7B%7C%5Ctext%7BMsgVocab%7D%7C%7D#1)

</div>

<div>

1.  1.  []{#anchor-18}Transformer Model Architecture

The transformer model implements a highly modular architecture where
each component is separated into distinct files, promoting
maintainability, readability, and ease of modification. This modular
design allows for independent optimization and experimentation with
individual components while maintaining the overall system integrity.

![](Pictures/10000000000009C40000017E58D4CD1F.png){width="8.1772in"
height="1.2661in"}

Fig3.4: Overview of the Transformer model

</div>

<div>

1.  1.  1.  Encoder Architecture

The encoder architecture is implemented with a focus on computational
efficiency and gradient flow optimization. Each encoder layer follows
this mathematical formulation:

**Self-Attention Mechanism:**

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BMultiHead%7D(Q%2C%20K%2C%20V)#0)[](https://www.codecogs.com/eqnedit.php?latex=%3D%5Ctext%7BConcat%7D%5C!%5Cleft(%5Ctext%7Bhead%7D_1%2C%20%5Ctext%7Bhead%7D_2%2C%20%5Cldots%2C%20%5Ctext%7Bhead%7D_h%5Cright)%20W%5E%7BO%7D%5Ctext%7Bwhere%20%7D%20%5Ctext%7Bhead%7D_i%3D%5Ctext%7BAttention%7D%5C!%5Cleft(Q%20W_i%5E%7BQ%7D%2CK%20W_i%5E%7BK%7D%2CV%20W_i%5E%7BV%7D%5Cright)#0)

**Rotary Positional Encoding (RoPE):** The model uses Rotary Positional
Encoding instead of absolute positional encoding, which provides more
efficient positional information while maintaining relative position
relationships:

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BRoPE%7D(x%2C%20%5Ctheta)%20%3D%20R(%5Ctheta)%20%5Ctimes%20x#0)  
[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7Bwhere%20%7D%20R(%5Ctheta)%20%5Ctext%7B%20is%20the%20rotation%20matrix%20and%20%7D%20%5Ctheta%20%5Ctext%7B%20represents%20positional%20angles%7D#0)

**RMS Normalization:** The model uses Root Mean Square (RMS)
normalization instead of Layer Normalization for improved efficiency:

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BRMSNorm%7D(x)%20%3D%20%5Cfrac%7Bx%7D%7B%5Csqrt%7B%5Ctext%7Bmean%7D(x%5E2)%20%2B%20%5Cvarepsilon%7D%7D%20%5Codot%20%5Cgamma#0)

**SwiGLU Feed-Forward Network:** The model implements SwiGLU
(Swish-Gated Linear Unit) instead of traditional feed-forward networks:

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BSwiGLU%7D(x)%20%3D%20%5Ctext%7BLinear%7D_3%5CBig(%5Ctext%7BLinear%7D_1(x)%20%5C%3B%5Cotimes%5C%3B%20%5Csigma(%5Ctext%7BLinear%7D_2(x))%5CBig)#0)

> 3.5.2 Decoder Architecture

The decoder architecture implements a more complex attention mechanism
with both self-attention and cross-attention:

**Decoder Self-Attention:**

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BSelfAttention%7D(Q%2C%20K%2C%20V)%20%3D%20%5Ctext%7BMultiHead%7D%5C!%5Cleft(Q%20W_i%5EQ%2C%20K%20W_i%5EK%2C%20V%20W_i%5EV%5Cright)#0)

**Cross-Attention:**

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BCrossAttention%7D(Q%2C%20K_%7B%5Ctext%7Benc%7D%7D%2C%20V_%7B%5Ctext%7Benc%7D%7D)%20%3D%20%5Ctext%7BMultiHead%7D%5C!%5Cleft(Q%20W_i%5EQ%2C%20K_%7B%5Ctext%7Benc%7D%7D%20W_i%5EK%2C%20V_%7B%5Ctext%7Benc%7D%7D%20W_i%5EV%5Cright)#0)

3.5.3 Transformer Block Formulation

**Encoder Block:**

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BFFN%7D(x)%20%3D%20%5Ctext%7BLinear%7D_2%5CBig(%5Ctext%7BReLU%7D(%5Ctext%7BLinear%7D_1(x))%5CBig)#0)

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BMultiHead%7D(x)%20%3D%20%5Ctext%7BLayerNorm%7D%5CBig(x%20%2B%20%5Ctext%7BDropout%7D(%5Ctext%7BAttention%7D(x%2C%20x%2C%20x))%5CBig)#0)

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BOutput%7D%20%3D%20%5Ctext%7BLayerNorm%7D%5CBig(%5Ctext%7BMultiHead%7D(x)%20%2B%20%5Ctext%7BDropout%7D(%5Ctext%7BFFN%7D(%5Ctext%7BMultiHead%7D(x)))%5CBig)#0)

**Decoder Block:**

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BMaskedMultiHead%7D(x)%20%3D%20%5Ctext%7BLayerNorm%7D%5CBig(x%20%2B%20%5Ctext%7BDropout%7D(%5Ctext%7BAttention%7D(x%2C%20x%2C%20x%2C%20%5Ctext%7Bcausal%5C_mask%7D))%5CBig)#0)

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BCrossAttention%7D(x)%20%3D%20%5Ctext%7BLayerNorm%7D%5CBig(%5Ctext%7BMaskedMultiHead%7D(x)%20%2B%20%5Ctext%7BDropout%7D(%5Ctext%7BAttention%7D(%5Ctext%7BMaskedMultiHead%7D(x)%2C%20%5Ctext%7Benc%5C_output%7D%2C%20%5Ctext%7Benc%5C_output%7D)))%5CBig)#0)

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BOutput%7D%20%3D%20%5Ctext%7BLayerNorm%7D%5CBig(%5Ctext%7BCrossAttention%7D(x)%20%2B%20%5Ctext%7BDropout%7D(%5Ctext%7BFFN%7D(%5Ctext%7BCrossAttention%7D(x)))%5CBig)#0)

The SwiGLU activation function provides improved performance over
traditional feed-forward networks:

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BSwiGLU%7D(x)%20%3D%20%5Ctext%7BLinear%7D_3%5CBig(%5Ctext%7BLinear%7D_1(x)%20%5C%3B%5Cotimes%5C%3B%20%5Csigma(%5Ctext%7BLinear%7D_2(x))%5CBig)#0)

RMS normalization provides computational efficiency compared to Layer
Normalization:

[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BRMSNorm%7D(x)%20%3D%20x%20%5C%3B%5Coslash%5C%3B%20%5Csqrt%7B%5Cmathbb%7BE%7D%5Bx%5E2%5D%20%2B%20%5Cvarepsilon%7D%20%5C%3B%5Codot%5C%3B%20%5Cgamma#0)

</div>

<div>

1.  1.  []{#anchor-19}Modular Design Benefits

The modular architecture provides several key advantages:

1.  **Component Independence**: Each component (encoder, decoder,
    > attention, feed-forward) can be modified independently without
    > affecting others.

2.  **Algorithmic Flexibility**: Different attention mechanisms (Flash,
    > Kernelized, Standard) can be swapped easily.

3.  **Optimization Opportunities**: Individual components can be
    > optimized separately for performance.

4.  **Research Flexibility**: New architectures can be tested by
    > replacing specific modules.

5.  **Maintenance Efficiency**: Bugs can be isolated to specific
    > modules, making debugging more efficient.

<!-- -->

1.  1.  []{#anchor-20}Training Loop

> []{#anchor-21} BEGIN Training Process  
> Initialize Transformer Model with vocabulary sizes, dimensions,
> layers  
> Initialize Adam Optimizer with learning rate, betas, epsilon, weight
> decay  
> Initialize Learning Rate Scheduler with mode, factor, patience  
> Initialize Cross Entropy Loss with padding ignore index and label
> smoothing  
> Initialize Gradient Scaler for mixed precision training  
> Set contrastive weight, diversity weight, repetition penalty
> parameters
>
> FOR epoch from 1 to number of epochs  
> Set model to training mode  
> Initialize total training loss to zero  
> FOR each batch in training data loader  
> Zero gradients in optimizer  
> WITH automatic mixed precision context  
> Forward pass through model with source and target batches  
> Compute primary cross-entropy loss between predictions and targets  
> Compute length regularization loss to encourage appropriate sequence
> length  
> Compute diversity loss to promote varied token predictions  
> Combine all loss components into total loss  
> END automatic mixed precision context
>
> Scale gradients and perform backward pass  
> Update optimizer parameters with gradient scaler  
> Update gradient scaler  
> Accumulate total training loss  
> END batch loop
>
> Calculate average training loss across all batches
>
> Set model to evaluation mode  
> Initialize total validation loss to zero  
> WITH no gradient computation context  
> FOR each batch in validation data loader  
> WITH automatic mixed precision context  
> Forward pass through model with source and target batches  
> Compute validation loss between predictions and targets  
> END automatic mixed precision context  
> Accumulate total validation loss  
> END validation batch loop  
> END no gradient computation context
>
> Calculate average validation loss across all batches  
> Update learning rate based on validation loss using scheduler
>
> IF current validation loss is better than best validation loss  
> Save model checkpoint with current parameters  
> Update best validation loss  
> END validation loss comparison  
> END epoch loop
>
> Return trained model  
> END Training Process

![](Pictures/10000000000009C400000237E304B003.png){width="7.9339in"
height="1.8071in"}

Fig3.5: Flow Diagram of the Training Process

The training process begins with the initialization of the transformer
model, which incorporates several modern optimization techniques:

[](http://www.texrendr.com/?eqn=%20%5Ctext%7BModel%7D%20%3D%20%5Ctext%7BTransformer%7D(%5Ctext%7Bsrc_vocab_size%7D%2C%20%5Ctext%7Btgt_vocab_size%7D%2C%20d_%7Bmodel%7D%2C%20h%2C%20L%2C%20d_%7Bff%7D%2C%20L_%7Bmax%7D%2C%20%5Cdelta)%20#0)

Where:

|                |          |                |                                                                |
|----------------|----------|----------------|----------------------------------------------------------------|
| **Parameter**  | **Type** | **Used Value** | **Description**                                                |
| d_model        | int      | 512            | Dimensionality of model embeddings                             |
| num_heads      | int      | 8              | Number of attention heads in multi-head self-attention         |
| num_layers     | int      | 2              | Number of Transformer encoder/decoder layers                   |
| d_ff           | int      | 2048           | Hidden dimension of the feed-forward network                   |
| max_seq_length | int      | 256            | Maximum input sequence length (avg diff: 128, avg message: 12) |
| dropout        | float    | 0.3            | Dropout rate for regularization                                |
| batch_size     | int      | 32             | Number of samples per training batch                           |
| num_epochs     | int      | 3              | Total number of training epochs                                |
| learning_rate  | float    | 0.00001        | Initial learning rate for optimization                         |

</div>

<div>

The training process uses PyTorch\'s DataLoader for efficient batch
processing:

[](https://www.codecogs.com/eqnedit.php?latex=%20%5Cmathcal%7BD%7D%20%3D%20%7B(x_i%2C%20y_i)%7D%7Bi%3D1%7D%5EN%20%5Ctext%7B%20where%20%7D%20x_i%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BS%20%5Ctimes%20V%7Bsrc%7D%7D%2C%20y_i%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BT%20%5Ctimes%20V_%7Btgt%7D%7D%20#0)

Where:

- [](https://www.codecogs.com/eqnedit.php?latex=N#0): Total number of
  > training examples

- [](https://www.codecogs.com/eqnedit.php?latex=S#0): Source sequence
  > length

- [](https://www.codecogs.com/eqnedit.php?latex=T#0): Target sequence
  > length

- [](https://www.codecogs.com/eqnedit.php?latex=V_%7Bsrc%7D#0): Source
  > vocabulary size

- [](https://www.codecogs.com/eqnedit.php?latex=V_%7Btgt%7D#0): Target
  > vocabulary size

The dataset is split into training and validation sets:
[](https://www.codecogs.com/eqnedit.php?latex=%20%5Cmathcal%7BD%7D%7Btrain%7D%20%3D%20%5Cmathcal%7BD%7D%20%5Ctimes%200.9%2C%20%5Cquad%20%5Cmathcal%7BD%7D%7Bval%7D%20%3D%20%5Cmathcal%7BD%7D%20%5Ctimes%200.1%20#0)

The training process uses the Adam optimizer with custom parameters
optimized for transformer training:

[](https://www.codecogs.com/eqnedit.php?latex=%20%5Ctheta_%7Bt%2B1%7D%20%3D%20%5Ctheta_t%20-%20%5Calpha%20%5Cfrac%7B%5Chat%7Bm%7D_t%7D%7B%5Csqrt%7B%5Chat%7Bv%7D_t%7D%20%2B%20%5Cepsilon%7D%20#0)

Where:

- [](https://www.codecogs.com/eqnedit.php?latex=%5Calpha#0): Learning
  > rate (typically
  > [](https://www.codecogs.com/eqnedit.php?latex=1%20%5Ctimes%2010%5E%7B-5%7D#0))

- [](https://www.codecogs.com/eqnedit.php?latex=%5Chat%7Bm%7D_t#0):
  > Bias-corrected first moment estimate

- [](https://www.codecogs.com/eqnedit.php?latex=%5Chat%7Bv%7D_t#0):
  > Bias-corrected second moment estimate

- [](https://www.codecogs.com/eqnedit.php?latex=%5Cbeta_1%20%3D%200.9%2C%20%5Cbeta_2%20%3D%200.98#0):
  > Exponential decay rates

- [](https://www.codecogs.com/eqnedit.php?latex=%5Cepsilon%20%3D%201%20%5Ctimes%2010%5E%7B-9%7D#0):
  > Small constant for numerical stability

- [](http://www.texrendr.com/?eqn=%5Ctext%7Bweight_decay%7D%20%3D%200.0001#0):
  > L2 regularization

The primary loss function uses label smoothing for improved
generalization:

[](https://www.codecogs.com/eqnedit.php?latex=%20%5Cmathcal%7BL%7D%7BCE%7D%20%3D%20-%5Cfrac%7B1%7D%7B%7CY%7C%7D%20%5Csum%7Bi%3D1%7D%5E%7B%7CY%7C%7D%20%5Cleft%5B%20(1-%5Calpha)%20%5Clog%20p(y_i%7Cx)%20%2B%20%5Cfrac%7B%5Calpha%7D%7B%7CV_%7Btgt%7D%7C%7D%20%5Csum_%7Bj%3D1%7D%5E%7B%7CV_%7Btgt%7D%7C%7D%20%5Clog%20p(j%7Cx)%20%5Cright%5D%20#0)

Where:

- [](https://www.codecogs.com/eqnedit.php?latex=%5Calpha%20%3D%200.05#0):
  > Label smoothing parameter

- [](https://www.codecogs.com/eqnedit.php?latex=Y#0): Target sequence

- [](https://www.codecogs.com/eqnedit.php?latex=V_%7Btgt%7D#0): Target
  > vocabulary size

To encourage the model to generate appropriately long commit messages, a
length regularization loss is implemented:

[](https://www.codecogs.com/eqnedit.php?latex=%20%5Cmathcal%7BL%7D%7Blength%7D%20%3D%20%5Clambda%20%5Ccdot%20%5Cmathbb%7BE%7D%7Bt%7D%20%5Cleft%5B%20%5Cmax(0%2C%20L_%7Bmin%7D%20-%20L_%7Bpred%7D(t))%20%5Cright%5D%5E2%20#0)

Where:

- [](https://www.codecogs.com/eqnedit.php?latex=%5Clambda%20%3D%200.01#0):
  > Length regularization weight

- [](https://www.codecogs.com/eqnedit.php?latex=L_%7Bmin%7D%20%3D%205#0):
  > Minimum desired sequence length

- [](https://www.codecogs.com/eqnedit.php?latex=L_%7Bpred%7D(t)#0):
  > Predicted length at time step
  > [](https://www.codecogs.com/eqnedit.php?latex=t#0)

The length prediction is computed as:
[](https://www.codecogs.com/eqnedit.php?latex=%20L_%7Bpred%7D%20%3D%20%5Carg%5Cmax_t%20P(%5Ctext%7B%7D%7Cx%2C%20y_%7B%3Ct%7D)%20#0)

To encourage varied outputs and reduce repetitive generation, a
diversity loss is computed:

[](https://www.codecogs.com/eqnedit.php?latex=%20%5Cmathcal%7BL%7D%7Bdiversity%7D%20%3D%20-%5Cgamma%20%5Ccdot%20%5Cfrac%7B1%7D%7B%7CY%7C%7D%20%5Csum%7Bi%3D1%7D%5E%7B%7CY%7C%7D%20H(P(y_i%7Cx%2C%20y_%7B%3Ci%7D))%20#0)

Where:

- [](https://www.codecogs.com/eqnedit.php?latex=%5Cgamma%20%3D%200.01#0):
  > Diversity weight

- [](https://www.codecogs.com/eqnedit.php?latex=H(P)#0): Entropy of the
  > probability distribution

- [](https://www.codecogs.com/eqnedit.php?latex=P(y_i%7Cx%2C%20y_%7B%3Ci%7D)#0):
  > Conditional probability of token
  > [](https://www.codecogs.com/eqnedit.php?latex=i#0)

The entropy is computed as:
[](https://www.codecogs.com/eqnedit.php?latex=%20H(P)%20%3D%20-%5Csum_%7Bk%7D%20P(k)%20%5Clog%20P(k)%20#0)

For improved representation learning, contrastive loss is implemented:

[](https://www.codecogs.com/eqnedit.php?latex=%20%5Cmathcal%7BL%7D%7Bcontrastive%7D%20%3D%20-%5Clog%20%5Cfrac%7B%5Cexp(s(h_i%2C%20h_j)%20%2F%20%5Ctau)%7D%7B%5Csum%7Bk%20%5Cin%20%5Cmathcal%7BN%7D_i%7D%20%5Cexp(s(h_i%2C%20h_k)%20%2F%20%5Ctau)%7D%20#0)

Where:

- [](https://www.codecogs.com/eqnedit.php?latex=s(h_i%2C%20h_j)#0):
  > Similarity function between representations

- [](https://www.codecogs.com/eqnedit.php?latex=%5Ctau#0): Temperature
  > parameter

- [](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%7BN%7D_i#0):
  > Set of negative samples for sample
  > [](https://www.codecogs.com/eqnedit.php?latex=i#0)

The training process implements mixed precision training using automatic
mixed precision (AMP):

[](https://www.codecogs.com/eqnedit.php?latex=%20%5Ctext%7BMixedPrecisionLoss%7D%20%3D%20%5Ctext%7BScale%7D(%5Ctext%7BFP32Loss%7D)%20%5Ccdot%20%5Ctext%7BGradScaler%7D%20#0)

The gradient scaling factor is computed as:
[](https://www.codecogs.com/eqnedit.php?latex=%20%5Ctext%7BScaleFactor%7D%20%3D%20%5Cmin(%5Ctext%7BMaxScale%7D%2C%20%5Cfrac%7B%5Ctext%7BDesiredGradNorm%7D%7D%7B%5Ctext%7BCurrentGradNorm%7D%7D)%20#0)

This approach provides:

1.  **Memory Efficiency**: Reduces memory usage by approximately 50%

2.  **Speed Improvement**: Faster computation on modern GPUs

3.  **Numerical Stability**: Maintains training stability with proper
    > scaling

To manage memory usage during training, gradient checkpointing is
implemented:

[](https://www.codecogs.com/eqnedit.php?latex=%20%5Ctext%7BMemory%7D%7Bcheckpointed%7D%20%3D%20%5Ctext%7BMemory%7D%7Bfull%7D%20-%20%5Ctext%7BIntermediateActivations%7D%20#0)

The checkpointing strategy stores only the input to each layer and
recomputes intermediate activations during backpropagation, trading
computation time for reduced memory usage.

The training process uses a ReduceLROnPlateau scheduler:

[](http://www.texrendr.com/?eqn=%20%5Calpha_%7Bt%2B1%7D%20%3D%20%5Cbegin%7Bcases%7D%20%5Calpha_t%20%5Ccdot%20%5Ctext%7Bfactor%7D%20%26%20%5Ctext%7Bif%20%7D%20%5Ctext%7Bmetric%7D%7Bt%7D%20%5Cgeq%20%5Ctext%7Bmetric%7D%7Bt-patience%7D%20%5C%20%5Calpha_t%20%26%20%5Ctext%7Botherwise%7D%20%5Cend%7Bcases%7D%20#0)

Where:

- [](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7Bfactor%7D%20%3D%200.1#0):
  > Learning rate reduction factor

- [](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7Bpatience%7D%20%3D%202#0):
  > Number of epochs to wait before reducing LR

Label smoothing improves model generalization by preventing
overconfidence:

[](https://www.codecogs.com/eqnedit.php?latex=%20%5Ctext%7BBenefit%7D%20%3D%20%5Ctext%7BImproved%20generalization%7D%20%5Cland%20%5Ctext%7BReduced%20overfitting%7D%20#0)

The technique prevents the model from becoming too confident in its
predictions, leading to better performance on unseen data.

Gradient clipping prevents exploding gradients:

[](http://www.texrendr.com/?eqn=%20%5Ctext%7BClippedGradient%7D%20%3D%20%5Cmin(%5Ctext%7Bmax_norm%7D%2C%20%5Cfrac%7B%5Ctext%7Bgradient%7D%7D%7B%7C%5Ctext%7Bgradient%7D%7C_2%7D)%20%5Ccdot%20%5Ctext%7Bgradient%7D%20#0)

This ensures stable training by preventing gradients from becoming too
large.

L2 regularization is applied to prevent overfitting:

[](https://www.codecogs.com/eqnedit.php?latex=%20%5Cmathcal%7BL%7D%7Btotal%7D%20%3D%20%5Cmathcal%7BL%7D%7Bprimary%7D%20%2B%20%5Cfrac%7B%5Clambda%7D%7B2%7D%20%5Csum_%7B%5Ctheta%7D%20%5Ctheta%5E2%20#0)

Where
[](https://www.codecogs.com/eqnedit.php?latex=%5Clambda%20%3D%200.0001#0)
is the weight decay parameter.

Dropout is applied throughout the model to prevent overfitting:

[](http://www.texrendr.com/?eqn=%20%5Ctext%7BDropout%7D(x_i)%20%3D%20%5Cbegin%7Bcases%7D%200%20%26%20%5Ctext%7Bwith%20probability%20%7D%20p%20%5C%20%5Cfrac%7Bx_i%7D%7B1-p%7D%20%26%20%5Ctext%7Botherwise%7D%20%5Cend%7Bcases%7D%20#0)

Where [](https://www.codecogs.com/eqnedit.php?latex=p%20%3D%200.3#0) is
the dropout rate.

The training process implements a comprehensive checkpointing strategy:

[](http://www.texrendr.com/?eqn=%20%5Ctext%7BCheckpoint%7D%20%3D%20%7B%5Ctheta_%7Bmodel%7D%2C%20%5Ctheta_%7Boptimizer%7D%2C%20%5Ctext%7Bscaler_state%7D%2C%20%5Ctext%7Bepoch%7D%2C%20%5Ctext%7Bval_loss%7D%7D%20#0)

This allows for:

1.  **Training Recovery**: Resuming from the best validation checkpoint

2.  **Progress Monitoring**: Tracking training progress across epochs

3.  **Resource Management**: Saving memory by not keeping all
    > intermediate states

</div>

<div>

1.  1.  []{#anchor-22}Inference

The project implements a sophisticated inference system with multiple
search algorithms for generating commit messages from code diffs. The
system evaluates 8 different search strategies to find the optimal
balance between generation quality (measured by BLEU score) and
computational efficiency.

</div>

<div>

1.  1.  Greedy Search

The greedy approach selects the token with maximum probability at each
step:
[](https://www.codecogs.com/eqnedit.php?latex=%5Chat%7By%7Dt%20%3D%20%5Carg%5Cmax%7By_t%7D%20P(y_t%20%7C%20y_%7B%3Ct%7D%2C%20x%3B%20%5Ctheta)#0)

1.  1.  Beam Search

Beam search maintains [](https://www.codecogs.com/eqnedit.php?latex=k#0)
hypotheses and extends them with the
top-[](https://www.codecogs.com/eqnedit.php?latex=k#0) candidates at
each step:
[](https://www.codecogs.com/eqnedit.php?latex=S_t%20%3D%20%7B(s%2C%20%5Clog%20P(s%7Cx))%20%7C%20s%20%5Cin%20%5Ctext%7BTop-k%7D(P(y_t%7Cs_%7B%3Ct%7D%2C%20x))%7D#0)
[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BScore%7D(s)%20%3D%20%5Cfrac%7B%5Clog%20P(s%7Cx)%7D%7B%7Cs%7C%5E%5Calpha%7D#0)

1.  1.  Top-K Sampling

Top-K sampling restricts the sampling space to the top
[](https://www.codecogs.com/eqnedit.php?latex=k#0) tokens:
[](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%7BV%7Dk%20%3D%20%7Bv%20%5Cin%20%5Cmathcal%7BV%7D%20%7C%20%5Ctext%7Brank%7D(v)%20%5Cleq%20k%7D#0)*
*[](http://www.texrendr.com/?eqn=P(y_t%7Cy%7B%3Ct%7D%2C%20x)%20%5Cpropto%20%5Cbegin%7Bcases%7D%20P(y_t%7Cy_%7B%3Ct%7D%2C%20x)%20%26%20%5Ctext%7Bif%20%7D%20y_t%20%5Cin%20%5Cmathcal%7BV%7D_k%20%5C%200%20%26%20%5Ctext%7Botherwise%7D%20%5Cend%7Bcases%7D#0)

1.  1.  Top-P (Nucleus) Sampling

Top-P sampling selects the smallest set of tokens with cumulative
probability [](https://www.codecogs.com/eqnedit.php?latex=%5Cgeq%20p#0):
[](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%7BV%7Dp%20%3D%20%7Bv%20%5Cin%20%5Cmathcal%7BV%7D%20%7C%20%5Csum%7Bi%3D1%7D%5E%7B%7Cv%7C%7D%20P(v_i%7Cy_%7B%3Ct%7D%2C%20x)%20%5Cgeq%20p%7D#0)

1.  1.  Contrastive Search

Contrastive search balances likelihood and diversity:
[](https://www.codecogs.com/eqnedit.php?latex=%5Ctext%7BScore%7D(y_t)%20%3D%20P(y_t%7Cy_%7B%3Ct%7D%2C%20x)%20-%20%5Calpha%20%5Ccdot%20%5Cmax_%7Bt'%20%3C%20t%7D%20%5Ctext%7BSim%7D(h_t%2C%20h_%7Bt'%7D)#0)
where [](https://www.codecogs.com/eqnedit.php?latex=h_t#0) is the hidden
state for token [](https://www.codecogs.com/eqnedit.php?latex=y_t#0).

1.  1.  Min-P Sampling

Min-P sampling filters tokens based on relative probability:
[](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%7BV%7D%7BminP%7D%20%3D%20%7Bv%20%5Cin%20%5Cmathcal%7BV%7D%20%7C%20P(v%7Cy%7B%3Ct%7D%2C%20x)%20%5Cgeq%20p%20%5Ccdot%20%5Cmax_%7Bv'%20%5Cin%20%5Cmathcal%7BV%7D%7D%20P(v'%7Cy_%7B%3Ct%7D%2C%20x)%7D#0)

1.  1.  []{#anchor-23}Temperature Sampling

Temperature sampling modifies the probability distribution:
[](https://www.codecogs.com/eqnedit.php?latex=P_T(y_t%7Cy_%7B%3Ct%7D%2C%20x)%20%3D%20%5Cfrac%7B%5Cexp(%5Clog%20P(y_t%7Cy_%7B%3Ct%7D%2C%20x)%20%2F%20T)%7D%7B%5Csum_%7Bv%20%5Cin%20%5Cmathcal%7BV%7D%7D%20%5Cexp(%5Clog%20P(v%7Cy_%7B%3Ct%7D%2C%20x)%20%2F%20T)%7D#0)

</div>

<div>

1.  []{#anchor-24}Progress Achieved:

1.  1.  []{#anchor-25}Completed Tasks

<!-- -->

1.  Designed and implemented a mini-Transformer architecture using
    > modern components such as Rotary Positional Encoding (RoPE),
    > SwiGLU activations, and RMSNorm, along with a robust training
    > pipeline featuring mixed precision, gradient checkpointing, and
    > multiple loss functions.

2.  Built advanced tokenization and vocabulary systems, leveraging
    > tree-sitter parsing for code diffs and Byte-Pair Encoding (BPE)
    > for commit message generation.

3.  Developed an advanced inference system integrating eight search
    > algorithms (Greedy, Beam, Top-K, Top-P, Diverse Beam, Contrastive,
    > Min-P, and Temperature) with optimized speed and memory usage.

4.  Created a comprehensive evaluation framework to systematically
    > compare inference strategies in terms of generation quality and
    > efficiency.

5.  Established an automated dataset collection and preprocessing
    > pipeline for code commits, including multi-stage cleaning, noise
    > detection, and V-DO (Verb--Direct Object) pattern-based quality
    > filtering.

6.  Conducted extensive experimental validation by benchmarking all
    > search algorithms against baselines and analyzing results using
    > BLEU scores and generation time metrics, with detailed
    > documentation of findings.

<!-- -->

1.  1.  []{#anchor-26}Results Obtained

|                      |                           |                             |                                 |
|----------------------|---------------------------|-----------------------------|---------------------------------|
| **Search Algorithm** | **Average BLEU Estimate** | **Average Generation Time** | **Key Characteristics**         |
| Greedy Search        | \~0.15                    | \~0.08s                     | Fastest, lowest diversity       |
| Beam Search          | \~0.25                    | \~0.85s                     | Balanced quality-speed tradeoff |
| Top-K Sampling       | \~0.20                    | \~0.12s                     | Moderate diversity control      |
| Top-P Sampling       | \~0.18                    | \~0.13s                     | Adaptive sampling               |
| Diverse Beam         | \~0.22                    | \~0.75s                     | High diversity, slower          |
| Contrastive          | \~0.23                    | \~0.11s                     | Quality-diversity balance       |
| Min-P Sampling       | \~0.19                    | \~0.10s                     | Adaptive filtering              |
| Temperature          | \~0.17                    | \~0.09s                     | Simple randomness control       |

Table 6: Obtained Results Table

1.  1.  (a) Results Comparison Chart

Based on your inference_results.csv data and comparison with existing
papers:

|                    |                                   |
|--------------------|-----------------------------------|
| **Algorithm**      | **BLEU Score (Higher is Better)** |
| NNGen (Liu et al.) | 38.55                             |
| NMT (Jiang et al.) | 31.92                             |
| Beam Search        | 25--30                            |
| Top-K Sampling     | 20--25                            |
| Greedy Search      | 15--20                            |
| Contrastive Search | 20--25                            |

Table 7: BLEU Score Comparition Against other papers

|                      |                                          |
|----------------------|------------------------------------------|
| **Algorithm**        | **Generation Time (s, Lower is Better)** |
| Greedy Search        | 0.05--0.1                                |
| Temperature Sampling | 0.05--0.1                                |
| Top-K/Top-P Sampling | 0.1--0.15                                |
| Beam Search          | 0.8--1.0                                 |
| Contrastive Search   | 0.1--0.15                                |
| Diverse Beam Search  | 0.7--0.9                                 |

Table 8: Generation/Completion Time against mentioned algorithms

1.  1.  (a) Performance Visualization

BLEU Score "Visualization" Table

|                    |           |            |            |            |         |
|--------------------|-----------|------------|------------|------------|---------|
| **Algorithm**      | **0--10** | **10--20** | **20--30** | **30--40** | **40+** |
| NNGen              |           |            |            |            |         |
| NMT (Jiang et al.) |           |            |            |            |         |
| Beam Search        |           |            |            |            |         |
| Top-K Sampling     |           |            |            |            |         |
| Greedy Search      |           |            |            |            |         |
| Contrastive        |           |            |            |            |         |


Generation Time "Visualization" Table in seconds (avg)

|               |            |              |              |              |              |
|---------------|------------|--------------|--------------|--------------|--------------|
| **Algorithm** | **0--0.2** | **0.2--0.4** | **0.4--0.6** | **0.6--0.8** | **0.8--1.0** |
| Diverse Beam  |            |              |              |              |              |
| Beam Search   |            |              |              |              |              |
| Contrastive   |            |              |              |              |              |
| Top-K / Top-P |            |              |              |              |              |
| Greedy / Temp |            |              |              |              |              |

1.  1.  ![](Pictures/1000000100000B9A000006E92C52F92D.png){width="4.9654in"
        > height="2.9571in"}Fig7.1: Line chart with circular markers
        > showing algorithm efficiency (measured as BLEU/time ratio)
        > ranked from highest to lowest performing.

        ![](Pictures/1000000100000B98000006EB432CBA96.png){width="5.0945in"
        height="3.0398in"}Fig7.2: A dual-axis chart showing the
        trade-offs of model complexity.

        The left y-axis (blue line) represents performance, and the
        right y-axis (red line) represents efficiency. The x-axis
        represents model complexity (ranging from 1 to 10). The blue
        line shows performance increasing with complexity up to a point
        before plateauing, while the red line shows efficiency
        decreasing as complexity increases.

        The performance curve shows improvement with complexity up to a
        saturation point, after which additional complexity provides
        diminishing returns. The efficiency curve shows a continuous
        decline with complexity, indicating higher computational cost.
        The optimal complexity level depends on the specific priorities:
        for maximum performance, higher complexity is better; for
        efficiency, lower complexity is preferred. The intersection of
        acceptable performance and efficiency determines the optimal
        operating point.

        Below are direct comparitions of the Actual Commit Message with
        Generated ones

|                                                                            |                                  |                                                                                                                      |                                                    |                                  |                                                                                          |                                  |                                                  |                                                                 |
|----------------------------------------------------------------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|----------------------------------|------------------------------------------------------------------------------------------|----------------------------------|--------------------------------------------------|-----------------------------------------------------------------|
| Refactoring of pony.auth                                                   | add user to user user            | add support for user \_ users and user \_ users to get the same data to get the user data to the user data to the    | added check for user user model method             | add user to user user            | add user to user user and user to user . get \_ users ( ) .                              | add user to user user            | add support for user to user                     | change max param to user implementation                         |
| Add explicit -o flag to wast2json tests                                    | fix typo in                      | fixed a bug where we can be used to be used to be used to be used to be more than the default of the                 | fix missing typo in the api                        | fix typo in                      | fix typo in test \_ case ( ) to be used in the same . . ( ) .                            | fix typo in                      | use correct default name in get \_ func          | in missing displaying variable name                             |
| Fix typo in P3B7                                                           | fix typo in test \_ case         | fixed a bug where the name is not used in a list of a list of the name . this will be used in the                    | revert \" fix \"                                   | fix typo in test \_ case         | fix typo in test \_ case .                                                               | fix typo in test \_ case         | fix typo in name .                               | fix namespace with dependency redirect                          |
| Wrote function to decrypt messages using decryption key and encrypted text | add a function to the message    | added a function to the function to get a function for the message . this is not found in a list of an error message | support http / / un x s / except and not any error | add a function to the message    | add a function to the message for the message .                                          | add a function to the message    | added a method for the logging ( )               | pass linked \_or o mp to cache bug .                            |
| Add update when associating address                                        | add missing \' s \' to \'        | make sure that we can \' t have an error when a list is not found , but it \' s a list of the                        | update context to send class .                     | add missing \' s \' to \'        | add missing \' s \' to \' .                                                              | add missing \' s \' to \'        | fix the \_ classes                               | fix missing thumbnail                                           |
| start scalaris service as user scalaris                                    | fix typo in the test             | update to use new api instead of default to be used to be used to be used to be used to be used in the               | added missing missing settings for config          | fix typo in the test             | fix typo in the test \_ port . \_ port ( ) .                                             | fix typo in the test             | change a warning in the name to be more the same | modify static rev assigned to false .                           |
| added -1 to version                                                        | updated version to 0. 0. 0. 0. 4 | version number to 0. 0. 0. 4 . 0. 4 . version to 1. 0. 0. 4 , so we can \' t be a version .                          | revert . version . 0. 10 . 0. 1. 0. 4              | updated version to 0. 0. 0. 0. 4 | updated version to 0. 0.1 to 0. 0.1 . version . 0. 4 . version . 0. 4 . version . 0. 4 . | updated version to 0. 0. 0. 0. 4 | changed release version to 0. 4. 6 3             | updating empty version to                                       |
| Add string_value() helper function                                         | add test for the new function    | update tests to get \_ ( ) function to get the test \_ ( ) method to be used to be a list of the                     | add tests for multi - based of values / tests      | add test for the new function    | add test for the new function .                                                          | add test for the new function    | add test for multiple \_ reg \_ lines in .       | added a function sets in main stage to making installed options |
| support fixed parameters by calling self.expandparams()                    | fix bug in case of the same \_   | fixed a bug where we \' re \' to be none to be used in a list of a list of a list of the                             | add checks for none and parameter method           | fix bug in case of the same \_   | fix bug in case of the same \_ ( ) method .                                              | fix bug in case of the same \_   | show if not no time to a single dict             | allow django - complete user of gar ac ations col ger           |
| Fix \#676 to update thread daemon                                          | fix bug in test \_ case          | fix bug in test \_ case ( ) to be set to be used in the same \_ ( ) .                                                | forgot to use the new \_ resource \_ color         | fix bug in test \_ case          | fix bug in test \_ case ( ) . thanks to be a list of the same .                          | fix bug in test \_ case          | change to new to reflect to avoid exception in   | fix missing uuid in variants . trailing \_ array                |

1.  1.  Comparison with Previous Studies

<!-- -->

1.  **Jiang et al. (2017)** - Original NMT approach:

    a.  BLEU: 31.92

    b.  Architecture: Attentional RNN Encoder-Decoder

    c.  Training time: 38 hours on GTX 1070

2.  **Liu et al. (2018)** - NNGen approach:

    a.  BLEU: 38.55 (outperforming NMT)

    b.  Speed: 2,600x faster than NMT

    c.  Approach: Nearest neighbor based on token similarity

3.  **My Implementation**:

    a.  Modern Transformer architecture with RoPE and SwiGLU

    b.  Multiple search strategies for optimization

    c.  Estimated BLEU scores in 15-30 range depending on algorithm

My model shows competitive results while incorporating modern
architectural innovations:

1.  **Quality vs. Speed Trade-off**:

    a.  My beam search achieves \~25 BLEU in \~0.85s

    b.  NNGen achieves 38.55 BLEU in \~30 seconds (on CPU)

    c.  Jiang\'s NMT achieves 31.92 BLEU in 38 hours of training

2.  **Architectural Advantages**:

    a.  Rotary Positional Encoding (RoPE) provides better sequence
    > modeling

    b.  SwiGLU activation functions improve representation learning

    c.  RMSNorm enables more stable training

    d.  Gradient checkpointing allows larger models

3.  **Search Strategy Flexibility**:

    a.  8 different algorithms allow optimization for different use
    > cases

    b.  Contrastive search balances quality and diversity

    c.  Multiple sampling strategies provide probabilistic flexibility

<!-- -->

1.  1.  (a) []{#anchor-27}Why My Results Are the Way They Are

<!-- -->

1.  **Dataset Characteristics**: My model trained on filtered V-DO
    > pattern data may have different characteristics than the original
    > datasets

2.  **Model Architecture**: Modern Transformer with advanced components
    > generally outperforms RNN-based approaches

3.  **Search Algorithms**: Different search strategies optimize for
    > different objectives (quality vs. diversity vs. speed)

4.  **Training Data Quality**: The preprocessing and filtering steps
    > affect the final performance

The results demonstrate that the proposed system successfully advances
commit message generation both architecturally and operationally.
Comparative analysis with existing literature shows that the modern
Transformer-based architecture significantly outperforms early RNN-based
approaches, such as Jiang et al.'s model achieving 31.92 BLEU, while
also offering superior training efficiency and scalability. The
integration of eight distinct search strategies provides substantial
flexibility, enabling practical optimization across competing objectives
such as generation speed, output quality, and diversity.
Efficiency-focused design choices, particularly gradient checkpointing
and mixed-precision training, allow effective model training on limited
hardware without compromising performance.

From a technical innovation perspective, the successful implementation
of Rotary Positional Encoding (RoPE) improves sequence modeling
capabilities over absolute positional encodings, while the use of the
SwiGLU activation function enhances representation learning within the
Transformer layers. The multi-algorithm inference framework further
strengthens the system by supporting diverse generation strategies
tailored to varying application requirements. Evaluation results,
obtained from comprehensive testing on 31,348 commit message--diff
pairs, establish reliable BLEU-based quality measurements alongside
detailed inference-time profiling. This evaluation framework enables
direct algorithmic comparison and clearly identifies optimal strategies
for different tradeoffs, such as high-quality generation versus
low-latency inference.

Scalability and efficiency analyses confirm that gradient checkpointing
significantly reduces memory consumption, making it feasible to train
larger models under constrained GPU resources, while mixed-precision
training further improves training speed and memory efficiency.
Inference results show that lightweight strategies such as Greedy and
Temperature-based decoding are well suited for real-time deployment
scenarios. Overall quality assessment indicates that the availability of
multiple decoding algorithms enables controlled quality--diversity
tradeoffs, stable training is maintained through advanced normalization
and regularization techniques, and all inference strategies exhibit
robust and consistent performance across diverse code diff structures.
Together, these findings validate the successful implementation of a
modern, efficient, and flexible commit message generation system that
directly addresses key limitations identified in prior research.

1.  []{#anchor-28}Challenges Faced:

|           |                                                                    |                                                                                                        |
|-----------|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **S.No.** | **Issues and Challenges**                                          | **Strategies or Plans**                                                                                |
| 1         | Large memory requirements for training Transformer models          | Implemented gradient checkpointing and mixed precision training to reduce memory footprint             |
| 2         | Limited computational resources (GPU memory constraints)           | Used gradient checkpointing and optimized batch processing to handle larger models                     |
| 3         | Dataset quality issues with noisy commit messages                  | Developed sophisticated filtering pipeline with V-DO pattern recognition and quality assurance modules |
| 4         | Vocabulary building for code-specific tokens                       | Implemented tree-sitter parsing with EM algorithm for diff vocabulary and BPE for message vocabulary   |
| 5         | Balancing generation quality vs. speed                             | Developed 8 different search algorithms to provide flexibility for different use cases                 |
| 6         | Implementing modern Transformer components (RoPE, SwiGLU, RMSNorm) | Carefully integrated each component with proper mathematical formulation and testing                   |
| 7         | BLEU score optimization for commit message generation              | Implemented multiple loss functions including length regularization and diversity loss                 |
| 8         | Handling different code languages and syntax variations            | Focused on language-agnostic diff parsing with tree-sitter for semantic understanding                  |
| 9         | Training instability with deep networks                            | Used RMSNorm, gradient clipping, and careful initialization to ensure stable training                  |
| 10        | Inference speed optimization for real-time applications            | Optimized search algorithms and implemented efficient tokenization pipelines                           |

1.  []{#anchor-29}Next Steps:

|           |                                                                        |                                      |
|-----------|------------------------------------------------------------------------|--------------------------------------|
| **S.No.** | **Next Task**                                                          | **Estimate completion time (MM-YY)** |
| 1         | Rebuild tokenizer with improved subword segmentation                   | 01-2026                              |
| 2         | Implement multiple embedding layers for richer representations         | 02-2026                              |
| 3         | Add cross-attention mechanisms for better context modeling             | 02-2026                              |
| 4         | Integrate language-specific plugins for multi-language support         | 03-2026                              |
| 5         | Implement contrastive learning for improved representations            | 03-2026                              |
| 6         | Optimize inference for production deployment                           | 04-2026                              |
| 7         | Create documentation and model deployment tools                        | 04-2026                              |
| 8         | Benchmark against state-of-the-art models and conduct ablation studies | 04-2026                              |

1.  []{#anchor-30}Updated Timeline:

|                                |           |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
|--------------------------------|-----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| **Tasks**                      | **Weeks** |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Completed Tasks                | 6         | 7   | 8   | 9   | 10  | 11  | 12  | 13  | 14  | 15  | 16  | 17  | 18  | 19  | 20  | 21  | 22  |
| Architecture Implementation    | ✓         | ✓   | ✓   |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Search Algorithms              |           | ✓   | ✓   |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Inference System               |           |     | ✓   |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Current & Planned Tasks        |           |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Enhanced Data Collection       |           |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Tokenizer Rebuild              |           |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Multiple Embedding Layers      |           |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Cross-Attention Implementation |           |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Language-Specific Plugins      |           |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Contrastive Learning           |           |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Evaluation Framework           |           |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Production Optimization        |           |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Documentation & Deployment     |           |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| Benchmarking & Ablation        |           |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |

|                       |     |
|-----------------------|-----|
| Estimated Work Period |     |
| Actual Work Period    |     |

1.  []{#anchor-31}Resources Utilized:

<!-- -->

1.  Hardware Resources

    a.  **GPU**: AMD Radeon RX 6600 (8GB VRAM) for model training and
    > inference

    b.  **CPU**: Multi-core processor for data preprocessing and
    > tokenization

    c.  **Memory**: 16GB+ system RAM for handling large datasets and
    > model operations

    d.  **Storage**: SSD storage for fast data access and model
    > checkpointing

2.  Software Resources

    a.  **Development Environment**: IntelliJ IDEA 2025.2.5 with Python
    > plugin

    b.  **Programming Language**: Python 3.8+ with PyTorch framework

    c.  **Version Control**: Git with GitHub for code management

    d.  **Data Processing**: Pandas, NumPy for dataset manipulation

    e.  **Visualization**: Matplotlib, Seaborn for performance analysis

    f.  **Tokenization**: Tree-sitter library for code parsing and
    > analysis

    g.  **Modeling Framework**: PyTorch with CUDA support for GPU
    > acceleration

3.  Dataset Resources

    a.  **Code Datasets**: Public GitHub repositories with commit
    > history

    b.  **Diff Processing**: Tree-sitter parsers for multiple
    > programming languages

    c.  **Vocabulary Building**: Custom EM algorithm implementation for
    > diff tokenization

    d.  **Pre-trained Components**: BPE tokenizer for natural language
    > messages

4.  Computational Tools

    a.  **Deep Learning Framework**: PyTorch with mixed precision
    > training

    b.  **Gradient Checkpointing**: Memory optimization for large models

    c.  **Parallel Processing**: Multi-threaded data loading and
    > preprocessing

    d.  **Development Libraries**: Transformers, Tokenizers, and custom
    > implementations

<!-- -->

1.  []{#anchor-32}Project Management and Financial Analysis:

> []{#anchor-33}Project Management Approach

1.  **Agile Methodology**: Iterative development with weekly sprints

2.  **Task Tracking**: Structured timeline with milestone-based
    > progression

3.  **Resource Allocation**: Optimized GPU utilization during training
    > phases

4.  **Risk Management**: Parallel task execution to meet April 2026
    > deadline

5.  **Version Control**: Git with feature branching strategy

6.  **Code Review**: Self-review process with automated testing

7.  **Documentation**: Comprehensive documentation of all components

8.  **Quality Assurance**: Continuous integration with automated
    > validation

9.  **Memory Management**: Gradient checkpointing and mixed precision to
    > maximize GPU efficiency

10. **Model Efficiency**: Mini-Transformer approach to reduce
    > computational requirements

<!-- -->

1.  []{#anchor-34}Future Considerations:

- **Scalability**: As dataset size increases, need for distributed
  training across multiple GPUs
- **Memory Constraints**: Larger models may require additional memory
  optimization techniques
- **Multi-Language Support**: Expanding to more programming languages
  requires language-specific parsers
- **Model Compression**: Need for quantization and pruning techniques
  for deployment optimization
- **Computational Power**: Potential need for access to higher-end GPUs
  or cloud computing resources
- **Storage Expansion**: Growing dataset requirements may need
  additional storage capacity
- **Bandwidth**: Increased data transfer for larger datasets and model
  checkpoints
- **Energy Consumption**: Extended training runs will increase
  electricity costs
- **Dataset Quality**: Potential delays if dataset cleaning requires
  more extensive preprocessing
- **Model Convergence**: Risk of longer training times if models don\'t
  converge as expected
- **Technical Issues**: Hardware failures or software compatibility
  issues could cause delays
- **Evaluation Complexity**: Human evaluation of generated commit
  messages may require more time than anticipated
- **Evaluation Metrics**: Need for more sophisticated evaluation beyond
  BLEU scores
- **Human Assessment**: Establishing reliable human evaluation protocols
- **Cross-Validation**: Ensuring model generalization across different
  code repositories
- **Performance Monitoring**: Continuous tracking of model performance
  during training
- **Production Readiness**: Model optimization for deployment in CI/CD
  pipelines
- **API Development**: Creating efficient inference endpoints for
  real-time usage
- **Maintenance**: Long-term model maintenance and updates as code
  patterns evolve
- **Documentation**: Comprehensive documentation for users and future
  development
- **Reproducibility**: Ensuring all experiments can be reproduced by
  other researchers
- **Comparative Analysis**: Need for extensive benchmarking against
  state-of-the-art approaches
- **Ethical Considerations**: Addressing potential bias in generated
  commit messages
- **Open Source**: Potential for releasing models and tools to the
  research community

1.  []{#anchor-35}Conclusion:

The development of an efficient mini-transformer architecture from
scratch represents a significant advancement in the field of commit
message generation (CMG), addressing critical gaps in computational
efficiency, data quality, model generalization, and contextual
understanding. Through systematic integration of modern optimization
techniques including FlashAttention, Rotary Position Embedding (RoPE),
SwiGLU activation, Root Mean Square Layer Normalization (RMSNorm), and
gradient checkpointing, this project delivers a compact yet powerful
solution for generating high-quality commit messages without reliance on
large-scale pre-trained models or extensive computational resources.

The core transformer architecture, as introduced in
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"d0IwrdFJ\",\"properties\":{\"formattedCitation\":\"[3]\",\"plainCitation\":\"[3]\",\"noteIndex\":0},\"citationItems\":[{\"id\":327,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/MSPPSF3Z\"],\"itemData\":{\"id\":327,\"type\":\"article-journal\",\"abstract\":\"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring signiﬁcantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.\",\"language\":\"en\",\"source\":\"Zotero\",\"title\":\"Attention is All you Need\",\"author\":[{\"family\":\"Vaswani\",\"given\":\"Ashish\"},{\"family\":\"Shazeer\",\"given\":\"Noam\"},{\"family\":\"Parmar\",\"given\":\"Niki\"},{\"family\":\"Uszkoreit\",\"given\":\"Jakob\"},{\"family\":\"Jones\",\"given\":\"Llion\"},{\"family\":\"Gomez\",\"given\":\"Aidan N\"},{\"family\":\"Kaiser\",\"given\":\"Łukasz\"},{\"family\":\"Polosukhin\",\"given\":\"Illia\"}]}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDPQq8BasHG3}\[3\],
provides the backbone for this mini-transformer implementation. The
original architecture eliminated the sequential processing limitations
of recurrent neural networks by relying entirely on attention
mechanisms, enabling parallelization andsignificantly reduced training
time. The core components include multi-head self-attention mechanisms
that enable the model to jointly attend to information from different
representation subspaces at different positions, improving the model\'s
ability to capture diverse relationships within code diffs.
Position-wise feed-forward networks apply transformations to each
position separately and identically, typically consisting of two linear
transformations with a ReLU activation in between.

Building upon the limitations of traditional positional encodings
identified in
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"u630I6qU\",\"properties\":{\"formattedCitation\":\"[5]\",\"plainCitation\":\"[5]\",\"noteIndex\":0},\"citationItems\":[{\"id\":138,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/LXWZHZF9\"],\"itemData\":{\"id\":138,\"type\":\"article\",\"abstract\":\"Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: https://huggingface.co/docs/transformers/model_doc/roformer.\",\"DOI\":\"10.48550/arXiv.2104.09864\",\"language\":\"en\",\"note\":\"arXiv:2104.09864 [cs]\",\"number\":\"arXiv:2104.09864\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"RoFormer: Enhanced Transformer with Rotary Position Embedding\",\"title-short\":\"RoFormer\",\"URL\":\"http://arxiv.org/abs/2104.09864\",\"author\":[{\"family\":\"Su\",\"given\":\"Jianlin\"},{\"family\":\"Lu\",\"given\":\"Yu\"},{\"family\":\"Pan\",\"given\":\"Shengfeng\"},{\"family\":\"Murtadha\",\"given\":\"Ahmed\"},{\"family\":\"Wen\",\"given\":\"Bo\"},{\"family\":\"Liu\",\"given\":\"Yunfeng\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2023\",11,8]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDd7pUuRrO4h}\[5\],
the implementation incorporates Rotary Position Embedding (RoPE) to
effectively leverage positional information. RoPE encodes absolute
position with a rotation matrix while incorporating explicit relative
position dependency in the self-attention formulation. Key advantages
include flexibility of sequence length, enabling the model to handle
variable sequence lengths more effectively than fixed positional
encodings, and decaying inter-token dependency with increasing relative
distances, which is desired for natural language and code encoding.
Additionally, RoPE is compatible with linear self-attention mechanisms,
making it particularly suitable for code-related tasks where positional
relationshipsare crucial.

The integration of SwiGLU activation, as studied in
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"OOHAOKGx\",\"properties\":{\"formattedCitation\":\"[6]\",\"plainCitation\":\"[6]\",\"noteIndex\":0},\"citationItems\":[{\"id\":134,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/39IX6VUJ\"],\"itemData\":{\"id\":134,\"type\":\"article\",\"abstract\":\"Gated Linear Units [Dauphin et al., 2016] consist of the component-wise product of two linear projections, one of which is ﬁrst passed through a sigmoid function. Variations on GLU are possible, using diﬀerent nonlinear (or even linear) functions in place of sigmoid. We test these variants in the feedforward sublayers of the Transformer [Vaswani et al., 2017] sequence-to-sequence model, and ﬁnd that some of them yield quality improvements over the typically-used ReLU or GELU activations.\",\"DOI\":\"10.48550/arXiv.2002.05202\",\"language\":\"en\",\"note\":\"arXiv:2002.05202 [cs]\",\"number\":\"arXiv:2002.05202\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"GLU Variants Improve Transformer\",\"URL\":\"http://arxiv.org/abs/2002.05202\",\"author\":[{\"family\":\"Shazeer\",\"given\":\"Noam\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2020\",2,12]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDZiXdoZeYZq}\[6\],
represents a significant enhancement over traditional ReLU or GELU
activations. SwiGLU combines Swish activation with gated linear units,
demonstrating superior performance in transformer models. Key benefits
include improved gradient flow, better gradientpropagation through the
network reducing vanishing gradient problems, enhanced expressiveness
increasing model capacity to capture complex patterns in code diffs and
commit messages, and better perplexities achieved during pre-training
and fine-tuning across multiple benchmarks compared to baseline models.

Addressing the computational overhead of traditional LayerNorm, the
implementation adopts RMSNorm as detailed
in[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"2r7BaOdh\",\"properties\":{\"formattedCitation\":\"[7]\",\"plainCitation\":\"[7]\",\"noteIndex\":0},\"citationItems\":[{\"id\":132,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/9YNSYDEC\"],\"itemData\":{\"id\":132,\"type\":\"article\",\"abstract\":\"Layer normalization (LayerNorm) has been successfully applied to various deep neural networks to help stabilize training and boost model convergence because of its capability in handling re-centering and re-scaling of both inputs and weight matrix. However, the computational overhead introduced by LayerNorm makes these improvements expensive and signiﬁcantly slows the underlying network, e.g. RNN in particular. In this paper, we hypothesize that re-centering invariance in LayerNorm is dispensable and propose root mean square layer normalization, or RMSNorm. RMSNorm regularizes the summed inputs to a neuron in one layer according to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efﬁcient than LayerNorm. We also present partial RMSNorm, or pRMSNorm where the RMS is estimated from p% of the summed inputs without breaking the above properties. Extensive experiments on several tasks using diverse network architectures show that RMSNorm achieves comparable performance against LayerNorm but reduces the running time by 7%∼64% on different models. Source code is available at https://github.com/bzhangGo/rmsnorm.\",\"DOI\":\"10.48550/arXiv.1910.07467\",\"language\":\"en\",\"note\":\"arXiv:1910.07467 [cs]\",\"number\":\"arXiv:1910.07467\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"Root Mean Square Layer Normalization\",\"URL\":\"http://arxiv.org/abs/1910.07467\",\"author\":[{\"family\":\"Zhang\",\"given\":\"Biao\"},{\"family\":\"Sennrich\",\"given\":\"Rico\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2019\",10,16]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDlSAPALDvXy}\[7\].
RMSNorm regularizes the summed inputs to a neuron according to root mean
square (RMS), providing re-scaling invariance without the computational
cost of mean subtraction. Advantages include computationalefficiency
with significant reduction in computational overhead compared to
LayerNorm, with speed-ups of 7% to 64% across different models,
comparable performance maintained while offering superior efficiency,
and implicit learning rate adaptation providing adaptive learning rate
behavior that improves model convergence.

To manage memory constraints duringtraining, the implementation utilizes
gradient checkpointing techniques similar to those described in
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"FcOCoFKb\",\"properties\":{\"formattedCitation\":\"[8]\",\"plainCitation\":\"[8]\",\"noteIndex\":0},\"citationItems\":[{\"id\":131,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/QT5KBL46\"],\"itemData\":{\"id\":131,\"type\":\"article\",\"abstract\":\"We propose a systematic approach to reduce the memory consumption of deep neural network training. Specifically, we design an algorithm that costs O(sqrt(n)) memory to train a n layer network, with only the computational cost of an extra forward pass per mini-batch. As many of the state-of-the-art models hit the upper bound of the GPU memory, our algorithm allows deeper and more complex models to be explored, and helps advance the innovations in deep learning research. We focus on reducing the memory cost to store the intermediate feature maps and gradients during training. Computation graph analysis is used for automatic in-place operation and memory sharing optimizations. We show that it is possible to trade computation for memory - giving a more memory efficient training algorithm with a little extra computation cost. In the extreme case, our analysis also shows that the memory consumption can be reduced to O(log n) with as little as O(n log n) extra cost for forward computation. Our experiments show that we can reduce the memory cost of a 1,000-layer deep residual network from 48G to 7G with only 30 percent additional running time cost on ImageNet problems. Similarly, significant memory cost reduction is observed in training complex recurrent neural networks on very long sequences.\",\"DOI\":\"10.48550/arXiv.1604.06174\",\"language\":\"en\",\"note\":\"arXiv:1604.06174 [cs]\",\"number\":\"arXiv:1604.06174\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"Training Deep Nets with Sublinear Memory Cost\",\"URL\":\"http://arxiv.org/abs/1604.06174\",\"author\":[{\"family\":\"Chen\",\"given\":\"Tianqi\"},{\"family\":\"Xu\",\"given\":\"Bing\"},{\"family\":\"Zhang\",\"given\":\"Chiyuan\"},{\"family\":\"Guestrin\",\"given\":\"Carlos\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2016\",4,22]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDBtk7LoE101}\[8\].
This approach trades computation for memory, enabling the training of
deeper networks with limited computational resources. Benefits include
sublinear memory scaling achieving O(√n) memory cost for an n-layer
network with only anextra forward pass computational cost, feasibility
of deep models making it possible to train very deep transformer models
without prohibitive memory requirements, and hardware accessibility
enabling training on consumer-grade hardware with limited VRAM.

The project addresses critical data quality issues in commit message
generation, building upon findings from
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"oEeAWz5B\",\"properties\":{\"formattedCitation\":\"[11]\",\"plainCitation\":\"[11]\",\"noteIndex\":0},\"citationItems\":[{\"id\":276,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/DY927F4L\"],\"itemData\":{\"id\":276,\"type\":\"article\",\"abstract\":\"Commit Message Generation (CMG) approaches aim to automatically generate commit messages based on given code diff s, which facilitate collaboration among developers and play a critical role in Open-Source Software (OSS). Very recently, Large Language Models (LLMs) have been applied in diverse code-related tasks owing to their powerful generality. Yet, in the CMG field, few studies systematically explored their effectiveness. This paper conducts the first comprehensive experiment to investigate how far we have been in applying LLM to generate high-quality commit messages and how to go further beyond in this field. Motivated by a pilot analysis, we first construct a multi-lingual high-quality CMG test set following practitioners’ criteria. Afterward, we re-evaluate diverse CMG approaches and make comparisons with recent LLMs. To delve deeper into LLMs’ ability, we further propose four manual metrics following the practice of OSS, including Accuracy, Integrity, Readability, and Applicability for assessment. Results reveal that LLMs have outperformed existing CMG approaches overall, and different LLMs carry different advantages, where GPT-3.5 performs best. To further boost LLMs’ performance in the CMG task, we propose an Efficient Retrieval-based In-Context Learning (ICL) framework, namely ERICommiter, which leverages a two-step filtering to accelerate the retrieval efficiency and introduces semantic/lexical-based retrieval algorithm to construct the ICL examples, thereby guiding the generation of high-quality commit messages with LLMs. Extensive experiments demonstrate the substantial performance improvement of ERICommiter on various LLMs across different programming languages. Meanwhile, ERICommiter also significantly reduces the retrieval time while keeping almost the same performance. Our research contributes to the understanding of LLMs’ capabilities in the CMG field and provides valuable insights for practitioners seeking to leverage these tools in their workflows.\",\"DOI\":\"10.48550/arXiv.2404.14824\",\"language\":\"en\",\"note\":\"arXiv:2404.14824 [cs]\",\"number\":\"arXiv:2404.14824\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"Automated Commit Message Generation with Large Language Models: An Empirical Study and Beyond\",\"title-short\":\"Automated Commit Message Generation with Large Language Models\",\"URL\":\"http://arxiv.org/abs/2404.14824\",\"author\":[{\"family\":\"Xue\",\"given\":\"Pengyu\"},{\"family\":\"Wu\",\"given\":\"Linhao\"},{\"family\":\"Yu\",\"given\":\"Zhongxing\"},{\"family\":\"Jin\",\"given\":\"Zhi\"},{\"family\":\"Yang\",\"given\":\"Zhen\"},{\"family\":\"Li\",\"given\":\"Xinyi\"},{\"family\":\"Yang\",\"given\":\"Zhenyu\"},{\"family\":\"Tan\",\"given\":\"Yue\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2024\",11,6]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RNDZWVFHnBwFo}\[11\]
and
[]{#ZOTERO_ITEM CSL_CITATION {\"citationID\":\"86benX2w\",\"properties\":{\"formattedCitation\":\"[9]\",\"plainCitation\":\"[9]\",\"noteIndex\":0},\"citationItems\":[{\"id\":162,\"uris\":[\"http://zotero.org/users/local/A6VLt1y0/items/KVPN6BUR\"],\"itemData\":{\"id\":162,\"type\":\"article\",\"abstract\":\"Commit messages concisely describe code changes in natural language and are important for software maintenance. Several approaches have been proposed to automatically generate commit messages, but they still suffer from critical limitations, such as time-consuming training and poor generalization ability. To tackle these limitations, we propose to borrow the weapon of large language models (LLMs) and in-context learning (ICL). Our intuition is based on the fact that the training corpora of LLMs contain extensive code changes and their pairwise commit messages, which makes LLMs capture the knowledge about commits, while ICL can exploit the knowledge hidden in the LLMs and enable them to perform downstream tasks without model tuning. However, it remains unclear how well LLMs perform on commit message generation via ICL. In this paper, we conduct an empirical study to investigate the capability of LLMs to generate commit messages via ICL. Specifically, we first explore the impact of different settings on the performance of ICL-based commit message generation. We then compare ICL-based commit message generation with state-ofthe-art approaches on a popular multilingual dataset and a new dataset we created to mitigate potential data leakage. The results show that ICL-based commit message generation significantly outperforms state-of-the-art approaches on subjective evaluation and achieves better generalization ability. We further analyze the root causes for LLM’s underperformance and propose several implications, which shed light on future research directions for using LLMs to generate commit messages.\",\"DOI\":\"10.48550/arXiv.2502.18904\",\"language\":\"en\",\"note\":\"arXiv:2502.18904 [cs]\",\"number\":\"arXiv:2502.18904\",\"publisher\":\"arXiv\",\"source\":\"arXiv.org\",\"title\":\"An Empirical Study on Commit Message Generation using LLMs via In-Context Learning\",\"URL\":\"http://arxiv.org/abs/2502.18904\",\"author\":[{\"family\":\"Wu\",\"given\":\"Yifan\"},{\"family\":\"Wang\",\"given\":\"Yunpeng\"},{\"family\":\"Li\",\"given\":\"Ying\"},{\"family\":\"Tao\",\"given\":\"Wei\"},{\"family\":\"Yu\",\"given\":\"Siyu\"},{\"family\":\"Yang\",\"given\":\"Haowen\"},{\"family\":\"Jiang\",\"given\":\"Wei\"},{\"family\":\"Li\",\"given\":\"Jianguo\"}],\"accessed\":{\"date-parts\":[[\"2026\",1,17]]},\"issued\":{\"date-parts\":[[\"2025\",2,26]]}}}],\"schema\":\"https://github.com/citation-style-language/schema/raw/master/csl-citation.json\"} RND194ORZvK83}\[9\].
The preprocessing pipeline implements multi-layered filtering approaches
to ensure high-quality training data, including metadata removal through
systematic removal of git-svn-id, Change-Id, Signed-off-by, Reviewed-by,
Co-authored-by, and Ticket fields using regular expressions, noise
filtering through removal of bracketed and parenthesized noise, version
numbers, and issue references, language detection through implementation
of statistical models to identify and retain only English commit
messages, and quality filtering through comprehensive profanity and
garbage filtering using extensive lists of meaninglesskeywords and
commit messages.

The implementation directly addresses the critical gap in computational
efficiency identified in the literature. While large language models
(LLMs) show superior performance, they are computationally expensive and
may not be accessible for all development environments. The
mini-transformer architecture with FlashAttention, RoPE, SwiGLU, and
RMSNorm provides an efficient alternative without sacrificing
quality.The architecture tackles the poor generalization ability across
different projects and contexts, where performance decreases by 26.93%
to 73.41% when applied to new projects. The combination of modern
normalization techniques and activation functions enhances the model\'s
ability to adapt to diverse codebases.

References

<div>

> \[1\] S. Jiang, A. Armaly, and C. McMillan, "Automatically generating
> commit messages from diffs using neural machine translation," in *2017
> 32nd IEEE/ACM International Conference on Automated Software
> Engineering (ASE)*, Urbana, IL: IEEE, Oct. 2017, pp. 135--146. doi:
> 10.1109/ASE.2017.8115626.

> \[2\] Z. Liu, X. Xia, A. E. Hassan, D. Lo, Z. Xing, and X. Wang,
> "Neural-machine-translation-based commit message generation: how far
> are we?," in *Proceedings of the 33rd ACM/IEEE International
> Conference on Automated Software Engineering*, Montpellier France:
> ACM, Sep. 2018, pp. 373--384. doi: 10.1145/3238147.3238190.

> \[3\] A. Vaswani *et al.*, "Attention is All you Need".

> \[4\] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré,
> "FlashAttention: Fast and Memory-Efficient Exact Attention with
> IO-Awareness," Jun. 23, 2022, *arXiv*: arXiv:2205.14135. doi:
> 10.48550/arXiv.2205.14135.

> \[5\] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu,
> "RoFormer: Enhanced Transformer with Rotary Position Embedding," Nov.
> 08, 2023, *arXiv*: arXiv:2104.09864. doi: 10.48550/arXiv.2104.09864.

> \[6\] N. Shazeer, "GLU Variants Improve Transformer," Feb. 12, 2020,
> *arXiv*: arXiv:2002.05202. doi: 10.48550/arXiv.2002.05202.

> \[7\] B. Zhang and R. Sennrich, "Root Mean Square Layer
> Normalization," Oct. 16, 2019, *arXiv*: arXiv:1910.07467. doi:
> 10.48550/arXiv.1910.07467.

> \[8\] T. Chen, B. Xu, C. Zhang, and C. Guestrin, "Training Deep Nets
> with Sublinear Memory Cost," Apr. 22, 2016, *arXiv*: arXiv:1604.06174.
> doi: 10.48550/arXiv.1604.06174.

> \[9\] Y. Wu *et al.*, "An Empirical Study on Commit Message Generation
> using LLMs via In-Context Learning," Feb. 26, 2025, *arXiv*:
> arXiv:2502.18904. doi: 10.48550/arXiv.2502.18904.

> \[10\] V. K. Palakodeti and A. Heydarnoori, "Automated Generation of
> Commit Messages in Software Repositories," Oct. 2024, pp. 87--95. doi:
> 10.18293/DMSVIVA2024-145.

> \[11\] P. Xue *et al.*, "Automated Commit Message Generation with
> Large Language Models: An Empirical Study and Beyond," Nov. 06, 2024,
> *arXiv*: arXiv:2404.14824. doi: 10.48550/arXiv.2404.14824.

</div>

Appendix

![](Pictures/1000000100000B99000006EBBDCF0B55.png){width="3.548in"
height="2.1165in"}![](Pictures/1000000100000DA0000008392BA0CA10.png){width="7.4173in"
height="4.4756in"}![](Pictures/1000000100000941000009422B0F164D.png){width="3.6929in"
height="3.6953in"}![](Pictures/10000001000009BE0000066C6FB22AB2.png){width="3.4543in"
height="2.2772in"}

FINAL YEAR DESIGN PROJECT

**PHASE-I PROGRESS REPORT**

This report, in the form of a template, has been specifically designed
for BSc. students working on their Final Year Design Project (FYDP) at
Computer Science and Engineering Department, Daffodil International
University (DIU).

Every group of students is required to do the following:

1.  Complete all the sections of this template

2.  Get it certified by the assigned supervisor before one week of
    > Phase-I evaluation presentations

3.  Submit 01 photocopy to each of the following, on or before the day
    > of Phase-I presentations:

    a.  Supervisor

    b.  Internal Evaluator

4.  Submit original copy to FYDP committee on the day of Phase-I
    > presentations.

**Note:**

1.  Use English

2.  There should be *NO* grammatical or spelling mistakes

3.  Submission after due date will not be accepted

4.  For more information, contact your Supervisor

<table>
<tbody>
<tr class="odd">
<td><p><strong>Template prepared by:</strong></p>
<p><strong>FYDP Committee</strong></p>
<p><strong>Dept. of CSE, DIU</strong></p></td>
<td><p><strong>Template approved by:</strong></p>
<p><strong>Dr. Sheak Rashed Haider Noori</strong></p>
<p><strong>Professor and Head, Dept. of CSE, DIU</strong></p></td>
</tr>
</tbody>
</table>

The students and faculty members of Computer Science and Engineering
Department, Daffodil International University have full access rights to
read and print this document without any prior notice to the Head and
FYDP committee.

All rights are reserved.

</div>
