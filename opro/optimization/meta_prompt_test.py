import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from opro.prompt_utils import call_openai_server_single_prompt

prompt= """Your task is to generate the Task <INS>. Below are some previous Tasks with their scores. The score ranges from 0 to 100.

# Task: Given the reviews (Text), answer if a paper would be accepted (Yes) or not (No) by an academic conference.
Score:
65



Below are some examples.


# Task: <INS>
# Text: REVIEW 
Summary:
This paper proposes Lever-LM, a small language model designed to configure effective in-context demonstration (ICD) for improving the in-context learning performance of large vision-language models. The authors construct a dataset of effective ICD sequences to train Lever-LM, which then generates new ICD configurations for novel queries to solve vision-language tasks through in-context learning. Experiments on image captioning and VQA tasks demonstrate that Lever-LM outperforms strong baselines and can capture statistical patterns in ICD sequences to effectively leverage LVLMs.

Soundness:
3: good

Presentation:
3: good

Contribution:
2: fair

Strengths:
- The paper introduces an interesting method using a small language model (Lever-LM) to configure in-context demonstrations for LVLMs.  Empirical results on image Captioning and VQA demonstrate the effectiveness of Lever-LM across these settings.
- The paper includes a wide range of ablation studies exploring various aspects of the proposed method, such as different configurations for constructing the training dataset, different model architectures, and the impact of ICD sequence length. These studies provide valuable insights into the factors affecting the performance of Lever-LM.
- The paper is easy to understand.

Weaknesses:
- In practice, we use ICL because we do not want to update the model parameters or use more computing, while this approach requires training a small model for ICD selection. Thus, the authors need to show that this method can generalize to more models (e.g., Qwen-VL, InternLM-XComposer2, IDEFICS2, GPT4, etc) and new tasks beyond VQA and captioning (e.g., [1]), and therefore show whether the additional computations worth it.
- Zero-shot performance should also be shown as a reference to few-shot performance in Table 1, 2, 3, and more.
- Lever-LM is trained on 2-shot and the authors show the extrapolation to up to 8 shots, and the performance is almost constant wrt. to the number of shots for extrapolation except for OF IC. This may indicate the small number of shots during training makes this strategy hard to generalize to more shots beyond, which limits more performance gain and applications such as many-shot ICL.

[1] VL-ICL Bench: The Devil in the Details of Benchmarking Multimodal In-Context Learning. arXiv, 2024.

Limitations:
See above.

Rating:
5: marginally below the acceptance threshold

Confidence:
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

REVIEW 
Summary:
This paper presents a novel approach called Lever LM, which uses a tiny language model to configure effective ICD sequences for LVLMs. The key innovation is leveraging the small Lever LM to select and order ICDs, improving the ICL performance of LVLMs on tasks like VQA and image captioning. The paper demonstrates that Lever LM captures statistical patterns in ICD configurations, leading to significant performance gains over existing methods.

Soundness:
3: good

Presentation:
3: good

Contribution:
4: excellent

Strengths:
This paper proposes an innovative approach: training Lever LM to select ICD for in-context learning. 

Treating ICD selection as a sequence generation task is novel. The experimental results also demonstrate the effectiveness of this modeling approach. Compared to heuristic ICD selection methods, Lever LM demonstrates greater robustness, significantly outperforming other selection methods in ICL across multiple models and tasks, achieving optimal performance. 

In addition, the method shows the surprising ability to perform k-shot learning despite being trained in a 2-shot setting. The golden fixed set is an interesting phenomenon, which points that finding golden fixed set may become an important research direction in the future.

Weaknesses:
1.      The model needs to be encoded with CLIP first. It's seemed CLIP is also a big model compare than Lever LM. 

2.      There is no comparison with traditional ICDs ranking methods.

Limitations:
N/A

Rating:
8: accept, good paper

Confidence:
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

REVIEW 
Summary:
This paper proposes using a Tiny Lever-LM to assist in ICD selection for LVLM's ICL scenarios, thereby enhancing ICL performance without significantly increasing computational costs. Lever-LM unifies the modeling of multiple scenarios (VQA, IC) in complex multimodal ICL, eliminating the need for manually designed heuristic ICD selection strategies. Additionally, Lever-LM jointly learns ICD selection and ordering, achieving end-to-end learning of ICD sequences. Lever-LM achieves excellent performance across multiple LVLM models, significantly outperforming existing multimodal ICD heuristic selection methods.

Soundness:
3: good

Presentation:
3: good

Contribution:
3: good

Strengths:
1. Lever-LM's structure uses only two layers of Transformer Decoder, making it a highly efficient model for generating ICD sequences.

2. Lever-LM unifies the modeling of complex multimodal ICL scenarios, eliminating the need for manually designed heuristic ICD selection strategies. Furthermore, Lever-LM simultaneously models both ICD selection and ordering steps, achieving end-to-end optimization of ICD sequence modeling. Besides, It is the first attempt to establish the ICD selection and sorting task as a generation task. 

3. Lever-LM demonstrates robustness on multiple levels. First, the authors test it across various tasks and models, consistently outperforming the best ICD selection methods in the current multimodal context. Meanwhile, manually designed heuristic ICD selection strategies show significant performance fluctuations across different models. Additionally, they evaluate the use of different metrics for constructing ICD sequence datasets and experiment with different language model architectures, all of which show excellent performance.

4. The authors validate that the sequence order generated by Lever-LM performs better than randomly generated sequence orders, proving that Lever-LM indeed learns the order information of ICD sequences.

5. Lever-LM demonstrates interesting length extrapolation ability after being trained on a 2-shot ICD sequence dataset. It performs strongly even when generating 4-shot, 6-shot, and 8-shot sequences. This proves the efficiency of the Lever LM method: it requires only short-shot ICD sequences to exhibit excellent performance on long-shot ICL.

Weaknesses:
1. It has not been explored whether the performance of Lever-LM strongly depends on its size. 

2. It seems like LLMs can also use this method for ICL tasks. It may be better to evaluate Lever-LM in LLMs to demonstrate its versatility.

Limitations:
Please see the Weaknesses.

Rating:
8: accept, good paper

Confidence:
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

REVIEW 
Summary:
The authors focus on configuring effective in-context demonstration (ICD) sequences to improve the In-Context Learinng (ICL) performance of LVLMs. The proposed Lever-LM enables the step-by-step generation of ICD configurations and simultaneously considers the selection of ICDs and the ordering of ICD sequences. Experimental results validate that Lever-LM can capture the statistical patterns for levering LVLMs compared with similarity-based retrieval methods.

Soundness:
2: fair

Presentation:
2: fair

Contribution:
2: fair

Strengths:
The tokens of the vocabulary in Lever-LM are the samples from the supporting set. Given the query sample, the ICDs can be selected one by one based on the token distribution produced by the trained Lever-LM.

Weaknesses:
1. In lines 196-197, what is the reason for “randomly” choosing samples to build the sub-supporting set? I think this method is suboptimal, personally.
2. Table 2 (15) uses fixed ICD configuration for any query input, and still returns significantly improved performance compared with other methods in IC. I think this phenomenon shows that the ICL method fails to work on IC. Or I am curious about the performance of “randomly” selected fixed ICD configuration compared with model selected.
3. The experiments part is not convincing in the field of MLLMs. I hope to see results with more benchmarks and models (e.g., M-ICL[1]). To assess the generalization capability of their approach, it would be advantageous for the authors to evaluate their methodology using benchmarks that emphasize fine-grained analysis, such as MM-Vet and SEED-Bench, particularly focusing on segments that require in-depth evaluation.
4. I think the performance between two order strategies are comparative in Table 4, personally. 

[1] What Makes Multimodal In-Context Learning Work?, CVPR 2024.

Limitations:
See weakness

Rating:
3: reject, not good enough

Confidence:
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.


# Answer: Yes



Generate a Task that is different from all the Tasks <INS> above, and has a higher score than all the Tasks <INS> above. The Task should begin with <INS> and end with </INS>. The Task should be concise, effective, and generally applicable to all problems above.

"""

response = call_openai_server_single_prompt(prompt, "gpt-4o-mini", temperature=0.7)
print(response)
