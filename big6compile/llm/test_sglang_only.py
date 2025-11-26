#!/usr/bin/env python3
"""Quick SGLang inference test - 3 requests to account for cold start."""

import time
import requests
import sys
import json
from datetime import datetime

# Three different ~1500 token system prompts for each request
# Each prompt shares 50% common prefix, then 50% unique content
SYSTEM_PROMPTS = [
    # System prompt 1 - Quantum Physics Expert
    """You are an expert educator teaching advanced physics concepts. Please provide a comprehensive, detailed explanation of quantum entanglement that covers the following aspects:

1. Historical background: Describe how quantum entanglement was first theorized by Einstein, Podolsky, and Rosen in their famous EPR paradox paper of 1935, and how they viewed it as a troubling aspect of quantum mechanics that suggested the theory might be incomplete.

2. The basic concept: Explain what quantum entanglement means at a fundamental level - how two or more particles can become correlated in such a way that the quantum state of each particle cannot be described independently, even when separated by large distances.

3. Bell's Theorem: Discuss John Stewart Bell's groundbreaking 1964 theorem and how experimental tests of Bell inequalities have consistently supported quantum mechanics over local hidden variable theories, confirming that entanglement represents genuine nonlocal correlations.

4. Measurement and correlation: Explain in detail what happens when we measure one particle of an entangled pair - how does the measurement instantly affect the state of the other particle, and why this doesn't violate special relativity's prohibition on faster-than-light communication?

5. Creating entanglement: Describe several methods by which entangled states can be created in laboratory settings, such as spontaneous parametric down-conversion in nonlinear crystals, or through interactions in atomic systems.

6. Applications: Discuss practical applications including quantum cryptography and quantum key distribution protocols like BB84 and E91, quantum teleportation experiments, quantum computing applications where entanglement enables quantum algorithms to outperform classical ones, and quantum sensing technologies.

7. Current research: Mention recent developments in maintaining entanglement over longer distances, efforts toward quantum networks and the quantum internet, and experiments with increasingly complex entangled systems.

Please make your explanation thorough, scientifically accurate, and accessible to someone with undergraduate-level physics knowledge. Include relevant equations where appropriate and use concrete examples to illustrate abstract concepts.

8. Mathematical formalism: Provide the mathematical description of entangled states using Dirac notation. For example, explain the Bell states for two qubits, such as the singlet state |Ψ⟩ = (1/√2)(|01⟩ - |10⟩), and describe how these maximally entangled states form a complete basis. Discuss density matrices and how they're used to describe mixed versus pure entangled states, and explain partial trace operations for describing subsystems of entangled pairs.

9. Types of entanglement: Distinguish between different forms of entanglement including bipartite versus multipartite entanglement, continuous variable versus discrete variable entanglement, and discuss measures of entanglement such as concurrence, entanglement entropy (von Neumann entropy), negativity, and the logarithmic negativity. Explain GHZ states and W states as examples of different types of multipartite entanglement with distinct properties.

10. Decoherence and entanglement loss: Explain how interaction with the environment leads to decoherence and the degradation of entanglement over time. Discuss the challenges this poses for quantum information processing and describe strategies to combat decoherence including dynamical decoupling, quantum error correction codes, and decoherence-free subspaces.

11. Experimental milestones: Review major experimental achievements such as Alain Aspect's experiments in the 1980s that closed important loopholes in Bell tests, the first quantum teleportation demonstrations by Anton Zeilinger's group and others in the 1990s, long-distance entanglement distribution via satellite demonstrated by the Chinese Micius satellite, and recent loophole-free Bell tests that simultaneously closed the locality and detection loopholes.

12. Philosophical implications: Address the interpretation questions that entanglement raises about the nature of reality, locality, and realism. Discuss how different interpretations of quantum mechanics (Copenhagen, many-worlds, pilot wave theory, etc.) handle entanglement and nonlocality differently, and explain why entanglement challenges classical intuitions about separability and independence of distant objects.

13. Quantum correlations vs classical correlations: Provide a clear distinction between quantum correlations arising from entanglement and classical correlations. Use examples like correlated classical bits versus entangled qubits to illustrate why quantum correlations are fundamentally different and more powerful, enabling violations of classical bounds like the CHSH inequality.

14. Entanglement in quantum field theory: Briefly discuss how entanglement appears in the context of quantum field theory, including concepts like the Reeh-Schlieder theorem, entanglement between vacuum fluctuations in different regions of space, and connections to black hole physics through the study of entanglement entropy in the Hawking radiation problem.

15. Future prospects: Speculate on future directions including room-temperature quantum computing enabled by better preservation of entanglement, global quantum communication networks, quantum-enhanced sensing reaching fundamental precision limits, and potential discoveries about the role of entanglement in fundamental physics, cosmology, and our understanding of spacetime itself through the ER=EPR conjecture and holographic principles.""",

    # System prompt 2 - Molecular Biology Expert
    """You are a distinguished molecular biologist and biochemistry professor with decades of research experience. Please provide an in-depth, comprehensive explanation of the CRISPR-Cas9 gene editing system covering these critical areas:

1. Discovery and history: Trace the discovery of CRISPR sequences in bacterial genomes in 1987 by Yoshizumi Ishino, their recognition as a bacterial immune system by Francisco Mojica in the early 2000s, and the groundbreaking 2012 paper by Jennifer Doudna and Emmanuelle Charpentier demonstrating CRISPR-Cas9 as a programmable gene editing tool that revolutionized molecular biology and earned them the 2020 Nobel Prize in Chemistry.

2. Molecular mechanism: Explain in detail how the CRISPR-Cas9 system works at the molecular level - describe the roles of the guide RNA (gRNA), the Cas9 endonuclease protein, the protospacer adjacent motif (PAM) sequence requirement, how the gRNA directs Cas9 to specific DNA sequences through Watson-Crick base pairing, and how Cas9's RuvC and HNH nuclease domains create double-strand breaks in the target DNA.

3. Repair pathways: Discuss the two main DNA repair mechanisms that cells employ after Cas9 creates a double-strand break - non-homologous end joining (NHEJ) which often results in small insertions or deletions (indels) that can knock out gene function, and homology-directed repair (HDR) which can be exploited to insert specific new sequences when a donor template is provided, along with the relative frequencies and cellular contexts where each pathway dominates.

4. Guide RNA design: Explain the principles of designing effective guide RNAs, including the typical 20-nucleotide target sequence, considerations for minimizing off-target effects, the importance of GC content, avoiding secondary structures, seed sequence specificity, and computational tools like CRISPOR, Benchling, and others used to predict on-target efficiency and potential off-target sites across the genome.

5. CRISPR variants: Describe the diverse CRISPR systems beyond Cas9 including Cas12a (Cpf1) with its different PAM requirements and staggered cut pattern, Cas13 for RNA targeting, miniature Cas proteins like CasΦ and Cas12f for easier delivery, and the development of catalytically dead Cas9 (dCas9) fused to various effector domains for transcriptional activation (CRISPRa), repression (CRISPRi), base editing, and epigenetic modifications without cutting DNA.

6. Base and prime editing: Explain how base editors (CBEs and ABEs) enable precise single-nucleotide changes without creating double-strand breaks by fusing deaminase enzymes to nickase Cas9, and describe prime editing as a "search-and-replace" technology using a prime editing guide RNA (pegRNA) and reverse transcriptase to install precise edits including insertions, deletions, and all possible base-to-base conversions at target sites.

7. Delivery methods: Discuss various strategies for delivering CRISPR components into cells including plasmid DNA transfection, viral vectors (AAV, lentivirus), electroporation, lipid nanoparticles (like those used in some therapeutic applications), and ribonucleoprotein (RNP) complexes of Cas9 protein pre-complexed with gRNA which offer advantages of rapid action and reduced off-target effects due to transient presence in cells.

8. Off-target effects: Address the challenge of unintended edits at genomic sites that share partial homology with the target sequence, methods to detect off-targets including GUIDE-seq, CIRCLE-seq, and whole-genome sequencing, and strategies to minimize off-targets such as using high-fidelity Cas9 variants (SpCas9-HF1, eSpCas9, HypaCas9), truncated gRNAs, and careful guide design.

9. Therapeutic applications: Review current clinical trials and approved therapies using CRISPR including treatments for sickle cell disease and beta-thalassemia (CTX001/exagamglogene autotemcel), cancer immunotherapies editing T cells, inherited blindness (Leber congenital amaurosis 10), and ongoing research for treating HIV, muscular dystrophies, cystic fibrosis, and other genetic disorders.

10. Agricultural applications: Discuss how CRISPR is being used to improve crop yields, enhance nutritional content, confer disease and pest resistance, improve drought tolerance, reduce agricultural chemical requirements, and how gene-edited crops differ from traditional GMOs in regulatory frameworks in various countries, with examples like non-browning mushrooms and high-amylose wheat.

11. Research applications: Explain how CRISPR has become an indispensable research tool for creating knockout cell lines and animal models, large-scale genetic screens to identify gene functions, studying gene regulation, modeling diseases, and investigating fundamental biological processes across all domains of life from bacteria to humans.

12. Ethical considerations: Address the profound ethical questions raised by CRISPR technology including the 2018 controversy around He Jiankui's editing of human embryos, the distinction between somatic and germline editing, concerns about eugenics and designer babies, equitable access to gene therapies, ecological impacts of gene drives, and the need for robust governance frameworks and public engagement.

13. Future directions: Speculate on emerging developments including improved delivery systems for in vivo editing, multiplexed editing of many genes simultaneously, expanded targeting range with new PAM variants and Cas proteins, RNA editing for reversible therapies, mitochondrial genome editing, potential cures for currently intractable genetic diseases, and the convergence of CRISPR with other technologies like synthetic biology and artificial intelligence for rational genome design.

14. Technical challenges: Discuss remaining hurdles including efficient delivery to specific tissues and cell types in vivo, immunogenicity of bacterial Cas proteins, achieving sufficiently high editing rates for therapeutic efficacy, controlling the balance between NHEJ and HDR, editing in non-dividing cells, and the substantial differences between editing cultured cells versus complex organisms.

15. Global impact: Reflect on how CRISPR technology is democratizing genetic engineering by making it accessible to smaller labs and researchers worldwide, accelerating biological research and drug discovery, creating new biotechnology companies and industries, and potentially transforming medicine from treating symptoms to curing genetic root causes of disease, while also raising important questions about biosecurity and dual-use research.""",

    # System prompt 3 - Computer Science Expert  
    """You are a renowned computer scientist and software architect with expertise in distributed systems, algorithms, and machine learning. Please provide a thorough, technically rigorous explanation of the Transformer architecture and attention mechanisms that covers these essential topics:

1. Historical context: Describe the evolution of sequence modeling in deep learning from recurrent neural networks (RNNs) and Long Short-Term Memory (LSTM) networks, the limitations these architectures faced including vanishing gradients and inability to parallelize training across sequence positions, and how the 2017 "Attention Is All You Need" paper by Vaswani et al. introduced the Transformer architecture that revolutionized natural language processing and beyond.

2. Self-attention mechanism: Explain in mathematical detail how self-attention works - describe the computation of queries (Q), keys (K), and values (V) from input embeddings through learned linear transformations, the scaled dot-product attention formula Attention(Q,K,V) = softmax(QK^T/√d_k)V, why the scaling factor √d_k is necessary, and how attention weights represent learned relationships between different positions in the sequence.

3. Multi-head attention: Discuss why using multiple attention heads in parallel is beneficial, how each head can learn to attend to different aspects of the input (e.g., syntactic vs semantic relationships), the computational implementation where d_model dimensions are split across h heads each with dimension d_k = d_model/h, and how outputs from all heads are concatenated and projected to produce the final multi-head attention output.

4. Positional encoding: Explain why Transformers need explicit positional information since self-attention is permutation-invariant, describe the sinusoidal positional encoding scheme using sin and cos functions of different frequencies (PE(pos,2i) = sin(pos/10000^(2i/d_model))), discuss alternative approaches like learned positional embeddings and relative positional encodings, and how positional information enables the model to utilize sequence order.

5. Encoder architecture: Detail the structure of Transformer encoder blocks including the multi-head self-attention sublayer, position-wise feed-forward networks (two linear transformations with ReLU/GELU activation), residual connections around each sublayer, layer normalization, and how multiple encoder blocks are stacked (typically 6-12 layers in the original paper, up to 96+ in large language models) to build increasingly abstract representations.

6. Decoder architecture: Explain the decoder's structure with its masked self-attention to prevent positions from attending to future positions during training, cross-attention over encoder outputs to incorporate source sequence information, the same feed-forward and normalization components as the encoder, and how autoregressive generation works during inference where outputs are fed back as inputs one token at a time.

7. Training dynamics: Discuss key training aspects including the use of teacher forcing where ground truth tokens rather than model predictions are fed to the decoder during training, the cross-entropy loss function typically used for next-token prediction, learning rate schedules like the warmup and decay approach in the original paper, regularization techniques including dropout and attention dropout, and typical training data requirements and computational costs.

8. Attention patterns and interpretability: Describe research into what different attention heads learn such as heads that focus on syntactic dependencies, positional patterns, or semantic relationships, visualization techniques for attention weights, the ongoing debate about whether attention weights provide faithful explanations of model behavior, and tools like BertViz and attention rollout for analyzing attention patterns.

9. Computational complexity: Analyze the O(n²d) time and space complexity of self-attention where n is sequence length and d is model dimension, explain why this quadratic scaling limits context lengths, and discuss how this compares to the O(nd²) complexity of recurrent models which have linear sequence length scaling but cannot parallelize across time steps.

10. Efficient Transformers: Survey the extensive research on reducing the quadratic complexity including sparse attention patterns (local, strided, fixed patterns), low-rank approximations (Linformer), kernel-based approaches (Performers, Linear Transformers), recurrent mechanisms (Transformer-XL), and the FlashAttention algorithm which optimizes attention computation for GPU memory hierarchy, along with tradeoffs between efficiency and model quality.

11. Pretrain and fine-tune paradigm: Explain how Transformers enabled the dominant paradigm of pretraining large models on massive unlabeled text corpora using self-supervised objectives like masked language modeling (BERT) or causal language modeling (GPT), followed by fine-tuning on downstream tasks, and how this approach achieves strong performance across diverse NLP tasks with relatively little task-specific data.

12. Beyond NLP applications: Discuss how the Transformer architecture has been successfully adapted beyond text including Vision Transformers (ViT) for image classification by treating image patches as tokens, audio processing with models like Whisper, protein structure prediction in AlphaFold2, reinforcement learning in Decision Transformers, multimodal models like CLIP and GPT-4 that process both images and text, and time series forecasting.

13. Scaling laws: Review empirical findings about how Transformer performance scales with model size (parameters), dataset size, and compute budget, the work by Kaplan et al. and Hoffmann et al. (Chinchilla) on optimal allocation of compute between model size and training data, the emergence of new capabilities at scale, and implications for training large language models efficiently.

14. Modern variants: Describe recent architectural innovations including RoPE (Rotary Position Embeddings), ALiBi (Attention with Linear Biases), SwiGLU activation functions, parallel attention and feedforward computation, removal of biases in linear layers, RMSNorm instead of LayerNorm, grouped-query attention for efficient inference, and sliding window attention for longer contexts as seen in models like LLaMA, Mistral, and others.

15. Future directions: Speculate on ongoing research including mixture-of-experts architectures for efficiently scaling to trillions of parameters, state-space models and structured state space sequences (S4, Mamba) as potential alternatives or complements to attention, in-context learning and few-shot capabilities, alignment techniques, reducing inference costs, extending context lengths to millions of tokens, and the potential for Transformers or their successors to achieve artificial general intelligence."""
]

URL = "http://localhost:8083/v1/chat/completions"
MODEL = "Qwen3-235B-A22B-Instruct-FP8"
MAX_TOKENS = 256
NUM_REQUESTS = 3

# Short user queries - different for each request
USER_QUERIES = [
    "Summarize quantum entanglement in 10-15 words.",
    "Explain CRISPR gene editing briefly in 10-15 words.",
    "Describe Transformer architecture concisely in 10-15 words."
]

print(f"\n{'='*80}")
print(f"Testing SGLang - Qwen3-235B-A22B-Instruct-FP8")
print(f"{'='*80}")
print(f"Endpoint: {URL}")
print(f"Running {NUM_REQUESTS} requests with different ~1500 token system prompts")
print(f"Each request uses 50% shared + 50% unique system prompt content")
print(f"{'='*80}\n")

results = []

try:
    for i in range(NUM_REQUESTS):
        print(f"Request {i+1}/{NUM_REQUESTS}...", end=" ", flush=True)
        
        # Use different system prompt and user query for each request
        # System prompts are 50% shared (first half) + 50% unique (second half)
        system_prompt = SYSTEM_PROMPTS[i]
        user_query = USER_QUERIES[i]
        
        # Use streaming to measure TTFT
        payload_stream = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
            "stream": True
        }
        
        start_time = time.time()
        first_token_time = None
        full_response = ""
        tokens_generated = 0
        
        response = requests.post(URL, json=payload_stream, stream=True, timeout=120)
        
        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            sys.exit(1)
        
        # Stream response to measure TTFT
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if first_token_time is None and 'choices' in chunk and len(chunk['choices']) > 0:
                            choice = chunk['choices'][0]
                            delta = choice.get('delta') if choice else None
                            if delta and delta.get('content') and len(delta.get('content', '').strip()) > 0:
                                first_token_time = time.time()
                        
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            choice = chunk['choices'][0]
                            delta = choice.get('delta') if choice else None
                            if delta:
                                content = delta.get('content', '')
                                if content:
                                    full_response += content
                        
                        if 'usage' in chunk and chunk['usage']:
                            tokens_generated = chunk['usage'].get('completion_tokens', 0)
                    except (json.JSONDecodeError, AttributeError, KeyError, TypeError) as e:
                        pass
        
        end_time = time.time()
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else 0
        
        # If tokens_generated is 0, estimate from response length
        # Rough approximation: ~4 chars per token for English text
        if tokens_generated == 0 and full_response:
            tokens_generated = max(1, len(full_response) // 4)
        
        tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
        
        results.append({
            "time": total_time,
            "ttft": ttft,
            "tokens": tokens_generated,
            "tps": tokens_per_second,
            "response": full_response,
            "query": user_query
        })
        
        print(f"✅ {total_time:.2f}s | TTFT: {ttft:.3f}s | {tokens_per_second:.2f} tok/s")
    
    # Show detailed results
    print(f"\n{'─'*80}")
    print(f"DETAILED RESULTS:")
    print(f"{'─'*80}")
    for i, r in enumerate(results, 1):
        label = "🥶 COLD START" if i == 1 else "🔥 WARM"
        print(f"{i}. {label:12} | Time: {r['time']:6.2f}s | TTFT: {r['ttft']:6.3f}s | Tokens/s: {r['tps']:6.2f} | Tokens: {r['tokens']}")
        print(f"   Query: {r['query']}")
        print(f"   Response: {r['response']}")
    
    # Calculate averages
    print(f"\n{'─'*80}")
    print(f"STATISTICS:")
    print(f"{'─'*80}")
    
    avg_time_all = sum(r['time'] for r in results) / len(results)
    avg_tps_all = sum(r['tps'] for r in results) / len(results)
    
    avg_time_warm = sum(r['time'] for r in results[1:]) / len(results[1:])
    avg_tps_warm = sum(r['tps'] for r in results[1:]) / len(results[1:])
    avg_ttft_all = sum(r['ttft'] for r in results) / len(results)
    avg_ttft_warm = sum(r['ttft'] for r in results[1:]) / len(results[1:])
    
    print(f"  All requests (including cold start):")
    print(f"    Average time:     {avg_time_all:.2f}s")
    print(f"    Average TTFT:     {avg_ttft_all:.3f}s")
    print(f"    Average tokens/s: {avg_tps_all:.2f}")
    print(f"\n  Warm requests only (excluding cold start):")
    print(f"    Average time:     {avg_time_warm:.2f}s")
    print(f"    Average TTFT:     {avg_ttft_warm:.3f}s")
    print(f"    Average tokens/s: {avg_tps_warm:.2f}")
    
    # Show sample response
    print(f"\n{'─'*80}")
    print(f"SAMPLE RESPONSE (Request 1):")
    print(f"{'─'*80}")
    print(results[0]['response'][:500] + "..." if len(results[0]['response']) > 500 else results[0]['response'])
    
    # Save results to file
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "backend": "sglang",
        "model": MODEL,
        "endpoint": URL,
        "test_description": "~1500 token system prompts (50% shared prefix + 50% unique), requesting 10-15 word responses",
        "max_tokens": MAX_TOKENS,
        "num_requests": NUM_REQUESTS,
        "results": results,
        "statistics": {
            "all_requests": {
                "avg_time": avg_time_all,
                "avg_ttft": avg_ttft_all,
                "avg_tokens_per_second": avg_tps_all
            },
            "warm_requests": {
                "avg_time": avg_time_warm,
                "avg_ttft": avg_ttft_warm,
                "avg_tokens_per_second": avg_tps_warm
            }
        }
    }
    
    output_file = f"/compile/llm/eval_sglang_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    sys.exit(1)

