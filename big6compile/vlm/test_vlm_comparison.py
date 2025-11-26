#!/usr/bin/env python3
"""Compare vLLM vs SGLang for Vision-Language Model (Qwen3-VL) inference."""

import time
import requests
import sys
import json
from datetime import datetime
from pathlib import Path

# Test configuration
VLLM_URL = "http://localhost:8006/v1/chat/completions"
SGLANG_URL = "http://localhost:8007/v1/chat/completions"
VLLM_MODEL = "qwen3-vl"
SGLANG_MODEL = "qwen3-vl-sglang"
MAX_TOKENS = 256
NUM_REQUESTS = 3

# Test image URL (using a sample image for testing)
TEST_IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

# Three different ~1500 token system prompts with vision analysis tasks
SYSTEM_PROMPTS = [
    # System prompt 1 - Detailed Image Analysis Expert
    """You are an expert computer vision analyst with deep knowledge in image understanding, object detection, and scene interpretation. Please provide comprehensive visual analysis covering these aspects:

1. Scene composition: Describe the overall scene layout, spatial relationships between objects, perspective, depth cues, and compositional elements including foreground, middle ground, and background layers.

2. Object detection and classification: Identify all significant objects in the image, their categories, approximate counts, positions, sizes relative to the frame, and hierarchical relationships between objects.

3. Visual attributes: Analyze colors, textures, materials, lighting conditions, shadows, highlights, color temperature, saturation levels, and overall color harmony or contrast in the composition.

4. Human analysis (if applicable): Describe any people present including their poses, activities, facial expressions, gestures, clothing, demographics (age range, gender presentation), interactions between individuals, and emotional states.

5. Text recognition: Identify and transcribe any visible text including signs, labels, captions, logos, or written content, noting font styles, sizes, and prominence in the scene.

6. Contextual understanding: Infer the location type, time of day, season, weather conditions, cultural context, and probable purpose or function of the scene captured.

7. Technical assessment: Evaluate image quality including resolution, focus, exposure, white balance, noise levels, motion blur, compression artifacts, and overall photographic technique.

8. Spatial reasoning: Describe relative positions using directional language (left, right, top, bottom, center), distances between objects, occlusions, and three-dimensional spatial understanding.

9. Activity and motion: Identify any implied or actual motion, action sequences, dynamic elements, or temporal aspects suggested by the scene composition.

10. Anomalies and notable features: Point out any unusual, unexpected, or particularly interesting elements that stand out in the image.

11. Aesthetic qualities: Discuss composition rules (rule of thirds, leading lines, symmetry), visual balance, focal points, and artistic qualities of the image.

12. Semantic relationships: Explain relationships between objects, cause-and-effect implications, functional connections, and narrative elements present in the scene.

13. Safety and risk assessment: Identify any potential hazards, unsafe conditions, warning signs, or elements requiring attention from a safety perspective.

14. Comparison and categorization: Compare elements within the image, group similar objects, identify patterns, repetitions, or systematic arrangements.

15. Detailed description synthesis: Provide a holistic understanding that integrates all observed elements into a coherent interpretation of what the image represents and communicates.""",

    # System prompt 2 - Medical/Scientific Image Analysis
    """You are a specialized medical imaging analyst and scientific visualization expert with expertise in interpreting diagnostic images, biological specimens, and scientific data visualizations. Provide thorough analysis covering:

1. Image modality identification: Determine the imaging technique or visualization method used (photograph, X-ray, MRI, CT, microscopy, diagram, chart, etc.) and its typical applications.

2. Anatomical or structural features: If medical/biological, identify organs, tissues, structures, anatomical landmarks, or cellular components visible in the image with proper medical terminology.

3. Pathological indicators: Look for abnormalities, lesions, structural changes, inflammation markers, or other signs of disease or injury if present in medical images.

4. Measurements and quantification: Note any visible scales, rulers, measurements, calibration markers, or quantitative data displayed in the image.

5. Image orientation and planes: For medical images, specify anatomical planes (axial, sagittal, coronal), orientation markers, and viewing perspective.

6. Contrast and enhancement: Describe any contrast agents, staining techniques, fluorescent markers, or image enhancement methods apparent in the visualization.

7. Multi-modal information: If graphs, charts, or data plots, analyze axes, data series, trends, statistical markers, error bars, legends, and data relationships.

8. Technical parameters: Identify imaging settings, resolution indicators, timestamp information, patient/specimen identifiers (if ethically appropriate to mention), and technical metadata.

9. Comparative analysis: If multiple images or time series, describe changes, progressions, comparative features, or temporal evolution of observed phenomena.

10. Diagnostic significance: Assess the clinical or scientific relevance of observed features, potential diagnoses, or research implications (while noting you're not providing medical advice).

11. Quality control: Evaluate image quality for diagnostic or research purposes including artifact presence, positioning accuracy, proper exposure, and technical adequacy.

12. Biological context: For specimens or organisms, identify species, developmental stages, experimental conditions, or ecological context when determinable.

13. Experimental setup: In research images, identify equipment, instruments, experimental apparatus, sample preparation methods, or laboratory conditions visible.

14. Data visualization best practices: For charts/graphs, assess clarity, appropriate chart type selection, data-ink ratio, and effectiveness of visual communication.

15. Integrated interpretation: Synthesize all observations into a coherent scientific or medical interpretation, noting confidence levels and areas of uncertainty.""",

    # System prompt 3 - Document and UI Analysis Expert
    """You are an expert in document analysis, user interface evaluation, and information design with extensive knowledge of typography, layout principles, and human-computer interaction. Analyze images comprehensively:

1. Document type classification: Identify whether the image shows a form, invoice, receipt, contract, presentation slide, webpage, application interface, infographic, or other document type.

2. Layout structure: Describe the hierarchical organization, grid systems, column layouts, sections, headers, footers, margins, and overall spatial organization of content.

3. Typography analysis: Identify fonts, type sizes, weights, styles (bold, italic), alignment (left, right, center, justified), line spacing, and typographic hierarchy.

4. Information architecture: Map the organization of content including navigation elements, menus, breadcrumbs, categories, labels, and information flow.

5. Interactive elements: Identify buttons, links, input fields, dropdowns, checkboxes, radio buttons, sliders, and other UI controls with their states (enabled, disabled, selected).

6. Visual hierarchy: Analyze how size, color, position, and styling create emphasis, guide attention, and establish importance of different elements.

7. Branding elements: Note logos, brand colors, style consistency, corporate identity elements, and adherence to brand guidelines.

8. Data presentation: For tables, forms, or structured data, describe field labels, data types, validation indicators, required fields, and data organization.

9. Accessibility considerations: Evaluate contrast ratios, font sizes, alternative text presence, keyboard navigation indicators, and inclusive design features.

10. Responsive design indicators: Identify breakpoints, mobile adaptations, flexible layouts, or device-specific design elements if apparent.

11. Content extraction: Transcribe key text content including headings, body text, labels, instructions, error messages, and microcopy.

12. Workflow and user flow: Map process steps, wizard progression, form completion sequences, or multi-step interactions shown in the interface.

13. Status indicators: Identify loading states, progress bars, notifications, alerts, success messages, error states, and system feedback elements.

14. Design patterns: Recognize common UI patterns (card layouts, hero sections, modals, sidebars) and assess their implementation quality.

15. Usability heuristics: Evaluate against principles like consistency, error prevention, recognition over recall, flexibility, and aesthetic minimalism, providing a holistic UX assessment."""
]

# Short vision queries - different for each request
USER_QUERIES = [
    "Describe the main subject in this image in 10-15 words.",
    "What is the setting or location shown in 10-15 words?",
    "List the key objects you can identify in 10-15 words."
]

def test_backend(backend_name, url, model_name, system_prompt, user_query):
    """Test a single backend with given prompt."""
    print(f"  Testing {backend_name}...", end=" ", flush=True)
    
    # Construct message with image
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_query},
                    {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}}
                ]
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": True
    }
    
    try:
        start_time = time.time()
        first_token_time = None
        full_response = ""
        tokens_generated = 0
        
        response = requests.post(url, json=payload, stream=True, timeout=180)
        
        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            return None
        
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
                    except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
                        pass
        
        end_time = time.time()
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else 0
        
        # Estimate tokens if not provided
        if tokens_generated == 0 and full_response:
            tokens_generated = max(1, len(full_response) // 4)
        
        tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
        
        print(f"✅ TTFT: {ttft:.3f}s | Total: {total_time:.2f}s | {tokens_per_second:.2f} tok/s")
        
        return {
            "backend": backend_name,
            "time": total_time,
            "ttft": ttft,
            "tokens": tokens_generated,
            "tps": tokens_per_second,
            "response": full_response,
            "query": user_query
        }
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def main():
    print(f"\n{'='*80}")
    print(f"VLM Inference Comparison: vLLM vs SGLang")
    print(f"Model: Qwen3-VL-30B-A3B-Instruct")
    print(f"{'='*80}")
    print(f"vLLM endpoint:   {VLLM_URL}")
    print(f"SGLang endpoint: {SGLANG_URL}")
    print(f"Test image:      {TEST_IMAGE_URL}")
    print(f"Running {NUM_REQUESTS} rounds with ~1500 token system prompts")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for i in range(NUM_REQUESTS):
        round_label = "COLD START" if i == 0 else f"WARM RUN {i}"
        print(f"Round {i+1}/{NUM_REQUESTS} ({round_label}):")
        
        system_prompt = SYSTEM_PROMPTS[i]
        user_query = USER_QUERIES[i]
        
        # Test vLLM
        vllm_result = test_backend("vLLM", VLLM_URL, VLLM_MODEL, system_prompt, user_query)
        if vllm_result:
            all_results.append(vllm_result)
        
        # Small delay between backends
        time.sleep(1)
        
        # Test SGLang
        sglang_result = test_backend("SGLang", SGLANG_URL, SGLANG_MODEL, system_prompt, user_query)
        if sglang_result:
            all_results.append(sglang_result)
        
        print()
    
    # Analyze results
    print(f"{'─'*80}")
    print(f"DETAILED RESULTS:")
    print(f"{'─'*80}")
    
    vllm_results = [r for r in all_results if r['backend'] == 'vLLM']
    sglang_results = [r for r in all_results if r['backend'] == 'SGLang']
    
    for i in range(NUM_REQUESTS):
        round_label = "🥶 COLD" if i == 0 else "🔥 WARM"
        print(f"\nRound {i+1} ({round_label}):")
        
        if i < len(vllm_results):
            r = vllm_results[i]
            print(f"  vLLM:   TTFT: {r['ttft']:6.3f}s | Total: {r['time']:6.2f}s | {r['tps']:6.2f} tok/s | Tokens: {r['tokens']}")
            print(f"  Response: {r['response'][:100]}...")
        
        if i < len(sglang_results):
            r = sglang_results[i]
            print(f"  SGLang: TTFT: {r['ttft']:6.3f}s | Total: {r['time']:6.2f}s | {r['tps']:6.2f} tok/s | Tokens: {r['tokens']}")
            print(f"  Response: {r['response'][:100]}...")
    
    # Calculate statistics
    print(f"\n{'─'*80}")
    print(f"STATISTICS:")
    print(f"{'─'*80}")
    
    if vllm_results:
        vllm_avg_ttft = sum(r['ttft'] for r in vllm_results) / len(vllm_results)
        vllm_avg_ttft_warm = sum(r['ttft'] for r in vllm_results[1:]) / max(1, len(vllm_results[1:]))
        vllm_avg_tps = sum(r['tps'] for r in vllm_results) / len(vllm_results)
        
        print(f"\nvLLM:")
        print(f"  Average TTFT (all):  {vllm_avg_ttft:.3f}s")
        print(f"  Average TTFT (warm): {vllm_avg_ttft_warm:.3f}s")
        print(f"  Average throughput:  {vllm_avg_tps:.2f} tok/s")
    
    if sglang_results:
        sglang_avg_ttft = sum(r['ttft'] for r in sglang_results) / len(sglang_results)
        sglang_avg_ttft_warm = sum(r['ttft'] for r in sglang_results[1:]) / max(1, len(sglang_results[1:]))
        sglang_avg_tps = sum(r['tps'] for r in sglang_results) / len(sglang_results)
        
        print(f"\nSGLang:")
        print(f"  Average TTFT (all):  {sglang_avg_ttft:.3f}s")
        print(f"  Average TTFT (warm): {sglang_avg_ttft_warm:.3f}s")
        print(f"  Average throughput:  {sglang_avg_tps:.2f} tok/s")
    
    if vllm_results and sglang_results:
        print(f"\nComparison (Warm runs):")
        ttft_diff = ((sglang_avg_ttft_warm - vllm_avg_ttft_warm) / vllm_avg_ttft_warm) * 100
        tps_diff = ((sglang_avg_tps - vllm_avg_tps) / vllm_avg_tps) * 100
        
        print(f"  TTFT: SGLang is {abs(ttft_diff):.1f}% {'faster' if ttft_diff < 0 else 'slower'} than vLLM")
        print(f"  Throughput: SGLang is {abs(tps_diff):.1f}% {'faster' if tps_diff > 0 else 'slower'} than vLLM")
    
    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "vlm_comparison",
        "model": "Qwen3-VL-30B-A3B-Instruct",
        "test_image": TEST_IMAGE_URL,
        "max_tokens": MAX_TOKENS,
        "num_requests": NUM_REQUESTS,
        "results": all_results,
        "statistics": {
            "vllm": {
                "avg_ttft_all": vllm_avg_ttft if vllm_results else None,
                "avg_ttft_warm": vllm_avg_ttft_warm if vllm_results else None,
                "avg_tps": vllm_avg_tps if vllm_results else None
            } if vllm_results else None,
            "sglang": {
                "avg_ttft_all": sglang_avg_ttft if sglang_results else None,
                "avg_ttft_warm": sglang_avg_ttft_warm if sglang_results else None,
                "avg_tps": sglang_avg_tps if sglang_results else None
            } if sglang_results else None
        }
    }
    
    output_file = f"/compile/vlm/eval_vlm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()







